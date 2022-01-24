import copy
import logging
# Set up logging:
logger = logging.getLogger()

from omegaconf import DictConfig

import tensorflow as tf
import numpy


from mlqm.hamiltonians import Hamiltonian
from mlqm.optimization import GradientCalculator
from mlqm.samplers     import Estimator, MetropolisSampler

from mlqm import MPI_AVAILABLE, MAX_PARALLEL_ITERATIONS, DEFAULT_TENSOR_TYPE



if MPI_AVAILABLE:
    import horovod.tensorflow as hvd

from .BaseAlgorithm import BaseAlgorithm

class AdamOptimizer(BaseAlgorithm):

    def __init__(self,
            sampler                   : MetropolisSampler,
            wavefunction              : callable,
            adaptive_wavefunction     : callable,
            hamiltonian               : Hamiltonian,
            optimizer_config          : DictConfig,
            sampler_config            : DictConfig,
        ):

        BaseAlgorithm.__init__(self,
            sampler, 
            wavefunction, 
            adaptive_wavefunction, 
            hamiltonian,
            optimizer_config,
            sampler_config)




    def compute_updates_and_metrics(self, current_psi):
    
        eps = self.optimizer_config.epsilon
        delta = self.optimizer_config.delta
        delta_p, opt_metrics, next_energy = self.flat_optimizer(current_psi, eps, delta)

        return delta_p, opt_metrics, next_energy


    def flat_optimizer(self, current_psi, eps, delta):

        dpsi_i    = self.estimator['dpsi_i']
        energy    = self.estimator["energy"]
        dpsi_i_EL = self.estimator["dpsi_i_EL"]
        dpsi_ij   = self.estimator["dpsi_ij"]

        dp_i, S_ij =  self.compute_gradients(dpsi_i, energy, dpsi_i_EL, dpsi_ij, eps)

        dp_i = delta * dp_i

        # Unpack the gradients
        gradients = self.unflatten_weights_or_gradients(self.flat_shape, self.correct_shape, dp_i)

        original_weights = self.wavefunction.trainable_variables

        # Even though it's a flat optimization, we recompute the energy to get the overlap too:
        loop_items = zip(self.adaptive_wavefunction.trainable_variables, original_weights, gradients)
        for weight, original_weight, gradient in loop_items:
            weight.assign(original_weight + gradient)

        # Compute the new energy:
        next_energy, overlap, acos = self.recompute_energy(self.adaptive_wavefunction, current_psi)

        # Compute the parameter distance:
        par_dist = self.gradient_calc.par_dist(dp_i, S_ij)
        ratio    = tf.abs(par_dist - acos) / tf.abs(par_dist + acos+ 1e-8)


        delta_metrics = {
            "optimizer/delta"   : delta,
            "optimizer/eps"     : eps,
            "optimizer/overlap" : overlap,
            "optimizer/par_dist": par_dist,
            "optimizer/acos"    : acos,
            "optimizer/ratio"   : ratio
        }

        return gradients,  delta_metrics, next_energy

