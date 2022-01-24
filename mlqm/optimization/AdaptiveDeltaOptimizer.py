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

class AdaptiveDeltaOptimizer(BaseAlgorithm):

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
        delta_p, opt_metrics, next_energy = self.optimize_delta(current_psi, eps)
        
        return delta_p, opt_metrics, next_energy

    def optimize_delta(self, current_psi, eps):

        # Get the natural gradients and S_ij
        f_i = self.gradient_calc.f_i(
                self.estimator['dpsi_i'],
                self.estimator["energy"],
                self.estimator["dpsi_i_EL"]
                )

        S_ij = self.gradient_calc.S_ij(
                self.estimator['dpsi_ij'],
                self.estimator['dpsi_i']
               )

        dp_i = None
        # Regularize S_ij with as small and eps as possible:
        for n in range(5):
            try:
                dp_i = self.gradient_calc.pd_solve(S_ij, eps, f_i)
            except tf.errors.InvalidArgumentError:
                print("Cholesky solve failed, continuing with higher regularization")
                eps *= 2.

            # No exception?  Succeeded, break the loop.
            break

        if dp_i is None:
            raise Exception("Could not invert S_ij for any epsilon tried.")

        # Get the unscaled gradients:
        delta_p = self.unflatten_weights_or_gradients(self.flat_shape, self.correct_shape, dp_i)
        # Now iterate over delta values to optimize the step size:
        delta_max = tf.constant(self.optimizer_config.delta_max, dtype=S_ij.dtype)
        delta_min = tf.constant(self.optimizer_config.delta_min, dtype=S_ij.dtype)

        # take the current energy as the starting miniumum:
        energy_min = self.estimator['energy'] + 1

        # Snapshot the current weights:
        original_weights = self.wavefunction.trainable_variables


        # Select the delta options:

        n_delta_iterations = 10

        delta_options = tf.linspace(tf.math.log(delta_max), tf.math.log(delta_min), n_delta_iterations)
        delta_options = tf.math.exp(delta_options)

        energies = []
        overlaps = []
        acoses   = []

        for i,this_delta in enumerate(delta_options):


            # We have the original energy, and gradient updates.
            # Apply the updates:
            loop_items = zip(self.adaptive_wavefunction.trainable_variables, original_weights, delta_p)
            for weight, original_weight, gradient in loop_items:
                weight.assign(original_weight + this_delta * gradient)

            # Compute the new energy:
            energy, overlap, acos = self.recompute_energy(self.adaptive_wavefunction, current_psi)
            energies.append(energy)
            overlaps.append(overlap)
            acoses.append(acos)

        delta_options = [ d.numpy() for d in delta_options ]

        # print("Energies: ",  energies)
        # print("Deltas: ", delta_options)
        # print("acoses: ", acoses)
        # print("overlaps: ", overlaps)

        # We find the best delta, with the constraint that overlap > 0.8 and par_dis < 0.4
        found = False

        energy_rms = tf.math.reduce_std(energies)


        while len(energies) > 0:
            # What's the smallest energy?
            i_e_min = numpy.argmin(energies)

            par_dist = self.gradient_calc.par_dist(delta_options[i_e_min]*dp_i, S_ij)

            ratio = tf.abs(par_dist - acoses[i_e_min]) / tf.abs(par_dist + acoses[i_e_min])

            #
            # print("i_e_min: ", i_e_min, ", Delta: ", delta_options[i_e_min], ", ratio: ", ratio, ", overlap: ", overlap[i_e_min], ", par_dist: ", par_dist, ", acos: ", acos)
            # print(hvd.rank(), " Delta: ", delta_options[i_e_min], ", par_dist: ", par_dist)
            # print(hvd.rank(), " Delta: ", delta_options[i_e_min], ", acos: ", acos)

            if par_dist < 0.1 and acoses[i_e_min] < 0.1  and overlaps[i_e_min] > 0.9 and ratio < 0.4:
                found = True
                final_overlap = overlaps[i_e_min]
                next_energy = energies[i_e_min]
                break
            else:
                logger.debug(f"Skipping this energy (acos: {acoses[i_e_min]}, overlap: {overlaps[i_e_min]}, par_dist: {par_dist}, ratio: {ratio})")

                # Remove these options
                energies.pop(i_e_min)
                overlaps.pop(i_e_min)
                acoses.pop(i_e_min)
                delta_options.pop(i_e_min)

                final_overlap = 2.0
                next_energy = None


        # print("i_e_min: ", i_e_min)
        if found:
            best_delta = delta_options[i_e_min]
        else:
            # Apply no update.  Rewalk and recompute.
            best_delta = 0.0
            ratio = 10.0
            acos = 10.
            overlap = 2.0


        gradients = [ best_delta * g for g in delta_p ]
        delta_metrics = {
            "optimizer/delta"   : best_delta,
            "optimizer/eps"     : eps,
            "optimizer/overlap" : final_overlap,
            "optimizer/par_dist": par_dist,
            "optimizer/acos"    : acos,
            "optimizer/energy_rms": energy_rms,
            "optimizer/ratio"   : ratio
        }
        return gradients, delta_metrics, next_energy
