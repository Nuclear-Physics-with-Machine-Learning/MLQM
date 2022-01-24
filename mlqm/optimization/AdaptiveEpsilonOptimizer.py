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

class AdaptiveEpsilonOptimizer(BaseAlgorithm):

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
    
        delta = self.optimizer_config.delta
        delta_p, opt_metrics, next_energy = self.optimize_eps(current_psi, delta)

        return delta_p, opt_metrics, next_energy

    # @tf.function
    def optimize_eps(self, current_psi, delta):

        f_i = self.gradient_calc.f_i(
                self.estimator['dpsi_i'],
                self.estimator["energy"],
                self.estimator["dpsi_i_EL"]
                )

        S_ij = self.gradient_calc.S_ij(
                self.estimator['dpsi_ij'],
                self.estimator['dpsi_i']
               )



        # We do a search over eps ranges to compute the optimal value:

        eps_max = tf.constant(self.optimizer_config.epsilon_max, dtype=S_ij.dtype)
        eps_min = tf.constant(self.optimizer_config.epsilon_min, dtype=S_ij.dtype)


        def evaluate(_s_ij, _f_i, current_psi, delta, _eps):

            # First, set metrics to null values:
            _metrics = {
                "optimizer/delta"   : delta,
                "optimizer/eps"     : _eps,
                "optimizer/overlap" : 2.0,
                "optimizer/par_dist": 2.0,
                "optimizer/acos"    : 10,
                "optimizer/ratio"   : 10,
            }

            try:
                dp_i = self.gradient_calc.pd_solve(_s_ij, _eps, _f_i)
            except tf.errors.InvalidArgumentError:
                print("Cholesky solve failed, continuing with higher regularization")
                return None, _metrics, 99999

            # print(dp_i)
            # Scale by the learning rate:
            dp_i = delta * dp_i

            # Unpack the gradients
            gradients = self.unflatten_weights_or_gradients(self.flat_shape, self.correct_shape, dp_i)


            original_weights = self.wavefunction.trainable_variables

            loop_items = zip(self.adaptive_wavefunction.trainable_variables, original_weights, gradients)
            for weight, original_weight, gradient in loop_items:
                weight.assign(original_weight + gradient)


            # Compute the new energy:
            next_energy, overlap, acos = self.recompute_energy(self.adaptive_wavefunction, current_psi)


            # Compute the parameter distance:
            par_dist = self.gradient_calc.par_dist(dp_i, _s_ij)
            ratio    = tf.abs(par_dist - acos) / tf.abs(par_dist + acos+ 1e-8)
            _metrics = {
                "optimizer/delta"   : delta,
                "optimizer/eps"     : _eps,
                "optimizer/overlap" : overlap,
                "optimizer/par_dist": par_dist,
                "optimizer/acos"    : acos,
                "optimizer/ratio"   : ratio
            }
            return gradients, _metrics, next_energy


        def metric_check(metrics):
            if metrics['optimizer/ratio'] > 0.4: return False
            if metrics['optimizer/overlap'] < 0.9: return False
            if metrics['optimizer/par_dist'] > 0.1: return False
            if metrics['optimizer/acos'] > 0.1: return False
            return True


        # First, evaluate at eps min and eps max:

        # print(eps_min)
        # print(eps_max)
        grad_low,  metrics_low,  energy_low  = evaluate(S_ij, f_i, current_psi, delta, eps_min)
        grad_high, metrics_high, energy_high = evaluate(S_ij, f_i, current_psi, delta, eps_max)

        # print(grad_high)
        # print(grad_low)

        # Take the current minimum as the high energy:
        if metric_check(metrics_high):
            current_minimum_energy = energy_high
            current_best_grad = grad_high
            current_best_metrics = metrics_high
        else:
            # If the highest eps metrics failed, we're not gonna succeed here.
            grad = [0*w for w in grad_high]
            return grad, metrics_high, None

        # We use a bisection section technique, in log space, to narrow down the right epsilon.
        converged = False
        for i in range(5):

            # And, compute the mid point:
            eps_mid = tf.sqrt(eps_max * eps_min)
            grad_mid,  metrics_mid,  energy_mid  = evaluate(S_ij, f_i, current_psi, delta, eps_mid)


            # We have 3 values, high, mid, and low eps.  With the same delta, the smallest eps
            # is the most aggressive update.  The biggest eps is the least aggressive.
            # (eps is applied before matrix inversion)
            # If all 3 points pass, we narrow in.


            # If we're here, the most aggressive update passed linear expansion checks.
            # Check the mid point anywyas:
            if not metric_check(metrics_mid):
                logger.debug("Skipping this energy.", metrics_mid)
                eps_min = eps_mid
                metrics_min = metrics_mid
                grad_min = grad_mid
                continue

            if energy_mid < current_minimum_energy:
                eps_max = eps_mid
                grad_max = grad_mid
                metrics_max = metrics_mid
                energy_max = energy_mid
                current_minimum_energy  = energy_mid
                current_minimum_grad    = grad_mid
                current_minimum_metrics = metrics_mid
                converged = True
            else:
                eps_min = eps_mid
                grad_min = grad_mid
                metrics_min = metrics_mid
                energy_min = energy_mid

        if not converged:
            grad = [0*w for w in grad_high]
            logger.debug("No update selected for this step.")
            return grad, metrics_high, None


        return current_minimum_grad, current_minimum_metrics, current_minimum_energy
