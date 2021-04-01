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

class StochasticReconfiguration(object):

    def __init__(self,
            sampler                   : MetropolisSampler,
            wavefunction              : callable,
            adaptive_wfn              : callable,
            hamiltonian               : Hamiltonian,
            optimizer_config          : DictConfig,
            sampler_config            : DictConfig,
        ):

        # Store the objects:
        self.sampler       = sampler
        self.wavefunction  = wavefunction
        self.hamiltonian   = hamiltonian
        # Initialize a Gradiant Calculator:
        self.gradient_calc = GradientCalculator()

        self.current_delta = None
        self.current_eps   = None

        self.optimizer_config = optimizer_config

        # Store the measurement configurations:
        self.n_observable_measurements = sampler_config.n_observable_measurements
        self.n_void_steps              = sampler_config.n_void_steps
        self.n_walkers_per_observation = sampler_config.n_walkers_per_observation
        self.n_concurrent_obs_per_rank = sampler_config.n_concurrent_obs_per_rank

        # MPI Enabled?
        if MPI_AVAILABLE:
            self.size = hvd.size()
            self.rank = hvd.rank()
        else:
            self.size = 1
            self.rank = 1

        self.estimator = Estimator()

        # Use a "disposable" wavefunction too:
        self.adaptive_wavefunction = adaptive_wfn

        self.correct_shape = [ p.shape for p in self.wavefunction.trainable_variables ]

        self.predicted_energy = None

    @tf.function
    def batched_jacobian(self, nobs, x_current_arr, wavefunction, jac_fnc):
        ret_jac = []
        for i in range(nobs):
            flattened_jacobian, flat_shape = jac_fnc(x_current_arr[i], wavefunction)
            ret_jac.append(flattened_jacobian)

        return ret_jac, flat_shape


    @tf.function
    def jacobian(self, x_current, wavefunction):
        tape = tf.GradientTape()
        # n_walkers = x_current.shape[0]

        # print("Doing forward pass")
        with tape:
            log_wpsi = wavefunction(x_current)


        jac = tape.jacobian(log_wpsi, wavefunction.trainable_variables)
        # jac = tape.jacobian(log_wpsi, wavefunction.trainable_variables, parallel_iterations = MAX_PARALLEL_ITERATIONS)


        # Grab the original shapes ([1:] means everything except first dim):
        jac_shape = [j.shape[1:] for j in jac]
        # get the flattened shapes:
        flat_shape = [[-1, tf.reduce_prod(js)] for js in jac_shape]
        # Reshape the

        # We have the flat shapes and now we need to make the jacobian into a single matrix

        flattened_jacobian = [tf.reshape(j, f) for j, f in zip(jac, flat_shape)]

        flattened_jacobian = tf.concat(flattened_jacobian, axis=-1)

        return flattened_jacobian, flat_shape



    @tf.function
    def compute_O_observables(self, flattened_jacobian, energy):

        # dspi_i is the reduction of the jacobian over all walkers.
        # In other words, it's the mean gradient of the parameters with respect to inputs.
        # This is effectively the measurement of O^i in the paper.
        dpsi_i = tf.reduce_mean(flattened_jacobian, axis=0)
        dpsi_i = tf.reshape(dpsi_i, [-1,1])

        # To compute <O^m O^n>
        dpsi_ij = tf.linalg.matmul(flattened_jacobian, flattened_jacobian, transpose_a = True) / self.n_walkers_per_observation

        # Computing <O^m H>:
        dpsi_i_EL = tf.linalg.matmul(tf.reshape(energy, [1,self.n_walkers_per_observation]), flattened_jacobian)
        # This makes this the same shape as the other tensors
        dpsi_i_EL = tf.reshape(dpsi_i_EL, [-1, 1])

        return dpsi_i, dpsi_ij, dpsi_i_EL




    @tf.function
    def apply_gradients(self, gradients, variables):

        # Update the parameters:
        for grad, var in zip(gradients, variables):
           var.assign_add(grad)

        return


    def equilibrate(self, n_equilibrations):

        kicker = tf.random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.4}

        acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=n_equilibrations)

        return acceptance

    # @tf.function
    def compile(self):
        # This step gets fresh walkers and computes their energy, which calls compile steps

        kicker = tf.random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.2}
        acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=1)
        x_current  = self.sampler.sample()
        energy, energy_jf, ke_jf, ke_direct, pe, logw_of_x = self.hamiltonian.energy(self.wavefunction, x_current)


    #
    # @tf.function
    def recompute_energy(self, test_wavefunction, current_psi, ):

        estimator = Estimator()
        estimator.clear()

        for next_x, this_current_psi in zip(self.sampler.get_all_walkers(), current_psi):

            # Compute the observables:
            energy, energy_jf, ke_jf, ke_direct, pe, logw_of_x = \
                self.hamiltonian.energy(test_wavefunction, next_x)

            # Here, we split the energy and other objects into sizes of nwalkers_per_observation
            # if self.n_concurrent_obs_per_rank != 1:
            next_x     = tf.split(next_x,    self.n_concurrent_obs_per_rank, axis=0)
            energy     = tf.split(energy,    self.n_concurrent_obs_per_rank, axis=0)
            logw_of_x  = tf.split(logw_of_x, self.n_concurrent_obs_per_rank, axis=0)

            # print("New energy: ", energy)
            # print("New psi: ", logw_of_x)

            # overlap of wavefunctions:
            wavefunction_ratio = [ tf.math.exp((next_psi - curr_psi)) for next_psi, curr_psi in zip(logw_of_x, this_current_psi) ]
            probability_ratio  = [ tf.reshape(wf_ratio**2, energy[i].shape) for i, wf_ratio in enumerate(wavefunction_ratio) ]

            # print("wavefunction_ratio: ", wavefunction_ratio)
            # print("probability_ratio: ", probability_ratio)

            for i_obs in range(self.n_concurrent_obs_per_rank):
                obs_energy = probability_ratio[i_obs] * energy[i_obs]

                # print(tf.reduce_sum(obs_energy))
                # print(tf.reduce_sum(obs_energy) / tf.reduce_sum(probability_ratio[i_obs]))

                estimator.accumulate("energy", tf.reduce_sum(obs_energy))
                estimator.accumulate("weight", tf.reduce_sum(probability_ratio[i_obs]))
                estimator.accumulate("wavefunction_ratio", tf.reduce_sum(wavefunction_ratio[i_obs]))
                estimator.accumulate("N", tf.convert_to_tensor(self.n_walkers_per_observation, dtype=DEFAULT_TENSOR_TYPE))


        if MPI_AVAILABLE:
            estimator.allreduce()


        # What's the total weight?  Use that for the finalization:
        total_weight = estimator['weight']

        # Get the overlap
        wavefunction_ratio = estimator['wavefunction_ratio']
        probability_ratio = estimator['weight']

        N = estimator['N']

        overlap2 = (wavefunction_ratio / N)**2 / (probability_ratio / N)


        estimator.finalize(total_weight)
        overlap  = tf.sqrt(overlap2)
        acos     = tf.math.acos(overlap)**2

        energy = estimator['energy'].numpy()

        return energy, overlap, acos

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

    # @tf.function
    def compute_gradients(self, dpsi_i, energy, dpsi_i_EL, dpsi_ij, eps):

        # Get the natural gradients and S_ij
        f_i = self.gradient_calc.f_i(dpsi_i, energy, dpsi_i_EL)

        S_ij = self.gradient_calc.S_ij(dpsi_ij, dpsi_i)

        # Regularize S_ij with as small and eps as possible:

        for n in range(5):
            try:
                dp_i = self.gradient_calc.pd_solve(S_ij, eps, f_i)
                break
            except tf.errors.InvalidArgumentError:
                print("Cholesky solve failed, continuing with higher regularization")
                eps *= 2.
            continue

        return dp_i, S_ij


    def walk_and_accumulate_observables(self,
            estimator,
            _wavefunction,
            _sampler,
            _n_loops_total,
            _kicker,
            _kicker_params,
        ):
        '''Internal function to take a wavefunction and set of walkers and compute observables

        [description]

        Arguments:
            _n_loops_total {[type]} -- [description]
            _n_concurrent_obs_per_rank {[type]} -- [description]
            _wavefunction {[type]} -- [description]
            _sampler {Sampler} -- [description]
            # Sampler object _kicker {[type]} -- [description]
            _kicker_params {[type]} -- [description]
            _n_void_steps {[type]} -- [description]
            _hamiltonian {[type]} -- [description]
             {[type]} -- [description]
        '''

        estimator.clear()

        current_psi = []

        for i_loop in range(_n_loops_total):
            # logger.debug(f" -- evaluating loop {i_loop} of {n_loops_total}")

            # First do a void walk to thermalize after a new configuration.
            # By default, this will use the previous walkers as a starting configurations.
            # #   This one does all the kicks in a compiled function.

            # UNCOMMENT STARTING HERE
            acceptance = _sampler.kick(_wavefunction, _kicker, _kicker_params, nkicks=self.n_void_steps)


            # Get the current walker locations:
            x_current  = _sampler.sample()

            # Compute the observables:
            energy, energy_jf, ke_jf, ke_direct, pe, logw_of_x = self.hamiltonian.energy(_wavefunction, x_current)


            # R is computed but it needs to be WRT the center of mass of all particles
            # So, mean subtract if needed:
            if _wavefunction.mean_subtract:
                mean = tf.reduce_mean(x_current, axis=1)
                r = x_current - mean[:,None,:]
            else:
                r = x_current

            r = tf.reduce_sum(r**2, axis=(1,2))
            r = tf.reduce_mean(tf.math.sqrt(r))

            # Here, we split the energy and other objects into sizes of nwalkers_per_observation
            # if self.n_concurrent_obs_per_rank != 1:
            x_current  = tf.split(x_current, self.n_concurrent_obs_per_rank, axis=0)
            energy     = tf.split(energy,    self.n_concurrent_obs_per_rank, axis=0)
            energy_jf  = tf.split(energy_jf, self.n_concurrent_obs_per_rank, axis=0)
            ke_jf      = tf.split(ke_jf,     self.n_concurrent_obs_per_rank, axis=0)
            ke_direct  = tf.split(ke_direct, self.n_concurrent_obs_per_rank, axis=0)
            pe         = tf.split(pe,        self.n_concurrent_obs_per_rank, axis=0)
            logw_of_x  = tf.split(logw_of_x, self.n_concurrent_obs_per_rank, axis=0)

            # print("Original logw_of_x: ", logw_of_x)
            # print("Original energy: ", energy)

            current_psi.append(logw_of_x)


            # For each observation, we compute the jacobian.
            # flattened_jacobian is a list, flat_shape is just one instance
            flattened_jacobian, flat_shape = self.batched_jacobian(
                self.n_concurrent_obs_per_rank, x_current, _wavefunction, self.jacobian)

            # Here, if MPI is available, we can do a reduction (sum) over walker variables

            # Now, compute observables, store them in an estimator:

            for i_obs in range(self.n_concurrent_obs_per_rank):
                obs_energy      = energy[i_obs]     / self.n_walkers_per_observation
                obs_energy_jf   = energy_jf[i_obs]  / self.n_walkers_per_observation
                obs_ke_jf       = ke_jf[i_obs]      / self.n_walkers_per_observation
                obs_ke_direct   = ke_direct[i_obs]  / self.n_walkers_per_observation
                obs_pe          = pe[i_obs]         / self.n_walkers_per_observation

                # print("obs_energy: ",    obs_energy)
                # print("obs_energy_jf: ", obs_energy_jf)
                # print("obs_ke_jf: ",     obs_ke_jf)
                # print("obs_ke_direct: ", obs_ke_direct)
                # print("obs_pe: ",        obs_pe)


                dpsi_i, dpsi_ij, dpsi_i_EL = self.compute_O_observables(flattened_jacobian[i_obs], obs_energy)

                # print("dpsi_i: ", dpsi_i)
                # print("dpsi_i_EL: ", dpsi_i_EL)
                # print("dpsi_ij: ", dpsi_ij)

                # print("flattened_jacobian: ", flattened_jacobian)

                # Accumulate variables:

                self.estimator.accumulate('energy',  tf.reduce_sum(obs_energy))
                self.estimator.accumulate('energy2',  tf.reduce_sum(obs_energy)**2)
                self.estimator.accumulate('energy_jf',  tf.reduce_sum(obs_energy_jf))
                self.estimator.accumulate('energy2_jf',  tf.reduce_sum(obs_energy_jf)**2)
                self.estimator.accumulate('ke_jf',  tf.reduce_sum(obs_ke_jf))
                self.estimator.accumulate('ke_direct',  tf.reduce_sum(obs_ke_direct))
                self.estimator.accumulate('pe',  tf.reduce_sum(obs_pe))
                self.estimator.accumulate('acceptance',  acceptance)
                self.estimator.accumulate('r',  r)
                self.estimator.accumulate('dpsi_i',  dpsi_i)
                self.estimator.accumulate('dpsi_i_EL',  dpsi_i_EL)
                self.estimator.accumulate('dpsi_ij',  dpsi_ij)
                self.estimator.accumulate('weight',  tf.convert_to_tensor(1., dtype=DEFAULT_TENSOR_TYPE ))


                # self.estimator.accumulate(
                #     energy     = tf.reduce_sum(obs_energy),
                #     energy_jf  = tf.reduce_sum(obs_energy_jf),
                #     ke_jf      = tf.reduce_sum(obs_ke_jf),
                #     ke_direct  = tf.reduce_sum(obs_ke_direct),
                #     pe         = tf.reduce_sum(obs_pe),
                #     acceptance = acceptance,
                #     r          = r,
                #     dpsi_i     = dpsi_i,
                #     dpsi_i_EL  = dpsi_i_EL,
                #     dpsi_ij    = dpsi_ij,
                # )



        # INTERCEPT HERE with MPI to allreduce the estimator objects.
        if MPI_AVAILABLE:
            self.estimator.allreduce()

        return flat_shape, current_psi


    # @tf.function

    def sr_step(self, n_thermalize):

        metrics = {}
        self.latest_gradients = None
        self.latest_psi       = None


        kicker = tf.random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.2}


        # We need to know how many times to loop over the walkers and metropolis step.
        # The total number of observations is set: self.n_observable_measurements
        # There is an optimization to walk in parallel with n_concurrent_obs_per_rank
        # Without MPI, the number of loops is then n_observable_measurements / n_concurrent_obs_per_rank
        # WITH MPI, we have to reduce the number of loops by the total number of ranks.

        n_loops_total = int(self.n_observable_measurements / self.n_concurrent_obs_per_rank)

        if MPI_AVAILABLE:
            n_loops_total = int(n_loops_total / self.size)
        # logger.debug(" -- Coordinating loop length")

        # We do a check that n_loops_total * n_concurrent_obs_per_rank matches expectations:
        if n_loops_total * self.n_concurrent_obs_per_rank*self.size != self.n_observable_measurements:
            exception_str = "Total number of observations to compute is unexpected!\n"
            exception_str += f"  Expected to have {self.n_observable_measurements}, have:\n"
            exception_str += f"  -- A loop of {self.n_concurrent_obs_per_rank} observations"
            exception_str += f" for {n_loops_total} loops over {self.size} ranks"
            exception_str += f"  -- ({self.n_concurrent_obs_per_rank})*({n_loops_total}"
            exception_str += f")*({self.size}) != {self.n_observable_measurements}\n"
            raise Exception(exception_str)

        # We do a thermalization step again:
        self.equilibrate(n_thermalize)

        # Clear the walker history:
        self.sampler.reset_history()

        # Now, actually apply the loop and compute the observables:
        self.flat_shape, current_psi = self.walk_and_accumulate_observables(
                    self.estimator,
                    self.wavefunction,
                    self.sampler,
                    n_loops_total,
                    _kicker = kicker,
                    _kicker_params = kicker_params,
                )

        # At this point, we need to average the observables that feed into the optimizer:
        self.estimator.finalize(self.n_observable_measurements)


        error = tf.sqrt((self.estimator["energy2"] - self.estimator["energy"]**2) \
            / (self.n_observable_measurements-1))
        error_jf = tf.sqrt((self.estimator["energy2_jf"] - self.estimator["energy_jf"]**2) \
            / (self.n_observable_measurements-1))

        # for key in self.estimator:
        #     print(f"{key}: {self.estimator[key]}")


        metrics['energy/energy']     = self.estimator["energy"]
        metrics['energy/error']      = error
        metrics['energy/energy_jf']  = self.estimator["energy_jf"]
        metrics['energy/error_jf']   = error_jf
        metrics['metropolis/acceptance'] = self.estimator["acceptance"]
        metrics['metropolis/r']      = self.estimator['r']
        metrics['energy/ke_jf']      = self.estimator["ke_jf"]
        metrics['energy/ke_direct']  = self.estimator["ke_direct"]
        metrics['energy/pe']         = self.estimator["pe"]

        # Here, we call the function to optimize eps and compute the gradients:

        if self.optimizer_config.form == "AdaptiveDelta":
            eps = self.optimizer_config.epsilon
            delta_p, opt_metrics, next_energy = self.optimize_delta(current_psi, eps)
        elif self.optimizer_config.form == "AdaptiveEpsilon":
            delta = self.optimizer_config.delta
            delta_p, opt_metrics, next_energy = self.optimize_eps(current_psi, delta)
        else:
            eps = self.optimizer_config.epsilon
            delta = self.optimizer_config.delta
            delta_p, opt_metrics, next_energy = self.flat_optimizer(current_psi, eps, delta)

        metrics.update(opt_metrics)

        # Compute the ratio of the previous energy and the current energy, if possible.
        if self.predicted_energy is not None:
            energy_diff = self.predicted_energy - self.estimator['energy']
        else:
            energy_diff = 0

        metrics['energy/energy_diff'] = energy_diff
        # dp_i, opt_metrics = self.gradient_calc.sr(
        #     self.estimator["energy"],
        #     self.estimator["dpsi_i"],
        #     self.estimator["dpsi_i_EL"],
        #     self.estimator["dpsi_ij"])




        # And apply them to the wave function:
        self.apply_gradients(delta_p, self.wavefunction.trainable_variables)
        self.latest_gradients = delta_p
        self.latest_psi = current_psi

        # Before moving on, set the predicted_energy:
        self.predicted_energy = next_energy


        return  metrics

    # @tf.function
    def unflatten_weights_or_gradients(self, flat_shape, correct_shape, weights_or_gradients):

        running_index = 0
        gradient = []
        for length in self.flat_shape:
            l = length[-1]
            end_index = running_index + l
            gradient.append(weights_or_gradients[running_index:end_index])
            running_index += l
        delta_p = [ tf.reshape(g, s) for g, s in zip(gradient, correct_shape)]

        return delta_p
