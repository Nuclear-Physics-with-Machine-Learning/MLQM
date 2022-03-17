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

class BaseAlgorithm(object):

    def __init__(self,
            sampler                   : MetropolisSampler,
            wavefunction              : callable,
            adaptive_wavefunction     : callable,
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
        self.use_spin                  = sampler_config.use_spin
        self.use_isospin               = sampler_config.use_isospin

        # MPI Enabled?
        if MPI_AVAILABLE:
            self.size = hvd.size()
            self.rank = hvd.rank()
        else:
            self.size = 1
            self.rank = 1

        self.estimator = Estimator()

        # Use a "disposable" wavefunction too:
        self.adaptive_wavefunction = adaptive_wavefunction

        self.correct_shape = [ p.shape for p in self.wavefunction.trainable_variables ]

        self.predicted_energy = None

    @tf.function
    def batched_jacobian(self, nobs, x_current_arr, spin_arr, isospin_arr, wavefunction, jac_fnc):
        ret_jac = []
        if not self.use_spin:
            spin_arr = [None] * nobs
        if not self.use_isospin:
            isospin_arr = [None] * nobs

        # # Stack up the objects to allow vectorized map_
        # spin_arr = tf.stack(spin_arr)
        # isospin_arr = tf.stack(isospin_arr)
        # x_current_arr = tf.stack(x_current_arr)
        #
        # jac = lambda x : jac_fnc(x[0],x[1],x[2],wavefunction)
        #
        #
        # flattened_jacobian = tf.vectorized_map(
        #     jac, (x_current_arr, spin_arr, isospin_arr)
        # )
        #
        # ret_jac = tf.split(flattened_jacobian, nobs)
        # flat_shape = flat_shape[0]
        #
        #
        for i in range(nobs):

            flattened_jacobian, flat_shape = jac_fnc(
                x_current_arr[i], spin_arr[i], isospin_arr[i], wavefunction)
            ret_jac.append(flattened_jacobian)


        return ret_jac, flat_shape


    @tf.function
    def jacobian(self, x_current, spin, isospin, wavefunction):
        tape = tf.GradientTape()
        # n_walkers = x_current.shape[0]
        with tape:
            wpsi = wavefunction(x_current, spin, isospin)


        jac = tape.jacobian(wpsi, wavefunction.trainable_variables)
        jac_shape = [j.shape[1:] for j in jac]


        # Normalize the jacobian estimation by the wavefunction:

        for i in range(len(jac)):
            new_shape =[1 for j in jac_shape[i]]
            jac[i] = jac[i] / tf.reshape(wpsi, [-1, *new_shape])
        # jac = tape.jacobian(wpsi, wavefunction.trainable_variables, parallel_iterations = MAX_PARALLEL_ITERATIONS)
        #

        # Grab the original shapes ([1:] means everything except first dim):
        # get the flattened shapes:
        flat_shape = [[-1, tf.reduce_prod(js)] for js in jac_shape]
        # Reshape the

        # We have the flat shapes and now we need to make the jacobian into a single matrix

        flattened_jacobian = [tf.reshape(j, f) for j, f in zip(jac, flat_shape)]

        flattened_jacobian = tf.concat(flattened_jacobian, axis=-1)

        return flattened_jacobian, flat_shape



    @tf.function
    def compute_O_observables(self, flattened_jacobian, energy, w_of_x):

        # dspi_i is the reduction of the jacobian over all walkers.
        # In other words, it's the mean gradient of the parameters with respect to inputs.
        # This is effectively the measurement of O^i in the paper.
        normed_fj = flattened_jacobian
        # normed_fj = flattened_jacobian / (w_of_x )


        dpsi_i = tf.reduce_mean(normed_fj, axis=0)
        dpsi_i = tf.reshape(dpsi_i, [-1,1])
        # To compute <O^m O^n>
        dpsi_ij = tf.linalg.matmul(normed_fj, normed_fj, transpose_a = True) / self.n_walkers_per_observation

        # Computing <O^m H>:
        e_reshaped = tf.reshape(energy, [1,self.n_walkers_per_observation])

        dpsi_i_EL = tf.linalg.matmul(e_reshaped, normed_fj)
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
        kicker_params = {"mean": 0.0, "stddev" : 0.6}

        acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=n_equilibrations)

        return acceptance

    # @tf.function
    def compile(self):
        # This step gets fresh walkers and computes their energy, which calls compile steps

        kicker = tf.random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.2}
        acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=1)
        x_current, spin, isospin  = self.sampler.sample()
        energy, energy_jf, ke_jf, ke_direct, pe, w_of_x = \
            self.hamiltonian.energy(self.wavefunction, x_current,
                spin, isospin)


    #
    # @tf.function
    @profile
    def recompute_energy(self, test_wavefunction, current_psi):

        estimator = Estimator()
        estimator.clear()

        for next_x, spins, isospins, this_current_psi in zip(*self.sampler.get_all_walkers(), current_psi):

            # Compute the observables:
            energy, energy_jf, ke_jf, ke_direct, pe, w_of_x = \
                self.hamiltonian.energy(test_wavefunction, next_x, spins, isospins)

            # Here, we split the energy and other objects into sizes of nwalkers_per_observation
            # if self.n_concurrent_obs_per_rank != 1:
            next_x     = tf.split(next_x, self.n_concurrent_obs_per_rank, axis=0)
            energy     = tf.split(energy, self.n_concurrent_obs_per_rank, axis=0)
            w_of_x     = tf.split(w_of_x, self.n_concurrent_obs_per_rank, axis=0)

            # overlap of wavefunctions:
            wavefunction_ratio = [ next_psi / (curr_psi + 1e-16) for next_psi, curr_psi in zip(w_of_x, this_current_psi) ]
            probability_ratio  = [ tf.reshape(wf_ratio**2, energy[i].shape) for i, wf_ratio in enumerate(wavefunction_ratio) ]


            for i_obs in range(self.n_concurrent_obs_per_rank):
                obs_energy = probability_ratio[i_obs] * energy[i_obs]


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

        energy = estimator['energy']

        return energy, overlap, acos


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
                logger.debug("Cholesky solve failed, continuing with higher regularization")
                eps *= 2.
            continue

        return dp_i, S_ij

    @profile
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
            x_current, spin, isospin  = _sampler.sample()
            # Compute the observables:
            # Here, perhaps we can compute the d_i of obs_energy:

            energy, energy_jf, ke_jf, ke_direct, pe, w_of_x = \
                self.hamiltonian.energy(_wavefunction, x_current, spin, isospin)

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
            # Split walkers:
            x_current  = tf.split(x_current, self.n_concurrent_obs_per_rank, axis=0)
            if self.use_spin:
                spin       = tf.split(spin,      self.n_concurrent_obs_per_rank, axis=0)
            else:
                spin = None
            if self.use_isospin:
                isospin    = tf.split(isospin,    self.n_concurrent_obs_per_rank, axis=0)
            else:
                isospin = None

            # Split observables:
            energy     = tf.split(energy,    self.n_concurrent_obs_per_rank, axis=0)
            energy_jf  = tf.split(energy_jf, self.n_concurrent_obs_per_rank, axis=0)
            ke_jf      = tf.split(ke_jf,     self.n_concurrent_obs_per_rank, axis=0)
            ke_direct  = tf.split(ke_direct, self.n_concurrent_obs_per_rank, axis=0)
            pe         = tf.split(pe,        self.n_concurrent_obs_per_rank, axis=0)
            w_of_x     = tf.split(w_of_x,    self.n_concurrent_obs_per_rank, axis=0)


            current_psi.append(w_of_x)


            # For each observation, we compute the jacobian.
            # flattened_jacobian is a list, flat_shape is just one instance
            flattened_jacobian, flat_shape = self.batched_jacobian(
                self.n_concurrent_obs_per_rank, x_current,
                spin, isospin, _wavefunction, self.jacobian)

            # Here, if MPI is available, we can do a reduction (sum) over walker variables

            # Now, compute observables, store them in an estimator:

            for i_obs in range(self.n_concurrent_obs_per_rank):
                obs_energy      = energy[i_obs]     / self.n_walkers_per_observation
                obs_energy_jf   = energy_jf[i_obs]  / self.n_walkers_per_observation
                obs_ke_jf       = ke_jf[i_obs]      / self.n_walkers_per_observation
                obs_ke_direct   = ke_direct[i_obs]  / self.n_walkers_per_observation
                obs_pe          = pe[i_obs]         / self.n_walkers_per_observation



                dpsi_i, dpsi_ij, dpsi_i_EL = self.compute_O_observables(
                    flattened_jacobian[i_obs], obs_energy, w_of_x[i_obs])



                #
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


    def compute_updates_and_metrics(self, current_psi):

        raise Exception("Must be implemented in child class")

    # @tf.function
    @profile
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

        delta_p, opt_metrics, next_energy = self.compute_updates_and_metrics(current_psi)

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
