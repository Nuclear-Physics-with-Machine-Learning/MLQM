import tensorflow as tf
import numpy

import logging
# Set up logging:
logger = logging.getLogger()

from mlqm.samplers     import Estimator

class StochasticReconfiguration(object):

    def __init__(self, sampler, wavefunction, hamiltonian, optimizer):

        self.sampler      = sampler
        self.hamiltonian  = hamiltonian
        self.wavefunction = wavefunction
        self.optimizer    = optimizer

        self.nwalkers_local     = self.sampler.nwalkers
        self.nwalkers_global    = self.nwalkers_local


        pass




    @tf.function
    def batched_jacobian(self, nobs, x_current_arr, wavefunction, jac_fnc):
        ret_jac = []
        ret_shape = []
        for i in range(nobs):
            flattened_jacobian, flat_shape = jac_fnc(x_current_arr[i], wavefunction)
            ret_jac.append(flattened_jacobian)
            ret_shape.append(flat_shape)

        return ret_jac, ret_shape





    @tf.function
    def jacobian(self, x_current, wavefunction):
        tape = tf.GradientTape()

        with tape:
            log_wpsi = wavefunction(x_current)

        jac = tape.jacobian(log_wpsi, wavefunction.trainable_variables)

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
        dpsi_ij = tf.linalg.matmul(flattened_jacobian, flattened_jacobian, transpose_a = True) / self.nwalkers_local

        # Computing <O^m H>:
        dpsi_i_EL = tf.linalg.matmul(tf.reshape(energy, [1,self.nwalkers_local]), flattened_jacobian)
        # This makes this the same shape as the other tensors
        dpsi_i_EL = tf.reshape(dpsi_i_EL, [-1, 1])

        return dpsi_i, dpsi_ij, dpsi_i_EL

    def update_model(self):

        if self.latest_gradients is not None:

            # Update the parameters:
            for i_param in range(len(self.wavefunction.trainable_variables)):
                self.wavefunction.trainable_variables[i_param].assign_add(self.latest_gradients[i_param])


    def equilibrate(self, n_equilibrations):

        kicker = tf.random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.4}

        acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=n_equilibrations)

        return acceptance

    def sr_step(self, nvoid):

        metrics = {}
        self.latest_gradients = None


        kicker = tf.random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.4}

        # First do a void walk to thermalize after a new configuration.
        # By default, this will use the previous walkers as a starting configurations.
        #   This one does all the kicks.
        acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=nvoid*10)



        # Get the current walker locations:
        x_current  = self.sampler.sample()

        # Compute the observables:
        energy, energy_jf, ke_jf, ke_direct, pe = self.hamiltonian.energy(self.wavefunction, x_current)

        # Here, if MPI is available, we can do a reduction (sum) over walker variables

        print(energy)


        flattened_jacobian, flat_shape = self.jacobian(x_current, self.wavefunction)

        dpsi_i, dpsi_ij, dpsi_i_EL = self.compute_O_observables(flattened_jacobian, energy)

        # Here, if MPI is available, AVERAGE the oberservables over all ranks
        # MPI_average(dspi_i)
        # MPI_average(dpsi_ij)
        # MPI_average(dpsi_i_EL)



        energy_summed     = tf.reduce_sum(energy) / self.nwalkers_local
        energy2_summed    = tf.reduce_sum(energy**2) / self.nwalkers_local
        energy_jf_summed  = tf.reduce_sum(energy_jf) / self.nwalkers_local
        energy2_jf_summed = tf.reduce_sum(energy_jf**2) / self.nwalkers_local




        print(energy_summed.shape)

        # We do here an MPI sum over observed variables, particularly energy

        # energy = 



        metrics = {}
        metrics["energy/ke_jf"] = tf.reduce_sum(energy_jf)
        metrics["energy/ke"] = tf.reduce_sum(ke_direct)
        metrics["energy/pe"] = tf.reduce_sum(pe)
        metrics["energy/ke_jf_std"] = tf.math.reduce_std(ke_jf)
        metrics["energy/ke_std"] = tf.math.reduce_std(ke_direct)
        metrics["energy/pe_std"] = tf.math.reduce_std(pe)


        # if MPI_AVAILABLE:
        #     # Here, we have to do a reduction over all params used to calculate gradients
        #     energy    = hvd.allreduce(energy)
        #     dpsi_i    = hvd.allreduce(dpsi_i)
        #     dpsi_i_EL = hvd.allreduce(dpsi_i_EL)
        #     dpsi_ij   = hvd.allreduce(dpsi_ij)
###########################################################################################
# NOTE: THE ABOVE REDUCTION WILL MESS UP THE ERROR CALCULATIONS
###########################################################################################
        # logger.info(f"psi norm{tf.reduce_mean(log_wpsi)}")
        dp_i = self.optimizer.sr(energy_summed,dpsi_i,dpsi_i_EL,dpsi_ij)

        # Here, we recover the shape of the parameters of the network:
        running_index = 0
        gradient = []
        for length in flat_shape:
            l = length[-1].numpy()
            end_index = running_index + l
            gradient.append(dp_i[running_index:end_index])
            running_index += l
        shapes = [ p.shape for p in self.wavefunction.trainable_variables ]
        delta_p = [ tf.reshape(g, s) for g, s in zip(gradient, shapes)]

        metrics['energy/energy']     = energy_summed
        metrics['energy/error']      = tf.sqrt(energy2_summed - energy_summed**2)
        metrics['energy/energy_jf']  = energy_jf_summed
        metrics['energy/error_jf']   = tf.sqrt(energy2_jf_summed - energy_jf_summed**2) 
        metrics['metropolis/acceptance'] = acceptance

        self.latest_gradients = delta_p

        return  metrics