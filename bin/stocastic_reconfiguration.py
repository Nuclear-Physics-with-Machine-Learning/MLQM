import sys, os
import pathlib
import configparser
import time

import argparse
import signal
import pickle


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
#
# try:
#     import horovod.tensorflow as hvd
#     hvd.init()
#     from mpi4py import MPI
#
#     # This is to force each rank onto it's own GPU:
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank() + 1)
#     MPI_AVAILABLE=True
# except:
#     MPI_AVAILABLE=False

# Use mixed precision for inference (metropolis walk)
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

import logging
logger = logging.getLogger()

# Create a handler for STDOUT, but only on the root rank:
# if not MPI_AVAILABLE or hvd.rank() == 0:
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


# Add the local folder to the import path:
mlqm_dir = os.path.dirname(os.path.abspath(__file__))
mlqm_dir = os.path.dirname(mlqm_dir)
sys.path.insert(0,mlqm_dir)
from mlqm.samplers     import Estimator
from mlqm.optimization import Optimizer
from mlqm import DEFAULT_TENSOR_TYPE


class exec(object):

    def __init__(self):

        # This technique is taken from: https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html
        parser = argparse.ArgumentParser(
            description='Run ML Based QMC Calculations',
            usage='python exec.py config.ini [<args>]')

        parser.add_argument("-c", "--config-file",
            type        = pathlib.Path,
            required    = True,
            help        = "Python configuration file to describe this run.")

        self.args = parser.parse_args()
        #
        # if MPI_AVAILABLE:
        #     self.rank = hvd.rank()
        #     self.size = hvd.size()
        #     self.local_rank = hvd.local_rank()

        # Open the config file:
        self.config = configparser.ConfigParser()
        self.config.read(self.args.config_file)

        # Use this flag to catch interrupts, stop the next step and write output.
        self.active = True



        optimization = self.config['Optimization']
        sr_parameters = ["iterations","nequilibrations","naverages","nobservations","nvoid", "delta", "eps"]
        for key in sr_parameters:
            if key not in optimization:
                raise Exception(f"Configuration for Optimization missing key {key}")

        self.iterations = int(optimization['iterations'])
        self.nequilibrations = int(optimization['nequilibrations'])
        self.naverages       = int(optimization['naverages'])
        self.nobservations   = int(optimization['nobservations'])
        self.nvoid           = int(optimization['nvoid'])
        self.delta           = float(optimization['delta'])
        self.eps             = float(optimization['eps'])
        self.gamma           = float(optimization['gamma'])
        self.eta             = float(optimization['eta'])

        self.global_step     = 0

        self.dimension  = int(self.config['General']['dimension'])
        self.nparticles = int(self.config['General']['nparticles'])
        self.save_path  = self.config["General"]["save_path"]
        self.model_name = self.config["General"]["model_name"]



        self.build_sampler()
        self.build_hamiltonian()

        # We append some extra information to the save path, if it's not there:
        # Hamiltonian:



        append_token = ""

        if self.hamiltonian_form not in self.save_path:
            append_token += f"/{self.hamiltonian_form}/"

        dimension = f"{self.dimension}D"
        if dimension not in self.save_path:
            append_token += f"/{dimension}/"

        n_part = f"{self.nparticles}particles"
        if n_part not in self.save_path:
            append_token += f"/{n_part}/"

        # If there is an ellipsis in the save path, we replace that.
        # Otherwise it just appends

        if "..." in self.save_path:
            self.save_path = self.save_path.replace("...", append_token)
        else:
            self.save_path += append_token


        self.writer = tf.summary.create_file_writer(self.save_path)

        self.model_path = self.save_path + self.model_name

        # We also snapshot the configuration into the log dir:
        with open(self.save_path + 'config.snapshot.ini', 'w') as cfg:
            self.config.write(cfg)


    def build_sampler(self):

        # First, check the sampler has all required config items:
        sampler = self.config["Sampler"]

        required_keys = ["nwalkers"]

        for key in required_keys:
            if key not in sampler:
                raise Exception(f"Configuration for Sampler missing key {key}")

        self.nwalkers   = int(sampler['nwalkers'])

        # if MPI_AVAILABLE and self.size != 1:
        #     # Scale down the number of walkers:
        #     nwalkers = int(self.nwalkers / self.size)
        #     if self.nwalkers / self.size != nwalkers:
        #         logger.error("ERROR: number of walkers is not evenly divided by MPI COMM size")
        #
        #     self.nwalkers = nwalkers


        from mlqm.samplers import MetropolisSampler

        # As an optimization, we increase the number of walkers by nobservations
        self.sampler = MetropolisSampler(
            n           = self.dimension,
            nparticles  = self.nparticles,
            nwalkers    = self.nwalkers*self.nobservations,
            initializer = tf.random.normal,
            init_params = {"mean": 0.0, "stddev" : 0.2},
            dtype       = DEFAULT_TENSOR_TYPE)

        return

    def check_potential_parameters(self, potential, parameters, config):

        for key in parameters:
            if key not in config:
                raise Exception(f"Configuration for {potential} missing key {key}")

    def build_hamiltonian(self):

        # First, ask for the type of hamiltonian
        self.hamiltonian_form = self.config["Hamiltonian"]["form"]
        if self.hamiltonian_form == "HarmonicOscillator":

            from mlqm.hamiltonians import HarmonicOscillator

            required_keys = ["mass", "omega"]
            self.check_potential_parameters(self.hamiltonian_form, required_keys, self.config["Hamiltonian"])


            self.hamiltonian = HarmonicOscillator(
                M           = float(self.config["Hamiltonian"]["mass"]),
                omega       = float(self.config["Hamiltonian"]["omega"]),
            )
        elif self.hamiltonian_form == "AtomicPotential":
            from mlqm.hamiltonians import AtomicPotential

            required_keys = ["mass", "Z"]
            self.check_potential_parameters(self.hamiltonian_form, required_keys, self.config["Hamiltonian"])

            self.hamiltonian = AtomicPotential(
                mu          = float(self.config["Hamiltonian"]["mass"]),
                Z           = int(self.config["Hamiltonian"]["Z"]),
            )
        elif self.hamiltonian_form == "NuclearPotential":
            from mlqm.hamiltonians import NuclearPotential

            required_keys = ["mass", "Z"]
            self.check_potential_parameters(self.hamiltonian_form, required_keys, self.config["Hamiltonian"])


            self.hamiltonian = NuclearPotential(
                M           = float(self.config["Hamiltonian"]["mass"]),
                omega       = float(self.config["Hamiltonian"]["omega"]),
            )
        else:
            raise Exception(f"Unknown potential requested: {self.hamiltonian_form}")

    def restore(self):
        self.wavefunction.load_weights(self.model_path)

        with open(self.save_path + "/global_step.pkl", 'rb') as _f:
            self.global_step = pickle.load(file=_f)
        with open(self.save_path + "/optimizer.pkl", 'rb') as _f:
            self.optimizer   = pickle.load(file=_f)


    def run(self):
        tf.keras.backend.set_floatx(DEFAULT_TENSOR_TYPE)
        x = self.sampler.sample()

        # For each dimension, randomly pick a degree
        degree = [ 1 for d in range(self.dimension)]

        from mlqm.models import DeepSetsWavefunction
        self.wavefunction = DeepSetsWavefunction(self.dimension, self.nparticles, mean_subtract=False)

        # Run the wave function once to initialize all its weights
        tf.summary.trace_on(graph=True)
        _ = self.wavefunction(x)
        tf.summary.trace_off()
        # We attempt to restore the weights:
        try:
            self.restore()
            print("Loaded weights, optimizer and global step!")
        except Exception as excep:
            print("Failed to load weights!")
            print(excep)
            pass



        # if MPI_AVAILABLE:
        #     # We have to broadcast the wavefunction parameter here:
        #     hvd.broadcast_variables(self.wavefunction.variables, 0)
        #

        from mlqm.optimization import Optimizer
        from mlqm.samplers     import Estimator




        # Create an optimizer
        self.optimizer = Optimizer(
            delta   = self.delta,
            eps     = self.eps,
            npt     = self.wavefunction.n_parameters(),
            gamma   = self.gamma,
            eta     = self.eta,
            dtype   = DEFAULT_TENSOR_TYPE)

        # energy, energy_by_parts = self.hamiltonian.energy(self.wavefunction, x)
        energy, energy_jf, ke_jf, ke_direct, pe = self.hamiltonian.energy(self.wavefunction, x)

        while self.global_step < self.iterations:
            if not self.active: break

            start = time.time()

            delta_p, metrics,  = self.sr_step(
                self.nequilibrations,
                self.naverages,
                self.nobservations,
                self.nvoid)

            self.summary(metrics, self.global_step)

            # Update the parameters:
            for i_param in range(len(self.wavefunction.trainable_variables)):
                self.wavefunction.trainable_variables[i_param].assign_add(delta_p[i_param])

            if self.global_step % 1 == 0:
                logger.info(f"step = {self.global_step}, energy = {metrics['energy/energy'].numpy():.3f}, err = {metrics['energy/error'].numpy():.3f}")
                logger.info(f"step = {self.global_step}, energy_jf = {metrics['energy/energy_jf'].numpy():.3f}, err = {metrics['energy/error_jf'].numpy():.3f}")
                logger.info(f"acc  = {metrics['metropolis/acceptance'].numpy():.3f}")
                logger.info(f"time = {time.time() - start:.3f}")

            # Iterate:
            self.global_step += 1

    # @tf.function
    def summary(self, metrics, step):
        with self.writer.as_default():
            for key in metrics:
                tf.summary.scalar(key, metrics[key], step=step)

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
        dpsi_ij = tf.linalg.matmul(flattened_jacobian, flattened_jacobian, transpose_a = True) / self.nwalkers

        # Computing <O^m H>:
        dpsi_i_EL = tf.linalg.matmul(tf.reshape(energy, [1,self.nwalkers]), flattened_jacobian)
        # This makes this the same shape as the other tensors
        dpsi_i_EL = tf.reshape(dpsi_i_EL, [-1, 1])

        return dpsi_i, dpsi_ij, dpsi_i_EL

    def sr_step(self, nequilibrations, naverages, nobservations, nvoid):

        metrics = {}

        nblock = nequilibrations + naverages
        block_estimator = Estimator(info=None)
        total_estimator = Estimator(info=None)

        # Metropolis sampler will sample configurations uniformly:
        # Sample initial configurations uniformy between -sig and sig
        x_original = self.sampler.sample()



        # This function operates with the following loops:
        #  - There are nblock "blocks" == nequilibrations + naverages
        #    - the first 'nequilibrations' are not used for gradient calculations
        #    - the next 'naverages' are used for accumulating wavefunction measurements
        #  - In each block, there are nsteps = nobservations * nvoid
        #    - The walkers are kicked at every step.
        #    - if the step is a multiple of nvoid, AND the block is bigger than nequilibrations, the calculations are done.
        kicker = tf.random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.2}

        # First do a void walk to equilabrate / thermalize after a new configuration:
        #   This first call is compiling the kicker
        acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=1)
        #   This one does all the kicks.
        acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=nvoid*nequilibrations*naverages)
        total_estimator, flat_shape, sub_metrics = self.sr_walk_and_compute_optimized(
            naverages = naverages,
            nobservations = nobservations,
            nvoid = nvoid,
            block_estimator = block_estimator,
            total_estimator = total_estimator,
            wavefunction = self.wavefunction,
            sampler = self.sampler)

        for key in sub_metrics:
            metrics[key] = sub_metrics[key]

        error, error_jf = total_estimator.finalize(naverages)
        energy = total_estimator.energy
        energy_jf = total_estimator.energy_jf
        acceptance = total_estimator.acceptance
        dpsi_i = total_estimator.dpsi_i
        dpsi_i_EL = total_estimator.dpsi_i_EL
        dpsi_ij = total_estimator.dpsi_ij
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
        dp_i = self.optimizer.sr(energy,dpsi_i,dpsi_i_EL,dpsi_ij)

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

        metrics['energy/energy']     = energy
        metrics['energy/error']      = error
        metrics['energy/energy_jf']  = energy_jf
        metrics['energy/error_jf']   = error_jf
        metrics['metropolis/acceptance'] = acceptance

        return delta_p, metrics

    # def average_over_iterations(self, x_current_list, hamiltonian, block_estimator)

    @tf.function
    def batched_jacobian(self, nobs, x_current_arr, wavefunction, jac_fnc):
        ret_jac = []
        ret_shape = []
        for i in range(nobs):
            flattened_jacobian, flat_shape = jac_fnc(x_current_arr[i], wavefunction)
            ret_jac.append(flattened_jacobian)
            ret_shape.append(flat_shape)

        return ret_jac, ret_shape

    # @profile
    def sr_walk_and_compute_optimized(self,naverages,nobservations,nvoid,block_estimator,total_estimator, wavefunction, sampler):
        kicker = tf.random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.2}

        # Now, we loop over the number of blocks.
        # For each block, accumulate nobservations times, with nvoid steps in between.
        total_estimator.reset()

        ke_jf_list      = []
        ke_direct_list  = []
        pe_list         = []

        for i_av in tf.range(naverages):
            block_estimator.reset()

            # Here, the loop over observations is optimized by running all
            # observations in parallel at the same time.
            acceptance = sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=nvoid)
            x_current  = sampler.sample()

            energy, energy_jf, ke_jf, ke_direct, pe = self.hamiltonian.energy(self.wavefunction, x_current)
            # print("energy.shape: ", energy.shape)
            # print("energy_jf.shape: ", energy_jf.shape)
            # print("ke_jf.shape: ", ke_jf.shape)
            # print("ke_direct.shape: ", ke_direct.shape)
            # print("pe.shape: ", pe.shape)

            # We break x_current into n_observations:
            x_current  = tf.split(x_current, nobservations, axis=0)
            energy     = tf.split(energy,    nobservations, axis=0)
            energy_jf  = tf.split(energy_jf, nobservations, axis=0)
            ke_jf      = tf.split(ke_jf, nobservations, axis=0)
            ke_direct  = tf.split(ke_direct, nobservations, axis=0)
            pe         = tf.split(pe, nobservations, axis=0)

            ke_jf     = [ tf.reduce_sum(_ke_jf).numpy() / self.nwalkers for _ke_jf in ke_jf]
            ke_direct = [ tf.reduce_sum(_ke_direct).numpy() / self.nwalkers for _ke_direct in ke_direct]
            pe        = [ tf.reduce_sum(_pe).numpy() / self.nwalkers for _pe in pe]
            ke_jf_list.append(tf.reduce_mean(ke_jf))
            ke_direct_list.append(tf.reduce_mean(ke_direct))
            pe_list.append(tf.reduce_mean(pe))


            # logger.debug(f"Kinetic Energy JF: {tf.reduce_sum(ke_jf, axis=-1).numpy()}")
            # logger.debug(f"Kinetic Energy direct: {tf.reduce_sum(ke_direct, axis=-1).numpy()}")
            #

            # flattened_jacobian is a list, flat_shape is tossed
            flattened_jacobian, flat_shape = self.batched_jacobian(nobservations, x_current, self.wavefunction, self.jacobian)

            for i_obs in range(nobservations):

                # Energy here is computed per walker.
                # Dividing by the number of walkers to normalize the energy
                obs_energy      = energy[i_obs]     / self.nwalkers
                obs_energy_jf   = energy_jf[i_obs]  / self.nwalkers

                dpsi_i, dpsi_ij, dpsi_i_EL = self.compute_O_observables(flattened_jacobian[i_obs], obs_energy)

                block_estimator.accumulate(
                    tf.reduce_sum(obs_energy),
                    tf.reduce_sum(obs_energy_jf),
                    acceptance,
                    1.,
                    dpsi_i,
                    dpsi_i_EL,
                    dpsi_ij,
                    1.)
            # Outside of the i_prop loop, we accumulate into the total_estimator:
            total_estimator.accumulate(
                block_estimator.energy,
                block_estimator.energy_jf,
                block_estimator.acceptance,
                0,
                block_estimator.dpsi_i,
                block_estimator.dpsi_i_EL,
                block_estimator.dpsi_ij,
                block_estimator.weight)

        # logger.debug(f"Kinetic Energy JF: {tf.reduce_mean(ke_jf_list):.2f} +- {tf.math.reduce_std(ke_jf_list):.2f}")
        # logger.debug(f"Kinetic Energy direct: {tf.reduce_mean(ke_direct_list):.2f} +- {tf.math.reduce_std(ke_direct_list):.2f}")
        # logger.debug(f"PE: {tf.reduce_mean(pe_list):.2f} +- {tf.math.reduce_std(pe_list):.2f}")

        metrics = {}
        metrics["energy/ke_jf"] = tf.reduce_mean(ke_jf_list)
        metrics["energy/ke"] = tf.reduce_mean(ke_direct_list)
        metrics["energy/pe"] = tf.reduce_mean(pe_list)
        metrics["energy/ke_jf_std"] = tf.math.reduce_std(ke_jf_list)
        metrics["energy/ke_std"] = tf.math.reduce_std(ke_direct_list)
        metrics["energy/pe_std"] = tf.math.reduce_std(pe_list)


        return total_estimator, flat_shape[0], metrics

    # @tf.function
    def sr_walk_and_compute(self,naverages,nobservations,nvoid,block_estimator,total_estimator, wavefunction, sampler):
        kicker = tf.random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.2}

        # Now, we loop over the number of blocks.
        # For each block, accumulate nobservations times, with nvoid steps in between.
        total_estimator.reset()

        for i_av in tf.range(naverages):
            block_estimator.reset()
            for i_prop in tf.range(nobservations):
                # Kick the walkers nvoid times:
                acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=nvoid)
                x_current = self.sampler.sample()

                # Energy here is computed per walker.
                energy, energy_jf = self.hamiltonian.energy(self.wavefunction, x_current)
                # Dividing by the number of walkers to normalize the energy
                energy = energy / self.nwalkers
                energy_jf = energy_jf / self.nwalkers

                flattened_jacobian, flat_shape = self.jacobian(x_current, self.wavefunction)

                dpsi_i, dpsi_ij, dpsi_i_EL = self.compute_O_observables(flattened_jacobian, energy)

                block_estimator.accumulate(
                    tf.reduce_sum(energy),
                    tf.reduce_sum(energy_jf),
                    acceptance,
                    1.,
                    dpsi_i,
                    dpsi_i_EL,
                    dpsi_ij,
                    1.)
            # Outside of the i_prop loop, we accumulate into the total_estimator:
            total_estimator.accumulate(
                block_estimator.energy,
                block_estimator.energy_jf,
                block_estimator.acceptance,
                0,
                block_estimator.dpsi_i,
                block_estimator.dpsi_i_EL,
                block_estimator.dpsi_ij,
                block_estimator.weight)



        return total_estimator, flat_shape

    def finalize(self):
        # Take the network and snapshot it to file:
        self.wavefunction.save_weights(self.model_path)
        print (self.model_path)
        # Save the global step:
        with open(self.save_path + "/global_step.pkl", 'wb') as _f:
            pickle.dump(self.global_step, file=_f)
        with open(self.save_path + "/optimizer.pkl", 'wb') as _f:
            pickle.dump(self.optimizer, file=_f)

    def interupt_handler(self, sig, frame):
        logger.info("Snapshoting weights...")
        self.active = False

if __name__ == "__main__":
    e = exec()
    signal.signal(signal.SIGINT, e.interupt_handler)

    # with tf.profiler.experimental.Profile('logdir'):
    e.run()
    e.finalize()
