import sys, os
import pathlib
import configparser
import time

import argparse
import signal
import pickle


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=fusible"

import tensorflow as tf

try:
    import horovod.tensorflow as hvd
    hvd.init()

    # This is to force each rank onto it's own GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
    MPI_AVAILABLE=True
except:
    MPI_AVAILABLE=False

# Use mixed precision for inference (metropolis walk)
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

import logging
logger = logging.getLogger()

# Create a handler for STDOUT, but only on the root rank:
if not MPI_AVAILABLE or hvd.rank() == 0:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
else:
    # in this case, MPI is available but it's not rank 0
    # create a null handler
    handler = logging.NullHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


# Add the local folder to the import path:
mlqm_dir = os.path.dirname(os.path.abspath(__file__))
mlqm_dir = os.path.dirname(mlqm_dir)
sys.path.insert(0,mlqm_dir)
from mlqm import hamiltonians
from mlqm.samplers     import Estimator
from mlqm.optimization import Optimizer, StochasticReconfiguration
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
        if MPI_AVAILABLE:
            self.rank = hvd.rank()
            self.size = hvd.size()
            self.local_rank = hvd.local_rank()
        else:
            self.rank = 0
            self.size = 1
            self.local_rank = 1

        # Open the config file:
        self.config = configparser.ConfigParser()
        self.config.read(self.args.config_file)

        # Use this flag to catch interrupts, stop the next step and write output.
        self.active = True



        optimization = self.config['Optimization']
        sampler      = self.config['Sampler']

        opt_parameters = [
            "iterations",
            "delta",
            "eps",
            "gamma",
        ]

        for key in opt_parameters:
            if key not in optimization:
                raise Exception(f"Configuration for Optimization missing key {key}")

        self.iterations      = int(  optimization['iterations'])
        self.delta           = float(optimization['delta'])
        self.eps             = float(optimization['eps'])
        self.gamma           = float(optimization['gamma'])


        sampler_parameters = [
            "n_thermalize",
            "n_void_steps",
            "n_observable_measurements",
            "n_walkers_per_observation",
        ]
        for key in sampler_parameters:
            if key not in sampler:
                raise Exception(f"Configuration for Sampler missing key {key}")


        self.n_thermalize               = int(sampler['n_thermalize'])
        self.n_observable_measurements  = int(sampler['n_observable_measurements'])
        self.n_void_steps               = int(sampler['n_void_steps'])
        self.n_walkers_per_observation  = int(sampler['n_walkers_per_observation'])

        if "n_concurrent_obs_per_rank" in sampler:
            self.n_concurrent_obs_per_rank = int(sampler['n_concurrent_obs_per_rank'])
        else:
            self.n_concurrent_obs_per_rank = 1

        self.global_step     = 0

        self.dimension  = int(self.config['General']['dimension'])
        self.nparticles = int(self.config['General']['nparticles'])
        self.save_path  =     self.config["General"]["save_path"]
        self.model_name =     self.config["General"]["model_name"]

        if "profile" in self.config["General"]:
            self.profile = self.config["General"]["profile"]
        else:
            self.profile = False

        self.set_compute_parameters()


        sampler     = self.build_sampler()
        hamiltonian = self.build_hamiltonian()

        x = sampler.sample()

        # Create a wavefunction:
        from mlqm.models import DeepSetsWavefunction
        wavefunction = DeepSetsWavefunction(self.dimension, self.nparticles, mean_subtract=False)


        # Run the wave function once to initialize all its weights
        _ = wavefunction(x)

        from mlqm.optimization import Optimizer


        # Create an optimizer
        optimizer = Optimizer(
            delta   = self.delta,
            eps     = self.eps,
            npt     = wavefunction.n_parameters(),
            gamma   = self.gamma,
            dtype   = DEFAULT_TENSOR_TYPE)


        self.sr_worker   = StochasticReconfiguration(
            sampler                   = sampler,
            wavefunction              = wavefunction,
            hamiltonian               = hamiltonian,
            optimizer                 = optimizer,
            n_observable_measurements = self.n_observable_measurements,
            n_void_steps              = self.n_void_steps,
            n_walkers_per_observation = self.n_walkers_per_observation,
            n_concurrent_obs_per_rank = self.n_concurrent_obs_per_rank
        )





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

        if not MPI_AVAILABLE or hvd.rank() == 0:
            self.writer = tf.summary.create_file_writer(self.save_path)

        self.model_path = self.save_path + self.model_name

        # We also snapshot the configuration into the log dir:
        if not MPI_AVAILABLE or hvd.rank() == 0:
            with open(self.save_path + 'config.snapshot.ini', 'w') as cfg:
                self.config.write(cfg)


    # def build_sr_worker(self, sampler, wavefunction, hamiltonian, optimizer):

    #     sr_worker = StochasticReconfiguration(sampler, wavefunction, hamiltonian, optimizer)

    #     return sr_worker

    def build_sampler(self):

        from mlqm.samplers import MetropolisSampler

        # As an optimization, we increase the number of walkers by n_concurrent_obs_per_rank
        sampler = MetropolisSampler(
            n           = self.dimension,
            nparticles  = self.nparticles,
            nwalkers    = self.n_walkers_per_observation * self.n_concurrent_obs_per_rank,
            initializer = tf.random.normal,
            init_params = {"mean": 0.0, "stddev" : 0.2},
            dtype       = DEFAULT_TENSOR_TYPE)

        return sampler

    def check_potential_parameters(self, potential, parameters, config):

        for key in parameters:
            if key not in config:
                raise Exception(f"Configuration for {potential} missing key {key}")

    def build_hamiltonian(self):

        # First, ask for the type of hamiltonian
        self.hamiltonian_form = self.config["Hamiltonian"]["form"]

        # Is this hamiltonian in the options?
        if self.hamiltonian_form not in hamiltonians.__dict__:
            raise NotImplementedError(f"Hamiltonian {self.hamiltonian_form} is not found.")

        parameters = self.config["Hamiltonian"]
        parameters.pop("form")

        hamiltonian = hamiltonians.__dict__[self.hamiltonian_form](**parameters)

        return hamiltonian

    def restore(self):
        if not MPI_AVAILABLE or hvd.rank() == 0:
            self.sr_worker.wavefunction.load_weights(self.model_path)

            with open(self.save_path + "/global_step.pkl", 'rb') as _f:
                self.global_step = pickle.load(file=_f)
            with open(self.save_path + "/optimizer.pkl", 'rb') as _f:
                self.sr_worker.optimizer   = pickle.load(file=_f)


    def set_compute_parameters(self):
        tf.keras.backend.set_floatx(DEFAULT_TENSOR_TYPE)
        tf.debugging.set_log_device_placement(False)
        tf.config.run_functions_eagerly(False)

        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    # @profile
    def run(self):



        #
        # with self.writer.as_default():
        #     tf.summary.graph(self.wavefunction.get_concrete_function().graph)

        # We attempt to restore the weights:
        try:
            self.restore()
            logger.debug("Loaded weights, optimizer and global step!")
        except Exception as excep:
            logger.debug("Failed to load weights!")
            logger.debug(excep)
            pass


        if MPI_AVAILABLE and hvd.size() > 1:
            logger.info("Broadcasting initial model and optimizer state.")
            # We have to broadcast the wavefunction parameter here:
            hvd.broadcast_variables(self.sr_worker.wavefunction.variables, 0)

            # Also ned to broadcast the optimizer state:
            self.sr_worker.optimizer = hvd.broadcast_object(
                self.sr_worker.optimizer, root_rank=0)
            # And the global step:
            self.global_step = hvd.broadcast_object(
                self.global_step, root_rank=0)
            logger.info("Done broadcasting initial model and optimizer state.")

        # First step - thermalize:
        logger.info("About to thermalize.")
        self.sr_worker.equilibrate(1)
        self.sr_worker.equilibrate(self.n_thermalize)
        logger.info("Finished thermalization.")

        # Now, call once to compile:
        logger.info("About to compile.")
        self.sr_worker.compile()
        logger.info("Finished compilation.")

        while self.global_step < self.iterations:
            if not self.active: break

            if self.profile:
                tf.profiler.experimental.start(self.save_path)
                tf.summary.trace_on(graph=True)


            start = time.time()

            metrics  = self.sr_worker.sr_step()

            self.summary(metrics, self.global_step)

            self.sr_worker.update_model()

            if self.global_step % 1 == 0:
                logger.info(f"step = {self.global_step}, energy = {metrics['energy/energy'].numpy():.3f}, err = {metrics['energy/error'].numpy():.3f}")
                logger.info(f"step = {self.global_step}, energy_jf = {metrics['energy/energy_jf'].numpy():.3f}, err = {metrics['energy/error_jf'].numpy():.3f}")
                logger.info(f"acc  = {metrics['metropolis/acceptance'].numpy():.3f}")
                logger.info(f"time = {time.time() - start:.3f}")

            # Iterate:
            self.global_step += 1

            if self.profile:
                tf.profiler.experimental.stop()
                tf.summary.trace_off()


    # @tf.function
    def summary(self, metrics, step):
        if not MPI_AVAILABLE or hvd.rank() == 0:
            with self.writer.as_default():
                for key in metrics:
                    tf.summary.scalar(key, metrics[key], step=step)


    def finalize(self):
        if not MPI_AVAILABLE or hvd.rank() == 0:
            # Take the network and snapshot it to file:
            self.sr_worker.wavefunction.save_weights(self.model_path)
            # Save the global step:
            with open(self.save_path + "/global_step.pkl", 'wb') as _f:
                pickle.dump(self.global_step, file=_f)
            with open(self.save_path + "/optimizer.pkl", 'wb') as _f:
                pickle.dump(self.sr_worker.optimizer, file=_f)

    def interupt_handler(self, sig, frame):
        logger.info("Snapshoting weights...")
        self.active = False

if __name__ == "__main__":
    e = exec()
    signal.signal(signal.SIGINT, e.interupt_handler)

    # with tf.profiler.experimental.Profile('logdir'):
    e.run()
    e.finalize()
