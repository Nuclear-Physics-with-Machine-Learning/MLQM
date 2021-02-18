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
# tf.random.set_seed(2)

try:
    import horovod.tensorflow as hvd
    hvd.init()

    # This is to force each rank onto it's own GPU:
    if (hvd.size() != 1 ):
        # Only set this if there is more than one GPU.  Otherwise, its probably
        # Set elsewhere
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if hvd and len(gpus) > 0:
            tf.config.set_visible_devices(gpus[hvd.local_rank() % len(gpus)],'GPU')
    MPI_AVAILABLE=True
except:
    MPI_AVAILABLE=False

# Use mixed precision for inference (metropolis walk)
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

import logging
from logging import handlers
logger = logging.getLogger()


# Create a handler for STDOUT, but only on the root rank:
if not MPI_AVAILABLE or hvd.rank() == 0:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    handler = handlers.MemoryHandler(capacity = 100, target=stream_handler)
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
from mlqm.optimization import GradientCalculator, StochasticReconfiguration
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
        opt_parameters = [
            "iterations",
            "delta",
            "eps",
            "optimizer",
        ]

        for key in opt_parameters:
            if key not in optimization:
                raise Exception(f"Configuration for Optimization missing key {key}")

        self.iterations      = int(  optimization['iterations'])
        delta                = float(optimization['delta'])
        self.eps             = float(optimization['eps'])
        self.optimizer_type  = optimization['optimizer']


        sampler      = self.config['Sampler']
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
        self.save_path  = str(self.config["General"]["save_path"]) # Cast to pathlib later
        self.model_name = pathlib.Path(self.config["General"]["model_name"])

        if "profile" in self.config["General"]:
            self.profile = self.config["General"].getboolean("profile")
        else:
            self.profile = False

        self.set_compute_parameters()


        sampler     = self.build_sampler()
        hamiltonian = self.build_hamiltonian()

        x = sampler.sample()

        # Create a wavefunction:
        from mlqm.models import DeepSetsWavefunction

        wavefunction_config = self.config['Model']


        wavefunction = DeepSetsWavefunction(self.dimension, self.nparticles, wavefunction_config)

        # Run the wave function once to initialize all its weights
        _ = wavefunction(x)


        n_parameters = 0
        for p in wavefunction.trainable_variables:
            n_parameters += tf.reduce_prod(p.shape)

        logger.info(f"Number of parameters in this network: {n_parameters}")


        gradient_calc = GradientCalculator(self.eps, dtype = tf.float64)


        self.sr_worker   = StochasticReconfiguration(
            sampler                   = sampler,
            wavefunction              = wavefunction,
            hamiltonian               = hamiltonian,
            gradient_calc             = gradient_calc,
            delta                     = delta,
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


        # Now, cast to pathlib:
        self.save_path = pathlib.Path(self.save_path)

        # We also snapshot the configuration into the log dir:
        if not MPI_AVAILABLE or hvd.rank() == 0:
            with open(self.save_path / pathlib.Path('config.snapshot.ini'), 'w') as cfg:
                self.config.write(cfg)



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
            logger.info("Trying to restore model")

            # Inject control flow here to restore from Jax models.

            # Does the model exist?
            # Note that tensorflow adds '.index' and '.data-...' to the name
            tf_p = pathlib.Path(str(self.model_name) + ".index")

            # Check for tensorflow first:

            model_restored = False
            tf_found_path = None
            for source_path in [self.save_path, pathlib.Path('./')]:
                if (source_path / tf_p).is_file():
                    # Note: we use the original path without the '.index' added
                    tf_found_path = source_path / pathlib.Path(self.model_name)
                    logger.info(f"Resolved weights path is {tf_found_path}")
                    break

            if tf_found_path is None:
                raise OSError(f"{self.model_name} not found.")
            else:
                try:
                    self.sr_worker.wavefunction.load_weights(tf_found_path)
                    model_restored = True
                    logger.info("Restored from tensorflow!")
                except Exception as e:
                    logger.info("Failed to load weights via keras load_weights function.")

            # Now, check for JAX only if tf failed:
            if not model_restored:
                jax_p = pathlib.Path(self.model_name)
                jax_found_path = None
                for source_path in [self.save_path, pathlib.Path('./')]:
                    if (source_path / jax_p).is_file():
                        # Note: we use the original path without the '.index' added
                        jax_found_path = source_path / jax_p
                        logger.info(f"Resolved weights path is {jax_found_path}")
                        break

                if jax_found_path is None:
                    raise OSError(f"{self.model_name} not found.")
                else:
                    try:
                        self.sr_worker.wavefunction.restore_jax(jax_found_path)
                        logger.info("Restored from jax!")
                    except Exception as e:
                        logger.info("Failed to load weights via tensorflow or jax, returning")
                        return


            # We get here only if one method restored.
            # Attempt to restore a global step and optimizer but it's not necessary
            try:
                with open(self.save_path / pathlib.Path("global_step.pkl"), 'rb') as _f:
                    self.global_step = pickle.load(file=_f)
                with open(self.save_path / pathlib.Path("optimizer.pkl"), 'rb') as _f:
                    self.sr_worker.optimizer   = pickle.load(file=_f)
            except:
                logger.info("Could not restore a global_step or an optimizer state.  Starting over with restored weights only.")

    def set_compute_parameters(self):
        tf.keras.backend.set_floatx(DEFAULT_TENSOR_TYPE)
        tf.debugging.set_log_device_placement(False)
        tf.config.run_functions_eagerly(False)

        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

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


        #
        # x_test = tf.stack([
        #     0.01*tf.reshape(tf.linspace(0,11,12), (4,3)),
        #     -0.02*tf.reshape(tf.linspace(0,11,12), (4,3))
        # ])
        #
        # # Compute the energy with this test:
        # energy, energy_jf, ke_jf, ke_direct, pe = self.sr_worker.hamiltonian.energy(self.sr_worker.wavefunction, x_test)
        #
        # print(energy)
        # print(energy_jf)
        #
        # exit()


        # If the file for the model path already exists, we don't change it until after restoring:
        self.model_path = self.save_path / self.model_name


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
        self.sr_worker.equilibrate(self.n_thermalize)
        logger.info("Finished thermalization.")

        # Now, call once to compile:
        logger.info("About to compile.")
        self.sr_worker.compile()
        logger.info("Finished compilation.")

        checkpoint_iteration = 2000

        # Before beginning the loop, manually flush the buffer:
        logger.handlers[0].flush()

        while self.global_step < self.iterations:
            if not self.active: break

            if self.profile:
                if not MPI_AVAILABLE or hvd.rank() == 0:
                    tf.profiler.experimental.start(self.save_path)
                    tf.summary.trace_on(graph=True)


            start = time.time()

            metrics  = self.sr_worker.sr_step(n_thermalize = 1000)


            metrics['time'] = time.time() - start

            self.summary(metrics, self.global_step)

            if self.global_step % 1 == 0:
                logger.info(f"step = {self.global_step}, energy = {metrics['energy/energy'].numpy():.3f}, err = {metrics['energy/error'].numpy():.3f}")
                logger.info(f"step = {self.global_step}, energy_jf = {metrics['energy/energy_jf'].numpy():.3f}, err = {metrics['energy/error_jf'].numpy():.3f}")
                logger.info(f"acc  = {metrics['metropolis/acceptance'].numpy():.3f}")
                logger.info(f"time = {metrics['time']:.3f}")

            # Iterate:
            self.global_step += 1

            if checkpoint_iteration % self.global_step == 0:
                if not MPI_AVAILABLE or hvd.rank() == 0:
                    self.save_weights()
                    pass

            if self.profile:
                if not MPI_AVAILABLE or hvd.rank() == 0:
                    tf.profiler.experimental.stop()
                    tf.summary.trace_off()

        # Save the weights at the very end:
        self.save_weights()



    # @tf.function
    def summary(self, metrics, step):
        if not MPI_AVAILABLE or hvd.rank() == 0:
            with self.writer.as_default():
                for key in metrics:
                    tf.summary.scalar(key, metrics[key], step=step)

    def save_weights(self):
        # Take the network and snapshot it to file:
        self.sr_worker.wavefunction.save_weights(self.model_path)
        # Save the global step:
        with open(self.save_path / pathlib.Path("global_step.pkl"), 'wb') as _f:
            pickle.dump(self.global_step, file=_f)

    def finalize(self):
        if not MPI_AVAILABLE or hvd.rank() == 0:
            self.save_weights()

    def interupt_handler(self, sig, frame):
        logger.info("Snapshoting weights...")
        self.active = False

if __name__ == "__main__":
    e = exec()
    signal.signal(signal.SIGINT, e.interupt_handler)

    # with tf.profiler.experimental.Profile('logdir'):
    e.run()
    e.finalize()
