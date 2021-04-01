import sys, os
import pathlib
import time

import signal
import pickle

# For configuration:
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.experimental import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log

hydra.output_subdir = None

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






# Add the local folder to the import path:
mlqm_dir = os.path.dirname(os.path.abspath(__file__))
mlqm_dir = os.path.dirname(mlqm_dir)
sys.path.insert(0,mlqm_dir)
from mlqm import hamiltonians
from mlqm.samplers     import Estimator
from mlqm.optimization import GradientCalculator, StochasticReconfiguration
from mlqm import DEFAULT_TENSOR_TYPE
from mlqm.models import DeepSetsWavefunction




class exec(object):

    def __init__(self, config):


        #
        if MPI_AVAILABLE:
            self.rank = hvd.rank()
            self.size = hvd.size()
            self.local_rank = hvd.local_rank()
        else:
            self.rank = 0
            self.size = 1
            self.local_rank = 1

        self.config = config

        self.configure_logger()
        logger = logging.getLogger()
        logger.info(OmegaConf.to_yaml(config))



        # Use this flag to catch interrupts, stop the next step and write output.
        self.active = True

        self.global_step     = 0

        self.save_path  = self.config["save_path"] # Cast to pathlib later
        self.model_name = pathlib.Path(self.config["model_name"])

        if "profile" in self.config:
            self.profile = bool(self.config["profile"])
        else:
            self.profile = False


        self.set_compute_parameters()


        sampler     = self.build_sampler()
        hamiltonian = self.build_hamiltonian()

        x = sampler.sample()


        wavefunction_config = self.config['wavefunction']

        # Create a wavefunction:
        wavefunction = DeepSetsWavefunction(self.config.dimension, self.config.nparticles, wavefunction_config)
        adaptive_wavefunction = DeepSetsWavefunction(self.config.dimension, self.config.nparticles, wavefunction_config)

        # Run the wave function once to initialize all its weights
        tf.summary.trace_on(graph=True, profiler=False)
        _ = wavefunction(x)
        _ = adaptive_wavefunction(x)
        tf.summary.trace_export("graph")
        tf.summary.trace_off()

        for w in adaptive_wavefunction.trainable_variables:
            w.assign(0. * w)

        n_parameters = 0
        for p in wavefunction.trainable_variables:
            n_parameters += tf.reduce_prod(p.shape)

        logger.info(f"Number of parameters in this network: {n_parameters}")
        #
        # x_test = tf.convert_to_tensor([[ 1.25291274, 1.15427136, -1.57162947],
        #                               [ 1.76117854, 0.26708064, -0.90399369]], dtype = DEFAULT_TENSOR_TYPE)
        # x_test = tf.reshape(x_test, (1,2,3))
        # print(x_test)
        #
        # print("Original: ", wavefunction.trainable_variables)
        #
        # # wavefunction.restore_jax("/home/cadams/ThetaGPU/AI-for-QM/full.model")
        # # print("Restored: ", wavefunction.trainable_variables)
        #
        # print(wavefunction(x_test))
        #
        # exit()

        # Read a target energy if it's there:
        self.target_energy = None
        if 'target_energy' in self.config:
            self.target_energy = self.config['target_energy']


        self.sr_worker   = StochasticReconfiguration(
            sampler          = sampler,
            wavefunction     = wavefunction,
            adaptive_wfn     = adaptive_wavefunction,
            hamiltonian      = hamiltonian,
            optimizer_config = self.config.optimizer,
            sampler_config   = self.config.sampler,
        )

        if not MPI_AVAILABLE or hvd.rank() == 0:
            # self.writer = tf.summary.create_file_writer(self.save_path)
            self.writer = tf.summary.create_file_writer(self.save_path + "/log/")


        # Now, cast to pathlib:
        self.save_path = pathlib.Path(self.save_path)


        # We also snapshot the configuration into the log dir:
        if not MPI_AVAILABLE or hvd.rank() == 0:
            with open(pathlib.Path('config.snapshot.yaml'), 'w') as cfg:
                OmegaConf.save(config=self.config, f=cfg)

    def configure_logger(self):

        logger = logging.getLogger()

        # Create a handler for STDOUT, but only on the root rank:
        if not MPI_AVAILABLE or hvd.rank() == 0:
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            handler = handlers.MemoryHandler(capacity = 10, target=stream_handler)
            logger.addHandler(handler)
            # Add a file handler:

            # Add a file handler too:
            log_file = self.config.save_path + "/process.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler = handlers.MemoryHandler(capacity=10, target=file_handler)
            logger.addHandler(file_handler)


            logger.setLevel(logging.DEBUG)
            # fh = logging.FileHandler('run.log')
            # fh.setLevel(logging.DEBUG)
            # logger.addHandler(fh)
        else:
            # in this case, MPI is available but it's not rank 0
            # create a null handler
            handler = logging.NullHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)


    def build_sampler(self):

        from mlqm.samplers import MetropolisSampler

        # As an optimization, we increase the number of walkers by n_concurrent_obs_per_rank
        n_walkers = self.config.sampler["n_walkers_per_observation"] * \
            self.config.sampler["n_concurrent_obs_per_rank"]

        sampler = MetropolisSampler(
            n           = self.config.dimension,
            nparticles  = self.config.nparticles,
            nwalkers    = n_walkers,
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
        self.hamiltonian_form = self.config["hamiltonian"]["form"]

        # Is this hamiltonian in the options?
        if self.hamiltonian_form not in hamiltonians.__dict__:
            raise NotImplementedError(f"hamiltonian {self.hamiltonian_form} is not found.")

        parameters = self.config["hamiltonian"]
        parameters = { p : parameters[p] for p in parameters.keys() if p != "form"}

        hamiltonian = hamiltonians.__dict__[self.hamiltonian_form](**parameters)

        return hamiltonian

    def restore(self):
        logger = logging.getLogger()
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


        logger = logging.getLogger()

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

            # And the global step:
            self.global_step = hvd.broadcast_object(
                self.global_step, root_rank=0)
            logger.info("Done broadcasting initial model and optimizer state.")



        # First step - thermalize:
        logger.info("About to thermalize.")
        self.sr_worker.equilibrate(self.config.sampler.n_thermalize)
        logger.info("Finished thermalization.")

        # Now, call once to compile:
        logger.info("About to compile.")
        self.sr_worker.compile()
        logger.info("Finished compilation.")

        checkpoint_iteration = 2000

        # Before beginning the loop, manually flush the buffer:
        logger.handlers[0].flush()

        best_energy = 999

        while self.global_step < self.config["iterations"]:
            if not self.active: break

            if self.profile:
                if not MPI_AVAILABLE or hvd.rank() == 0:
                    tf.profiler.experimental.start(str(self.save_path))
                    tf.summary.trace_on(graph=True)


            start = time.time()

            metrics = self.sr_worker.sr_step(n_thermalize = 1000)

            # Check if we've reached a better energy:
            if metrics['energy/energy'] < best_energy:
                best_energy = metrics['energy/energy']

                # If below the target energy, snapshot the weights as the best-yet
                if self.target_energy is None:
                    pass
                elif best_energy < self.target_energy:
                    if not MPI_AVAILABLE or hvd.rank() == 0:
                        self.save_weights(name="best_energy")
                        pass

            metrics['time'] = time.time() - start

            self.summary(metrics, self.global_step)

            # Add the gradients and model weights to the summary every 25 iterations:
            if self.global_step % 25 == 0:
                if not MPI_AVAILABLE or hvd.rank() == 0:
                    weights = self.sr_worker.wavefunction.trainable_variables
                    gradients = self.sr_worker.latest_gradients
                    self.model_summary(weights, gradients, self.global_step)
                    self.wavefunction_summary(self.sr_worker.latest_psi, self.global_step)


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
        if not MPI_AVAILABLE or hvd.rank() == 0:
            self.save_weights()


    def model_summary(self, weights, gradients, step):
        with self.writer.as_default():
            for w, g in zip(weights, gradients):
                tf.summary.histogram("weights/"   + w.name, w, step=step)
                tf.summary.histogram("gradients/" + w.name, g, step=step)

    def wavefunction_summary(self, latest_psi, step):
        with self.writer.as_default():
            tf.summary.histogram("psi", latest_psi, step=step)


    # @tf.function
    def summary(self, metrics, step):
        if not MPI_AVAILABLE or hvd.rank() == 0:
            with self.writer.as_default():
                for key in metrics:
                    tf.summary.scalar(key, metrics[key], step=step)


    def save_weights(self, name = "checkpoint" ):

        # If the file for the model path already exists, we don't change it until after restoring:
        self.model_path = self.save_path / pathlib.Path(name) / self.model_name

        # Take the network and snapshot it to file:
        self.sr_worker.wavefunction.save_weights(self.model_path)
        # Save the global step:
        with open(self.save_path /  pathlib.Path(name) / pathlib.Path("global_step.pkl"), 'wb') as _f:
            pickle.dump(self.global_step, file=_f)

    def finalize(self):
        if not MPI_AVAILABLE or hvd.rank() == 0:
            self.save_weights()

    def interupt_handler(self, sig, frame):
        logger.info("Snapshoting weights...")
        self.active = False


@hydra.main(config_path="../config", config_name="config")
def main(cfg : OmegaConf) -> None:

    # Prepare directories:
    work_dir = pathlib.Path(cfg.save_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir = pathlib.Path(cfg.save_path + "/log/")
    log_dir.mkdir(parents=True, exist_ok=True)

    # cd in to the job directory since we disabled that with hydra:
    # os.chdir(cfg.hydra.run.dir)
    e = exec(cfg)
    signal.signal(signal.SIGINT, e.interupt_handler)
    e.run()
    e.finalize()

if __name__ == "__main__":
    import sys
    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += ['hydra.run.dir=.', 'hydra/job_logging=disabled']
    main()
