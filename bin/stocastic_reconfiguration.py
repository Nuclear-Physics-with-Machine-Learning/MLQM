import sys, os
import pathlib
import configparser
import time

import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


import logging
logger = logging.getLogger()

# Create a handler for STDOUT:
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

'''
This script runs the nieve, lattice-based optimization.

'''


################

# Add the local folder to the import path:
mlqm_dir = os.path.dirname(os.path.abspath(__file__))
mlqm_dir = os.path.dirname(mlqm_dir)
sys.path.insert(0,mlqm_dir)
from mlqm.samplers     import Estimator


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


        # Open the config file:
        self.config = configparser.ConfigParser()
        self.config.read(self.args.config_file)

        self.dimension  = int(self.config['General']['dimension'])
        self.nparticles = int(self.config['General']['nparticles'])

        self.build_sampler()
        self.build_hamiltonian()


    def build_sampler(self):

        # First, check the sampler has all required config items:
        sampler = self.config["Sampler"]

        required_keys = ["nwalkers"]

        for key in required_keys:
            if key not in sampler:
                raise Exception(f"Configuration for Sampler missing key {key}")
        
        self.nwalkers   = int(sampler['nwalkers'])


        from mlqm.samplers import MetropolisSampler

        self.sampler = MetropolisSampler(
            n           = self.dimension,
            nparticles  = self.nparticles,
            nwalkers    = self.nwalkers,
            initializer = tf.random.normal, 
            init_params = {"mean": 0.0, "stddev" : 0.2})

        return

    def build_hamiltonian(self):

        from mlqm.hamiltonians import HarmonicOscillator

        self.hamiltonian = HarmonicOscillator(
            n           = self.dimension,
            nparticles  = self.nparticles,
            M           = float(self.config["Hamiltonian"]["mass"]),
            omega       = float(self.config["Hamiltonian"]["omega"]),
        )

    def run(self):
        print("running")
        x = self.sampler.sample()

        # For each dimension, randomly pick a degree
        degree = [ 1 for d in range(self.dimension)]

        from mlqm.models import DeepSetsWavefunction
        self.wavefunction = DeepSetsWavefunction(self.dimension, self.nparticles)


        from mlqm.optimization import Optimizer
        from mlqm.samplers     import Estimator

        optimization = self.config['Optimization']
        sr_parameters = ["neq","nav","nprop","nvoid","nblock"]
        for key in sr_parameters:
            if key not in optimization:
                raise Exception(f"Configuration for Optimization missing key {key}")
        
        # Run the wave function once to initialize all it's weights
        _ = self.wavefunction(x)

        # 
        energy, energy_by_parts = self.hamiltonian.energy(self.wavefunction, x)
        print(energy)

        self.sr_step(
            int(optimization['neq']), 
            int(optimization['nav']), 
            int(optimization['nprop']), 
            int(optimization['nvoid']))

        # optimizer = tf.keras.optimizers.Adam()

        # for i in range(int(self.config['Optimization']['iterations'])):

        #     tape = tf.GradientTape(watch_accessed_variables = False)
        #     with tape:
        #         tape.watch(ho_w.trainable_variables)
        #         energy, energy_by_parts = self.hamiltonian.energy(ho_w, x)

        #         energy = tf.reduce_sum(energy * self.sampler.voxel_size())
        #         print("Energy: ", energy)

        #     # Unroll the tape and go backwards:
        #     gradients = tape.gradient(energy, ho_w.trainable_variables)

        #     optimizer.apply_gradients(zip(gradients,ho_w.trainable_variables))


    # @tf.function
    def sr_step(self, neq, nav, nprop, nvoid):
        nblock = neq + nav
        nstep = nprop * nvoid
        block_estimator = Estimator(info=None)
        block_estimator.reset()
        total_estimator = Estimator(info=None)
        total_estimator.reset()

        # Metropolis sampler will sample configurations uniformly:
        # Sample initial configurations uniformy between -sig and sig
        x_original = self.sampler.sample()

        kicker = tf.random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.2}


        start = time.time()

        for i_block in range (nblock):
            block_estimator.reset()
            if (i_block == neq) :
               total_estimator.reset()
            for j_step in range (nstep):

                # Here, we update the position of each particle for SR:
                acceptance = self.sampler.kick(shape, walkers, self.wavefunction, kicker, kicker_params)
                x_current = self.sampler.sample()


                # Compute energy and accumulate estimators within a given block
                if ( (j_step+1) % nvoid == 0 and i_block >= neq ):
                    energy, energy_jf = self.hamiltonian.energy(self.wavefunction, x_current)
                    energy = energy / self.nwalkers
                    energy_jf = energy_jf / self.nwalkers
                    # energy.detach_()
                    # energy_jf.detach_()

                    # Compute < O^i_step >, < H O^i_step >,  and < O^i_step O^j_step > 
                    tape = tf.GradientTape()
                    with tape:
                        log_wpsi = self.wavefunction(x_current)


                    # flat_params = self.wavefunction.flattened_params()

                    jac = tape.jacobian(log_wpsi, self.wavefunction.trainable_variables)
                    # jac = torch.zeros(size=[self.nwalkers,wavefunction.npt])
                    # for n in range(self.nwalkers):
                    #     log_wpsi_n = log_wpsi[n]
                    #     wavefunction.zero_grad()
                    #     params = wavefunction.parameters()
                    #     dpsi_dp = torch.autograd.grad(log_wpsi_n, params, retain_graph=True)
                    #     dpsi_i_n, indeces_flat = wavefunction.flatten_params(dpsi_dp)
                    #     jac[n,:] = torch.t(dpsi_i_n)

                    end = time.time()

                    print(f"Time to here: {end - start}")

                    print(self.wavefunction.trainable_variables[0].shape)
                    print(jac[0].shape)
                    print(jac.shape)

    #                log_wpsi_n.detach_()
                    dpsi_i = torch.sum(jac, dim=0) / nwalk
                    dpsi_i = dpsi_i.view(-1,1)
                    dpsi_i_EL = torch.matmul(energy, jac).view(-1,1)
                    dpsi_ij = torch.mm(torch.t(jac), jac) / nwalk

    #                print("dpsi_i", dpsi_i)
    #                print("dpsi_ij", dpsi_ij)
    #                exit()
                    block_estimator.accumulate(torch.sum(energy),torch.sum(energy_jf),acceptance,1.,dpsi_i,dpsi_i_EL,dpsi_ij,1.)
    # Accumulate block averages
            if ( i_block >= neq ):
                total_estimator.accumulate(block_estimator.energy,block_estimator.energy_jf,block_estimator.acceptance,0,block_estimator.dpsi_i,
                    block_estimator.dpsi_i_EL,block_estimator.dpsi_ij,block_estimator.weight)

        error, error_jf = total_estimator.finalize(nav)
        energy = total_estimator.energy
        energy_jf = total_estimator.energy_jf
        acceptance = total_estimator.acceptance
        dpsi_i = total_estimator.dpsi_i
        dpsi_i_EL = total_estimator.dpsi_i_EL
        dpsi_ij = total_estimator.dpsi_ij

        logger.info(f"psi norm{torch.mean(log_wpsi)}")

        with torch.no_grad(): 
            dp_i = opt.sr(energy,dpsi_i,dpsi_i_EL,dpsi_ij)
            gradient = wavefunction.recover_flattened(dp_i, indeces_flat, wavefunction)
            delta_p = [ g for g in gradient]

        return energy, error, energy_jf, error_jf, acceptance, delta_p


if __name__ == "__main__":
    e = exec()
    e.run()


