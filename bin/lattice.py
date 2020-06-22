import sys, os
import pathlib
import configparser

import argparse

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

        required_keys = ["delta", "min", "max"]

        for key in required_keys:
            if key not in sampler:
                raise Exception(f"Configuration for Sampler missing key {key}")

        from mlqm.samplers import CartesianSampler

        mins  = float(sampler['min'])
        maxes = float(sampler['max'])

        self.sampler = CartesianSampler(
            n           = self.dimension,
            nparticles  = self.nparticles,
            delta       = float(sampler['delta']),
            mins        = mins,
            maxes       = maxes,)

        return

    # def build_wavefunction(self):
    #     from mlqm.models import NeuralWavefunction

    #     self.wavefunction = NeuralWavefunction(self.dimension, self.nparticles)

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
        ho_w = DeepSetsWavefunction(self.dimension, self.nparticles)


        # from mlqm.models import PolynomialWavefunction
        # ho_w = PolynomialWavefunction(self.dimension, self.nparticles, degree=2)


        wavefunction = ho_w(x)

        energy, energy_by_parts = self.hamiltonian.energy(ho_w, x)


        optimizer = tf.keras.optimizers.Adam()

        for i in range(int(self.config['Optimization']['iterations'])):

            tape = tf.GradientTape(watch_accessed_variables = False)
            with tape:
                tape.watch(ho_w.trainable_variables)
                energy, energy_by_parts = self.hamiltonian.energy(ho_w, x)

                energy = tf.reduce_sum(energy * self.sampler.voxel_size())
                print("Energy: ", energy)

            # Unroll the tape and go backwards:
            gradients = tape.gradient(energy, ho_w.trainable_variables)

            optimizer.apply_gradients(zip(gradients,ho_w.trainable_variables))


if __name__ == "__main__":
    e = exec()
    e.run()


