import jax.numpy as numpy
import jax.random as random

from jax import vmap, jit

import flax.linen as nn
from typing import Tuple, Sequence

from . DeepSetsCorrelator import DeepSetsCorrelator, initialize_correlator


def initialize_wavefunction(walkers, spin, isospin, key, sampler_config, network_config):


    print(sampler_config)
    print(network_config)

    spatial_layers = [
        network_config.spatial_cfg.n_filters_per_layer \
        for l in range(network_config.spatial_cfg.n_layers) 
    ]

    spatial_layers[-1] = 1
    print(spatial_layers)

    i_layers = [
        network_config.deep_sets_cfg.n_filters_per_layer \
            for l in range(network_config.deep_sets_cfg.n_layers) 
    ]
    a_layers = [
        network_config.deep_sets_cfg.n_filters_per_layer \
            for l in range(network_config.deep_sets_cfg.n_layers) 
    ]
    a_layers[-1] = 1


    wavefunction = ManyBodyWavefunction(
        spatial_layers    = tuple(spatial_layers),
        individual_layers = tuple(i_layers),
        aggregate_layers  = tuple(a_layers),
        n_particles       = sampler_config.n_particles,
        n_spin_up         = sampler_config.n_spin_up,
        n_protons         = sampler_config.n_protons,
        mean_subtract     = network_config.mean_subtract,
        confinement       = network_config.deep_sets_cfg.confinement
    )


    wavefunction_variables = wavefunction.init(key, walkers[0], spin[0], isospin[0])
    # wavefunction.individual_network.init(key, walkers[0])

    # Call the wavefunction:
    w_out = wavefunction.apply(wavefunction_variables, walkers[0], spin[0], isospin[0])
    print(w_out.shape)

    wavefunction.apply = jit(vmap(jit(wavefunction.apply), in_axes=[None, 0, 0, 0]))

    return wavefunction, wavefunction_variables


class NeuralSpatialComponent(nn.Module):
    n_outputs: Tuple[int, ...]

    def setup(self):
        self.layers = [ nn.Dense(n_out) for n_out in self.n_outputs]


    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = nn.sigmoid(layer(out))
            
        return out


class ManyBodyWavefunction(nn.Module):
    """
    This class describes a many body wavefunction.

    Composed of several components:
     - a many-body, fully-symmetric correlator component based on the DeepSets wavefunction
     - Individual spatial wavefunctions based on single-particle neural networks

    The apply call computes the many-body correlated multipled by the slater determinant.
    """

    spatial_layers:    Tuple[int, ...]
    individual_layers: Tuple[int, ...]
    aggregate_layers:  Tuple[int, ...]
    n_particles:       int
    n_spin_up:         int
    n_protons:         int
    mean_subtract:     bool
    confinement:       float


    def setup(self):

        # How many particles? Create one network per particle:
        self.networks = [ 
            NeuralSpatialComponent(self.spatial_layers) \
                for p in range(self.n_particles)
        ]


        self.correlator = DeepSetsCorrelator(        
            individual_layers = self.individual_layers,
            aggregate_layers  = self.aggregate_layers,
            confinement       = self.confinement
        )

        # Here, we construct, up front, the appropriate spinor (spin + isospin)
        # component of the slater determinant.

        # The idea is that for each row (horizontal), the state in question
        # is ADDED to this matrix to get the right values.

        # We have two of these, one for spin and one for isospin. After the
        # addition of the spinors, apply a DOT PRODUCT between these two matrices
        # and then a DOT PRODUCT to the spatial matrix

        # Since the spinors are getting shuffled by the metropolis alg,
        # It's not relevant what the order is as long as we have
        # the total spin and isospin correct.

        # Since we add the spinors directly to this, we need
        # to have the following mappings:
        # spin up -> 1 for the first n_spin_up state
        # spin up -> 0 for the last nparticles - n_spin_up states
        # spin down -> 0 for the first n_spin_up state
        # spin down -> 1 for the last nparticles - n_spin_up states

        # so, set the first n_spin_up entries to 1, which
        # (when added to the spinor) yields 2 for spin up and 0 for spin down

        # Set the last nparticles - n_spin_up states to -1, which
        # yields 0 for spin up and -2 for spin down.

        # After doing the additions, it is imperative to apply a factor of 0.5!
        spin_spinor_2d = numpy.zeros(shape=(self.n_particles, self.n_particles))
        spin_spinor_2d = spin_spinor_2d.at[0:self.n_spin_up,:].set(1.)
        self.spin_spinor_2d = spin_spinor_2d.at[self.n_spin_up:,:].set(-1.)

        isospin_spinor_2d = numpy.zeros(shape=(self.n_particles, self.n_particles))
        isospin_spinor_2d = isospin_spinor_2d.at[0:self.n_protons,:].set(1.)
        self.isospin_spinor_2d = isospin_spinor_2d.at[self.n_protons:,:].set(-1.)


    def compute_spatial_slater(self, _xinputs):
        # We get individual response for every particle in the slater determinant
        # The axis of the particles is 1 (nwalkers, nparticles, ndim)

        # We also have to build a matrix of size [nparticles, nparticles] which we
        # then take a determinant of.

        # The axes of the determinant are state (changes vertically) and
        # particle (changes horizontally)

        # We need to compute the spatial components.
        # This computes each spatial net (nparticles of them) for each particle (nparticles)
        # You can see how this builds to an output shape of (Np, nP)
        #
        # Flatten the input for a neural network to compute
        # over all particles at once, for each network:

        # The matrix indexes in the end should be:
        # [walker, state, particle]
        # In other words, columns (y, 3rd index) share the particle
        # and rows (x, 2nd index) share the state


        # Apply the neural network to every particle at once:
        spatial_slater = [ n(_xinputs) for n in self.networks]
        # stack the output into a matrix:
        spatial_slater = numpy.concatenate(spatial_slater,axis=1)
        return spatial_slater

    def compute_spin_slater(self, spin, state_matrix):
        repeated_spin_spinor = numpy.tile(spin, reps = (1, self.n_particles))
        repeated_spin_spinor = numpy.reshape(repeated_spin_spinor, (-1, self.n_particles, self.n_particles))
        spin_slater = numpy.power(0.5*(repeated_spin_spinor + self.spin_spinor_2d), 2)

        return spin_slater


    # @tf.function
    # def construct_slater_matrix(self, inputs, spin, isospin):


    #     spatial_slater = self.compute_spatial_slater(inputs)
    #     if self.use_spin:
    #         spin_slater = self.compute_spin_slater(spin)
    #         spatial_slater *= spin_slater
    #     if self.use_isospin:
    #         isospin_slater = self.compute_isospin_slater(isospin)
    #         spatial_slater *= isospin_slater

    #     # Compute the determinant here
    #     # Determinant is not pos def
    #     #
    #     # When computed, it gets added to the deep sets correlator.
    #     #
    #     # we are using psi = e^U(X) * slater_det, but parametrizing log(psi)
    #     # with this wave function. Therefore, we return:
    #     # log(psi) = U(x) + log(slater_det)
    #     # Since the op we use below is `slogdet` and returns the sign + log(abs(det))
    #     # Then the total function is psi = e^(U(X))*[sign(det(S))*exp^(log(abs(det(S))))]
    #     # log(psi) = U(X) + log(sign(det(S)) +log(abs(det(S)))
    #     #

    #     return spatial_slater


    def __call__(self, x, spin, isospin):

        # First, do we mean subtract?
        if self.mean_subtract:
            mean = x.mean(axis=0)
            inputs = x - mean
        else:
            inputs = x

        # Call the correlator:
        correlation = self.correlator(inputs)

        # Compute the spin component of the spatial matrix:

        spin_state_matrix = self.compute_spin_slater(spin, self.spin_spinor_2d)

        isospin_state_matrix = self.compute_spin_slater(isospin, self.isospin_spinor_2d)

        spatial_slater = self.compute_spatial_slater(inputs)

        slater = spin_state_matrix * isospin_state_matrix * spatial_slater

        # Finally, the return value is the correlation * det(slater)
        w = correlation * numpy.linalg.det(slater)

        # Flatten it:
        return w.reshape(())




    #     return tf.reshape(wavefunction, (-1, 1))
