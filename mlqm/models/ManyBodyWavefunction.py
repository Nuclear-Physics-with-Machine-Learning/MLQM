import numpy
import tensorflow as tf
from mlqm import DEFAULT_TENSOR_TYPE

from . DeepSetsCorrelator import DeepSetsCorrelator
from mlqm.models.building_blocks import ResidualBlock, DenseBlock
from . NeuralSpatialComponent import NeuralSpatialComponent

class ManyBodyWavefunction(tf.keras.models.Model):
    """
    This class describes a many body wavefunction.

    Composed of several components:
     - a many-body, fully-symmetric correlator component based on the DeepSets wavefunction
     - Individual spatial wavefunctions based on single-particle neural networks

    The forward call computes the many-body correlated multipled by the slater determinant.
    """

    def __init__(self, ndim : int,
        nparticles: int,
        configuration: dict,
        n_spin_up : int,
        n_protons : float):
        """
        Constructs a new instance of the ManyBodyWavefunction


        :param      ndim:           Number of dimensions
        :type       ndim:           int
        :param      nparticles:     Number of particles
        :type       nparticles:     int
        :param      configuration:  Configuration parameters of the network
        :type       configuration:  dict
        :param      n_spin_up:      The spin
        :type       n_spin_up:      int
        :param      n_protons:      The n_protons
        :type       n_protons:      int
        """

        tf.keras.models.Model.__init__(self)

        self.ndim = ndim
        if self.ndim < 1 or self.ndim > 3:
           raise Exception("Dimension must be 1, 2, or 3 for ManyBodyWavefunction")

        self.nparticles = nparticles

        self.config = configuration

        self.mean_subtract = self.config.mean_subtract

        # try:
        #     activation = tf.keras.activations.__getattribute__(self.config['activation'])
        # except e:
        #     print(e)
        #     print(f"Could not use the activation {self.config['activation']} - not in tf.keras.activations.")

        # We need two components of this wavefunction:
        self.correlator  = DeepSetsCorrelator(
            ndim          = self.ndim,
            nparticles    = self.nparticles,
            configuration = self.config.deep_sets_cfg
        )

        # We need a spatial component for _every_ particle

        self.spatial_nets = []
        for i_particle in range(self.nparticles):
            self.spatial_nets.append(NeuralSpatialComponent(
                ndim          = self.ndim,
                nparticles    = self.nparticles,
                configuration = self.config.spatial_cfg)
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

        spin_spinor_2d = numpy.zeros(shape=(nparticles, nparticles))
        spin_spinor_2d[0:n_spin_up,:] = 1
        spin_spinor_2d[n_spin_up:,:]  = -1

        self.spin_spinor_2d = tf.constant(spin_spinor_2d, dtype = DEFAULT_TENSOR_TYPE)


        isospin_spinor_2d = numpy.zeros(shape=(nparticles, nparticles))
        isospin_spinor_2d[0:n_protons,:] = 1
        isospin_spinor_2d[n_protons:,:]  = -1

        self.isospin_spinor_2d = tf.constant(isospin_spinor_2d, dtype = DEFAULT_TENSOR_TYPE)

        # After doing the additions, it is imperative to apply a factor of 0.5!


    # @tf.function
    def compute_spatial_slater(self, xinputs):
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

        slater_rows = []
        for i_particle in range(self.nparticles):
            this_input = xinputs[:,i_particle,:]

            slater_rows.append(tf.concat([self.spatial_nets[j_state_function](this_input) \
                for j_state_function in range(self.nparticles) ],
                axis=-1)
            )
            # print(slater_rows)
        spatial_slater = tf.stack(slater_rows, axis=-1)
        # spatial_slater = tf.transpose(spatial_slater, perm=(0,2,1))


        return spatial_slater

    # @tf.function
    # @tf.function(jit_compile=True)
    def __call__(self, inputs, spin, isospin, training=True):


        n_walkers = inputs.shape[0]

        # Mean subtract for all particles:
        if self.nparticles > 1 and self.mean_subtract:
            mean = tf.reduce_mean(inputs, axis=1)
            xinputs = inputs - mean[:,None,:]
        else:
            xinputs = inputs




        correlation = self.correlator(xinputs)


        spatial_slater = self.compute_spatial_slater(xinputs)


        repeated_spin_spinor = tf.tile(spin, multiples = (1, self.nparticles))
        repeated_spin_spinor = tf.reshape(repeated_spin_spinor, (-1, self.nparticles, self.nparticles))
        spin_slater = 0.5*(repeated_spin_spinor + self.spin_spinor_2d)


        repeated_isospin_spinor = tf.tile(isospin, multiples = (1, self.nparticles))
        repeated_isospin_spinor = tf.reshape(repeated_isospin_spinor, (-1, self.nparticles, self.nparticles))
        isospin_slater = 0.5*(repeated_isospin_spinor + self.isospin_spinor_2d)


        # Compute the determinant here
        # Determinant is not pos def
        #
        # When computed, it gets added to the deep sets correlator.
        #
        # we are using psi = e^U(X) * slater_det, but parametrizing log(psi)
        # with this wave function. Therefore, we return:
        # log(psi) = U(x) + log(slater_det)
        # Since the op we use below is `slogdet` and returns the sign + log(abs(det))
        # Then the total function is psi = e^(U(X))*[sign(det(S))*exp^(log(abs(det(S))))]
        # log(psi) = U(X) + log(sign(det(S)) +log(abs(det(S)))
        #

        slater_det = spin_slater * isospin_slater * spatial_slater


        sign, logdet = tf.linalg.slogdet(slater_det)

        wavefunction = correlation + tf.reshape(logdet, (-1, 1))

        # print("Wavefunction shape: ", wavefunction.shape)

        return wavefunction, sign
