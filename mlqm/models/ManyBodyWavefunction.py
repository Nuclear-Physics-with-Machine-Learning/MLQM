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
        n_protons : int,
        use_spin: bool,
        use_isospin: bool,):
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

        self.use_spin = use_spin
        self.use_isospin = use_isospin

        # We need two components of this wavefunction:
        self.correlator  = DeepSetsCorrelator(
            ndim          = self.ndim,
            nparticles    = self.nparticles,
            configuration = self.config.deep_sets_cfg
        )

        # We need a spatial component for _every_ particle
        #
        self.spatial_nets = []
        if self.use_spin or self.use_isospin:
            for i in range(self.nparticles):
                self.spatial_nets.append(NeuralSpatialComponent(
                        configuration = self.config.spatial_cfg
                    )
                )
        # for i_particle in range(self.nparticles):
        #     if i_particle < 4:
        #         self.spatial_nets.append(n)

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

        if self.use_spin:
            spin_spinor_2d = numpy.zeros(shape=(nparticles, nparticles))
            spin_spinor_2d[0:n_spin_up,:] = 1
            spin_spinor_2d[n_spin_up:,:]  = -1

            self.spin_spinor_2d = tf.constant(spin_spinor_2d, dtype = DEFAULT_TENSOR_TYPE)


        if self.use_isospin:
            isospin_spinor_2d = numpy.zeros(shape=(nparticles, nparticles))
            isospin_spinor_2d[0:n_protons,:] = 1
            isospin_spinor_2d[n_protons:,:]  = -1

            self.isospin_spinor_2d = tf.constant(isospin_spinor_2d, dtype = DEFAULT_TENSOR_TYPE)

        # After doing the additions, it is imperative to apply a factor of 0.5!

    @tf.function(jit_compile=True)
    def compute_row(self, _wavefunction, _input):
        # Use the vectorized_map function to map over particles:
        transposed_inputs = tf.transpose(_input, perm=(1,0,2))
        # mapped_values = (this_input)
        temp_value = tf.vectorized_map(\
            lambda x : _wavefunction(x), transposed_inputs)
        temp_value = tf.reshape(temp_value,(transposed_inputs.shape[0], -1))
        temp_value = tf.transpose(temp_value)
        return temp_value

    @tf.function(jit_compile=True)
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

        nwalkers   = _xinputs.shape[0]
        nparticles = _xinputs.shape[1]

        # for i in range(nparticles):
        #
        slater_rows = [ tf.reshape(n(_xinputs),  (nwalkers, nparticles)) for n in self.spatial_nets]

        spatial_slater = tf.stack(slater_rows, axis=-1)
        spatial_slater = tf.transpose(spatial_slater, perm=(0,2,1))

        return spatial_slater

    @tf.function(jit_compile=True)
    def compute_spin_slater(self, spin):
        repeated_spin_spinor = tf.tile(spin, multiples = (1, self.nparticles))
        repeated_spin_spinor = tf.reshape(repeated_spin_spinor, (-1, self.nparticles, self.nparticles))
        spin_slater = tf.pow(0.5*(repeated_spin_spinor + self.spin_spinor_2d), 2)

        return spin_slater

    @tf.function(jit_compile=True)
    def compute_isospin_slater(self, isospin):

        repeated_isospin_spinor = tf.tile(isospin, multiples = (1, self.nparticles))
        repeated_isospin_spinor = tf.reshape(repeated_isospin_spinor, (-1, self.nparticles, self.nparticles))
        isospin_slater = tf.pow(0.5*(repeated_isospin_spinor + self.isospin_spinor_2d), 2)

        return isospin_slater

    @tf.function
    def construct_slater_matrix(self, inputs, spin, isospin):


        spatial_slater = self.compute_spatial_slater(inputs)
        if self.use_spin:
            spin_slater = self.compute_spin_slater(spin)
            spatial_slater *= spin_slater
        if self.use_isospin:
            isospin_slater = self.compute_isospin_slater(isospin)
            spatial_slater *= isospin_slater

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

        return spatial_slater


    # @tf.function(jit_compile=True)
    @tf.function(jit_compile=False)
    def __call__(self, inputs, spin=None, isospin=None):


        n_walkers = inputs.shape[0]
        rank      = inputs.shape[1]

        # Mean subtract for all particles:
        if self.nparticles > 1 and self.mean_subtract:
            mean = tf.reduce_mean(inputs, axis=1)
            xinputs = inputs - mean[:,None,:]
        else:
            xinputs = inputs

        correlation = self.correlator(xinputs)
        # return tf.math.exp(correlation)

        if self.use_spin or self.use_isospin:
            slater_matrix = self.construct_slater_matrix(xinputs, spin, isospin)
            # sign, logdet = tf.linalg.slogdet(slater_matrix)
            # det = sign * tf.exp(logdet)
            # det = tf.linalg.det(slater_matrix)
            det = self.custom_determinant(slater_matrix, rank)
            wavefunction = tf.math.exp(correlation) * tf.reshape(det, (-1, 1))
        else:
            wavefunction = tf.math.exp(correlation)

        # This code uses logdet:
        # sign, logdet = tf.linalg.slogdet(slater_matrix)
        #
        # wavefunction = correlation + tf.reshape(logdet, (-1, 1))
        #
        # return tf.reshape(sign, (-1, 1)) * tf.math.exp(wavefunction)

        # This code computes the determinant directly:



        return tf.reshape(wavefunction, (-1, 1))

    # @tf.function(jit_compile=True, experimental_relax_shapes=True)
    def sub_matrix(self, batch_matrix, row, column):
        left = batch_matrix[:,0:row,:]
        right = batch_matrix[:,row+1:,:]
        row_removed = tf.concat((left, right), axis=1)
        top = row_removed[:,:,0:column]
        bottom = row_removed[:,:,column+1:]
        return tf.concat((top, bottom), axis=2)

    # @tf.function(jit_compile=True, experimental_relax_shapes=True)
    def custom_determinant(self, _matrix, rank):



        # Here is a custom, maybe slower, determinant implementation.
        # It operates over the batch size

        # The matrix should be a size [N, m, m] where N is the batch size.

        # Implementing this recursively, so start with the base case:

        if rank == 1:
            return tf.reshape(_matrix, (-1,))
        else:
            # Need to get the submatrixes:
            sign = 1.0
            det  = 0.0
            for i in range(rank):
                sm = self.sub_matrix(_matrix, row=0, column=i)
                sub_det = sign *_matrix[:,0,i]*self.custom_determinant(sm, rank-1)
                contribution =  sub_det
                det += contribution
                sign *= -1.

            return det
