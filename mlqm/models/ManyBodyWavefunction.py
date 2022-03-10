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
        #
        self.spatial_nets = []
        if self.use_spin or self.use_isospin:
            for i in range(self.nparticles):
                self.spatial_nets.append(NeuralSpatialComponent(
                    ndim          = self.ndim,
                    nparticles    = self.nparticles,
                    configuration = self.config.spatial_cfg)
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

            # print("Spin spinor: ", self.spin_spinor_2d)

        if self.use_isospin:
            isospin_spinor_2d = numpy.zeros(shape=(nparticles, nparticles))
            isospin_spinor_2d[0:n_protons,:] = 1
            isospin_spinor_2d[n_protons:,:]  = -1

            self.isospin_spinor_2d = tf.constant(isospin_spinor_2d, dtype = DEFAULT_TENSOR_TYPE)
            # print("Isospin spinor: ", self.isospin_spinor_2d)

        # After doing the additions, it is imperative to apply a factor of 0.5!


    @tf.function
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

        @tf.function
        def compute_row(_wavefunction, _input):
            # Use the vectorized_map function to map over particles:
            transposed_inputs = tf.transpose(_input, perm=(1,0,2))
            # print(transposed_inputs)
            # mapped_values = (this_input)
            temp_value = tf.vectorized_map(\
                lambda x : _wavefunction(x), transposed_inputs)
            temp_value = tf.reshape(temp_value,(transposed_inputs.shape[0], -1))
            # )
            # print("temp_value: ", temp_value)
            temp_value = tf.transpose(temp_value)
            return temp_value

        slater_rows = [compute_row(w, _xinputs) for w in self.spatial_nets]
        # for j_state_function in range(self.nparticles):
            # slater_rows.append([])
            #
            # for i_particle in range(self.nparticles):
            #     this_input = _xinputs[:,i_particle,:]
            #     print("this_input.shape: ", this_input.shape)
            #     slater_rows[j_state_function].append(self.spatial_nets[j_state_function](this_input))
            #     # print(f"  {this_input} mapped to {slater_rows[i_particle][-1]}")
            # slater_rows[-1] = tf.concat(slater_rows[-1], axis=-1)



            # slater_rows.append(temp_value)


        # print("new slater rows: ", new_slater_rows)
        spatial_slater = tf.stack(slater_rows, axis=-1)
        spatial_slater = tf.transpose(spatial_slater, perm=(0,2,1))

        return spatial_slater

    @tf.function
    def compute_spin_slater(self, spin):
        repeated_spin_spinor = tf.tile(spin, multiples = (1, self.nparticles))
        repeated_spin_spinor = tf.reshape(repeated_spin_spinor, (-1, self.nparticles, self.nparticles))
        spin_slater = tf.pow(0.5*(repeated_spin_spinor + self.spin_spinor_2d), 2)

        return spin_slater

    @tf.function
    def compute_isospin_slater(self, isospin):

        # print("  Isospin: ", isospin)
        # print("  self.isospin_spinor_2d", self.isospin_spinor_2d)

        repeated_isospin_spinor = tf.tile(isospin, multiples = (1, self.nparticles))
        # print("  repeated_isospin_spinor: ", repeated_isospin_spinor)
        repeated_isospin_spinor = tf.reshape(repeated_isospin_spinor, (-1, self.nparticles, self.nparticles))
        # repeated_isospin_spinor = tf.transpose(repeated_isospin_spinor, perm=(0,2,1))
        # print("  repeated_isospin_spinor: ", repeated_isospin_spinor)
        isospin_slater = tf.pow(0.5*(repeated_isospin_spinor + self.isospin_spinor_2d), 2)

        return isospin_slater

    # @tf.function
    @tf.function
    def construct_slater_matrix(self, inputs, spin, isospin):


        spatial_slater = self.compute_spatial_slater(inputs)
        # print("spatial_slater: ", spatial_slater)
        spin_slater = self.compute_spin_slater(spin)
        # print("spin_slater: ", spin_slater)
        isospin_slater = self.compute_isospin_slater(isospin)
        # print("isospin_slater: ", isospin_slater)
        # print("spin-isospin slater: ", spin_slater * isospin_slater)

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

        # slater_matrix = spin_slater * isospin_slater
        # print("spin_slater: ", spin_slater)
        # print("isospin_slater: ", isospin_slater)
        # print("spatial_slater: ", spatial_slater)
        slater_matrix = spin_slater * isospin_slater * spatial_slater
        # print("slater_matrix: ", slater_matrix)
        return slater_matrix


    # @tf.function(jit_compile=True)
    # @profile
    @tf.function
    def __call__(self, inputs, spin=None, isospin=None, training=True):


        n_walkers = inputs.shape[0]

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
            det = tf.linalg.det(slater_matrix)
            # print("Det: ", det)
            wavefunction = tf.math.exp(correlation) * tf.reshape(det, (-1, 1))
            # print("correlation: ", correlation)
        else:
            wavefunction = tf.math.exp(correlation)

        # This code uses logdet:
        # sign, logdet = tf.linalg.slogdet(slater_matrix)
        #
        # wavefunction = correlation + tf.reshape(logdet, (-1, 1))
        # # print("Wavefunction shape: ", wavefunction.shape)
        #
        # return tf.reshape(sign, (-1, 1)) * tf.math.exp(wavefunction)

        # This code computes the determinant directly:



        return tf.reshape(wavefunction, (-1, 1))
