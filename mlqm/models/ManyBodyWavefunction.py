import numpy
import tensorflow as tf
from mlqm import DEFAULT_TENSOR_TYPE

from . DeepSetsCorrelator import DeepSetsCorrelator
from mlqm.models.building_blocks import ResidualBlock, DenseBlock
# from . NeuralSpatialComponent import NeuralSpatialComponent

class ManyBodyWavefunction(tf.keras.models.Model):
    """
    This class describes a many body wavefunction.

    Composed of several components:
     - a many-body, fully-symmetric correlator component based on the DeepSets wavefunction
     - Individual spatial wavefunctions based on single-particle neural networks

    The forward call computes the many-body correlated multipled by the slater determinant.
    """

    def __init__(self, ndim : int, nparticles: int, configuration: dict):
        """
        Constructs a new instance of the ManyBodyWavefunction
        
        :param      ndim:           Number of dimensions
        :type       ndim:           int
        :param      nparticles:     Number of particles
        :type       nparticles:     int
        :param      configuration:  Configuration parameters of the network
        :type       configuration:  dict
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

        # self.spatial_nets = []
        # for i_particle in range(self.nparticles):
        #     self.spatial_nets.append(NeuralSpatialComponent(
        #         ndim          = self.ndim, 
        #         nparticles    = self.nparticles, 
        #         configuration = self.config.spatial_cfg)
        #     )

        self.simple_spatial_net = tf.keras.models.Sequential(
        [
            DenseBlock(nparticles, use_bias=True, activation="tanh"),
            DenseBlock(nparticles, use_bias=True, activation="tanh"),
        ]  
    )

    # @tf.function
    def __call__(self, inputs, spin, isospin, training=True):
            

        n_walkers = inputs.shape[0]

        # Mean subtract for all particles:
        if self.nparticles > 1 and self.mean_subtract:
            mean = tf.reduce_mean(inputs, axis=1)
            xinputs = inputs - mean[:,None,:]
        else:
            xinputs = inputs


        correlation = self.correlator(xinputs)

        # We get individual response for every particle in the slater determinant
        # The axis of the particles is 1 (nwalkers, nparticles, ndim)
        
        # We also have to build a matrix of size [nparticles, nparticles] which we
        # then take a determinant of.
        
        # The components of the spin in the wavefunction are always of the form:
        # (1/2)(1 +/- s_z) which maps to 0/1 for spin up, 1/0 for spin down in our coordinate system

        # Compute them up front:
        spin_plus  = 0.5*(1 + spin)
        spin_minus = 0.5*(1 - spin)

        # We also need to compute the spatial components.
        # This computes each spatial net (nparticles of them) for each particle (nparticles)
        # You can see how this builds to an output shape of (Np, nP)
        # 
        # Flatten the input for a neural network to compute
        # over all particles at once, for each network:
        flat_input = tf.reshape(xinputs, (-1, self.ndim)) # leave the spatial dimension

        flat_output = self.simple_spatial_net(flat_input)
        # print(flat_output.shape)

        # Reshape this to the proper shape for the slater determinant
        spatial_det = tf.reshape(flat_output, (n_walkers, self.nparticles, self.nparticles))

        # For spinors, we need to 

        # Compute the determinant here
        # Determinant is not pos def
        # 
        # When computed, it gets added to the deep sets correlator.
        # 
        # we are using psi = e^U(X) * slater_det, but parametrizing log(psi)
        # with this wave function. Therefore, we return:
        # log(psi) = U(x) + log(slater_det)
        
        slater_det = spatial_det


        sign, logdet = tf.linalg.slogdet(slater_det)

        # print("Logdet shape: ", logdet.shape)

        # print("correlation shape: ", correlation.shape)

        wavefunction = correlation + tf.reshape(logdet, (-1, 1))
        
        # print("Wavefunction shape: ", wavefunction.shape)

        return wavefunction



