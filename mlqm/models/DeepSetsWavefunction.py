import numpy
import tensorflow as tf
from mlqm import DEFAULT_TENSOR_TYPE

#from .ExponentialBoundaryCondition import ExponentialBoundaryCondition

class ResidualBlock(tf.keras.models.Model):
    """A dense layer with a bypass lane
    
    Computes the residual of the inputs.  Will error if n_output != n_input
    
    Extends:
        tf.keras.models.Model
    """
    def __init__(self, n_output, activation, use_bias):
        tf.keras.models.Model.__init__(self)

        self.layer = tf.keras.layers.Dense(n_output, activation = activation, use_bias = use_bias)

    def call(self, inputs):

        x = self.layer(inputs)

        return inputs + x


class DeepSetsWavefunction(tf.keras.models.Model):
    """Create a neural network eave function in N dimensions

    Boundary condition, if not supplied, is gaussian in every dimension

    Extends:
        tf.keras.models.Model
    """
    def __init__(self, ndim : int, nparticles: int, mean_subtract : bool, boundary_condition :tf.keras.layers.Layer = None):
        '''Deep Sets wavefunction for symmetric particle wavefunctions

        Implements a deep set network for multiple particles in the same system

        Arguments:
            ndim {int} -- Number of dimensions
            nparticles {int} -- Number of particls

        Keyword Arguments:
            boundary_condition {tf.keras.layers.Layer} -- [description] (default: {None})

        Raises:
            Exception -- [description]
        '''
        tf.keras.models.Model.__init__(self)

        self.ndim = ndim
        if self.ndim < 1 or self.ndim > 3:
           raise Exception("Dimension must be 1, 2, or 3 for DeepSetsWavefunction")

        self.nparticles = nparticles

        self.mean_subtract = mean_subtract

        self.activation  = tf.keras.activations.tanh
        # self.initializer = tf.keras.initializers.HeNormal

        self.individual_net = tf.keras.models.Sequential()
        self.individual_net.add(
            tf.keras.layers.Dense(32,
                use_bias = False)
            )
        self.individual_net.add(
            ResidualBlock(32,
                use_bias    = True,
                # kernel_initializer = self.initializer,
                activation = self.activation)
            )
        self.individual_net.add(
            ResidualBlock(32,
                use_bias = True,
                # kernel_initializer = self.initializer,
                activation = self.activation)
            )




        self.aggregate_net = tf.keras.models.Sequential()
        self.aggregate_net.add(
            ResidualBlock(32,
                use_bias = False,
                # kernel_initializer = self.initializer,
                activation = self.activation)
            )
        self.aggregate_net.add(
            ResidualBlock(32,
                use_bias = False,
                # kernel_initializer = self.initializer,
                activation = self.activation)
            )
        self.aggregate_net.add(tf.keras.layers.Dense(1,
            use_bias = False))


        # self.normalization_exponent = tf.Variable(2.0, dtype=DEFAULT_TENSOR_TYPE)
        # self.normalization_weight   = tf.Variable(-0.1, dtype=DEFAULT_TENSOR_TYPE)

    @tf.function(experimental_compile=False)
    def call(self, inputs, trainable=None):
        # Mean subtract for all particles:
        if self.nparticles > 1 and self.mean_subtract:
            mean = tf.reduce_mean(inputs, axis=1)
            xinputs = inputs - mean[:,None,:]
        else:
            xinputs = inputs

        x = []
        for p in range(self.nparticles):
            x.append(self.individual_net(xinputs[:,p,:]))

        x = tf.add_n(x)
        x = self.aggregate_net(x)

        # Compute the initial boundary condition, which the network will slowly overcome
        # boundary_condition = tf.math.abs(self.normalization_weight * tf.reduce_sum(xinputs**self.normalization_exponent, axis=(1,2))
        boundary_condition = -.2 * tf.reduce_sum(xinputs**2, axis=(1,2))
        boundary_condition = tf.reshape(boundary_condition, [-1,1])


        return x + boundary_condition

    def n_parameters(self):
        return tf.reduce_sum( [ tf.reduce_prod(p.shape) for p in self.trainable_variables ])
