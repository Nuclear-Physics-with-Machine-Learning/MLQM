import numpy
import tensorflow as tf

#from .ExponentialBoundaryCondition import ExponentialBoundaryCondition

class DeepSetsWavefunction(tf.keras.models.Model):
    """Create a neural network eave function in N dimensions

    Boundary condition, if not supplied, is gaussian in every dimension

    Extends:
        tf.keras.models.Model
    """
    def __init__(self, ndim : int, nparticles: int, boundary_condition :tf.keras.layers.Layer = None):
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

        self.individual_net = tf.keras.models.Sequential()
        self.individual_net.add(tf.keras.layers.Dense(32, use_bias = False, activation = tf.keras.activations.softplus))
        self.individual_net.add(tf.keras.layers.Dense(32, use_bias = False, activation = tf.keras.activations.softplus))
        self.individual_net.add(tf.keras.layers.Dense(100, use_bias = False))


        self.aggregate_net = tf.keras.models.Sequential()
        self.aggregate_net.add(tf.keras.layers.Dense(32, use_bias = False, activation = tf.keras.activations.softplus))
        self.aggregate_net.add(tf.keras.layers.Dense(32, use_bias = False, activation = tf.keras.activations.softplus))
        self.aggregate_net.add(tf.keras.layers.Dense(1, use_bias = False))

    # @tf.function
    def call(self, inputs, trainable=None):
        # Mean subtract for all particles:
        if self.nparticles > 1:
            mean = tf.reduce_mean(inputs, axis=1)
            xinputs = inputs - mean[:,None,:]
        else:
            xinputs = inputs

        x = []
        for p in range(self.nparticles):
            x.append(self.individual_net(xinputs[:,p,:]))

        x = tf.add_n(x)

        return self.aggregate_net(x)

    def n_parameters(self):
        return tf.reduce_sum( [ tf.reduce_prod(p.shape) for p in self.trainable_variables ])

    def flattened_params(self, params = None):
        '''Flatten the parameters of this model

        Additionally, store the raveling pattern to unflatten parameters.
        '''

        if params is None:
            params_to_flatten = self.trainable_variables
        indexs = []
        shapes = []
        params = []
        print(params_to_flatten)
        for p in params_to_flatten:
            shapes.append(p.shape)
            # Passing shape=[-1] to reshape flattens
            params.append(tf.reshape(p, shape=[-1]))
            indexs.append(tf.reduce_prod(p.shape))

            print(p.shape)

        print(params)
        return tf.concat(params, axis=0)