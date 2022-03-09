import numpy
import tensorflow as tf
from mlqm import DEFAULT_TENSOR_TYPE


import copy
#from .ExponentialBoundaryCondition import ExponentialBoundaryCondition

from . building_blocks import ResidualBlock, DenseBlock

class DeepSetsCorrelator(tf.keras.models.Model):
    """Create a neural network eave function in N dimensions

    Boundary condition, if not supplied, is gaussian in every dimension

    Extends:
        tf.keras.models.Model
    """
    def __init__(self, ndim : int, nparticles: int, configuration: dict):
        '''Deep Sets wavefunction for symmetric particle wavefunctions

        Implements a deep set network for multiple particles in the same system

        Arguments:
        :param      ndim:           Number of dimensions
        :type       ndim:           int
        :param      nparticles:     Number of particles
        :type       nparticles:     int
        :param      configuration:  Configuration parameters of the network
        :type       configuration:  dict

        Keyword Arguments:
            boundary_condition {tf.keras.layers.Layer} -- [description] (default: {None})

        Raises:
            Exception -- [description]
        '''
        tf.keras.models.Model.__init__(self)

        self.ndim = ndim
        if self.ndim < 1 or self.ndim > 3:
           raise Exception("Dimension must be 1, 2, or 3 for DeepSetsCorrelator")

        self.nparticles = nparticles

        self.config = configuration

        n_filters_per_layer = self.config.n_filters_per_layer
        n_layers            = self.config.n_layers
        bias                = self.config.bias
        residual            = self.config.residual

        try:
            activation = tf.keras.activations.__getattribute__(self.config['activation'])
        except Exception as e:
            print(e)
            print(f"Could not use the activation {self.config['activation']} - not in tf.keras.activations.")



        self.individual_net = tf.keras.models.Sequential()

        self.individual_net.add(
            DenseBlock(n_filters_per_layer,
                use_bias   = bias,
                activation = activation)
            )

        # The above layer counts as a layer!
        for l in range(n_layers-1):
            if l == n_layers - 2:
                _activation = None
            else:
                _activation = activation
            if residual:
                self.individual_net.add(
                    ResidualBlock(n_filters_per_layer,
                        use_bias    = bias,
                        activation = _activation)
                    )
            else:
                self.individual_net.add(
                    DenseBlock(n_filters_per_layer,
                        use_bias    = bias,
                        activation = _activation)
                    )


        self.aggregate_net = tf.keras.models.Sequential()

        for l in range(n_layers):
            if residual:
                self.aggregate_net.add(
                    ResidualBlock(n_filters_per_layer,
                        use_bias    = bias,
                        activation = activation)
                    )
            else:
                self.aggregate_net.add(
                    DenseBlock(n_filters_per_layer,
                        use_bias    = bias,
                        activation = activation)
                    )
        self.aggregate_net.add(tf.keras.layers.Dense(1,
            use_bias = False))

        self.confinement   = tf.constant(
                self.config.confinement,
                dtype = DEFAULT_TENSOR_TYPE
            )

        # self.normalization_exponent = tf.Variable(2.0, dtype=DEFAULT_TENSOR_TYPE)
        # self.normalization_weight   = tf.Variable(-0.1, dtype=DEFAULT_TENSOR_TYPE)

    def clone(self):

        new_ob = copy.deepcopy(self)

        new_ob.aggregate_net  = tf.keras.models.clone_model(self.aggregate_net)
        new_ob.individual_net = tf.keras.models.clone_model(self.individual_net)

        return new_ob

    @tf.function()
    def __call__(self, inputs, training=None):
        """
        If mean subtraction happens, it is in the call one layer up!
        """

        # ################################################
        # # Old way
        # x = []
        # for p in range(self.nparticles):
        #     x.append(self.individual_net(inputs[:,p,:]))
        #     # print("p: ", x[-1].shape)
        #
        # x = tf.add_n(x)
        # ################################################

        ################################################
        # Improved efficiency:
        transposed_inputs = tf.transpose(inputs, perm=(1,0,2))
        latent_space = tf.vectorized_map(lambda x : self.individual_net(x), transposed_inputs)
        x = tf.reduce_sum(latent_space, axis=0)


        x = self.aggregate_net(x)

        # # Compute the initial boundary condition, which the network will slowly overcome
        # # boundary_condition = tf.math.abs(self.normalization_weight * tf.reduce_sum(xinputs**self.normalization_exponent, axis=(1,2))
        boundary_condition = -self.confinement * tf.reduce_sum(inputs**2, axis=(1,2))
        boundary_condition = tf.reshape(boundary_condition, [-1,1])
        return x + boundary_condition
        # return x

    def n_parameters(self):
        return tf.reduce_sum( [ tf.reduce_prod(p.shape) for p in self.trainable_variables ])

    def restore_jax(self, model_path):
        import pickle


        with open(model_path, 'rb') as _f:
            weights = pickle.load(_f)

        i_this_model = 0
        i_jax = 0



        for w in weights:
            if len(w) == 0:
                # This is an activation layer
                continue
            elif len(w) == 2:
                # It's a weight and bias:
                target = self.trainable_variables[i_this_model]
                target.assign(w[0])
                i_this_model += 1; i_jax += 1

                target = self.trainable_variables[i_this_model]
                target.assign(w[1])
                i_this_model += 1; i_jax += 1
            else:
                # This is probably the FINAL layer:
                t = tf.convert_to_tensor(w)
                if t.shape == self.trainable_variables[i_this_model].shape:
                    self.trainable_variables[i_this_model].assign(t)

        return
