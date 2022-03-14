import numpy
import tensorflow as tf
from mlqm import DEFAULT_TENSOR_TYPE


from mlqm.models.building_blocks import ResidualBlock, DenseBlock

class NeuralSpatialComponent(tf.keras.models.Model):
    """Create a neural network wave function in N dimensions

    Applies a neural network to a _single_ particle

    Extends:
        tf.keras.models.Model
    """
    def __init__(self, ndim : int, nparticles: int, configuration: dict):
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



        self.net = []

        self.net.append(
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
                self.net.append(
                    ResidualBlock(n_filters_per_layer,
                        use_bias    = bias,
                        activation = _activation)
                    )
            else:
                self.net.append(
                    DenseBlock(n_output = n_filters_per_layer,
                        use_bias    = bias,
                        activation = _activation)
                    )

        self.net.append(tf.keras.layers.Dense(1,
            use_bias = False))


        # Represent the confinement as a function of r only, which is represented as a neural netowrk

        # self.confinement   = tf.constant(
        #         self.config.confinement,
        #         dtype = DEFAULT_TENSOR_TYPE
        #     )

        # self.normalization_exponent = tf.Variable(2.0, dtype=DEFAULT_TENSOR_TYPE)
        # self.normalization_weight   = tf.Variable(-0.1, dtype=DEFAULT_TENSOR_TYPE)


    @tf.function
    # @tf.function(jit_compile=True)
    def __call__(self, inputs, training=None):
        # return tf.constant(1.0, dtype=tf.float64)
        x = inputs
        for layer in self.net:
            x = layer(x)
        return x

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
