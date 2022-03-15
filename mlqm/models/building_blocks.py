import tensorflow as tf

from mlqm import DEFAULT_TENSOR_TYPE

class DenseBlock(tf.keras.models.Model):
    """A dense layer with a bypass lane

    Computes the residual of the inputs.  Will error if n_output != n_input

    Extends:
        tf.keras.models.Model
    """
    def __init__(self, n_output, use_bias, activation):
        tf.keras.models.Model.__init__(self, dtype=DEFAULT_TENSOR_TYPE)

        self.layer = tf.keras.layers.Dense(n_output,
            activation = activation, use_bias = use_bias,
            kernel_initializer = tf.keras.initializers.GlorotNormal,
            bias_initializer   = tf.keras.initializers.RandomNormal,
            )

    @tf.function(jit_compile=True)
    def __call__(self, inputs, training=None, mask=None):

        x = self.layer(inputs)
        return x


class ResidualBlock(DenseBlock):
    """A dense layer with a bypass lane

    Computes the residual of the inputs.  Will error if n_output != n_input

    Extends:
        tf.keras.models.Model
    """
    def __init__(self, n_output, use_bias, activation):
        DenseBlock.__init__(self, n_output, use_bias, activation)


    @tf.function(jit_compile=True)
    def __call__(self, inputs):

        x = self.layer(inputs)

        return inputs + x
