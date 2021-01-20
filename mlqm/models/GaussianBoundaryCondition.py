import tensorflow as tf
import numpy

class GaussianBoundaryCondition(tf.keras.layers.Layer):
    """A simple module for applying an exponential boundary condition in N dimensions

    Note that the exponent is *inside* of the power of 2 in the exponent.
    This is to prevent divergence when it is trainable and goes negative.

    Extends:
        tf.keras.layers.Layer
    """

    def __init__(self, n : int, exp : float=0.1, trainable : bool=True, dtype = tf.float64):
        """Initializer

        Create a new exponentional boundary condition

        Arguments:
            n {int} -- Number of dimensions

        Keyword Arguments:
            exp {float} -- Starting value of exponents.  Must be broadcastable to the number of dimensions (default: {1.0})
            trainable {bool} -- Whether to allow the boundary condition to be trainable (default: {True})
        """
        tf.keras.layers.Layer.__init__(self, dtype=dtype)

        self.mean_subtract = True

        if n < 1:
            raise Exception("Dimension must be at least 1 for GaussianBoundaryCondition")


        # This is the parameter controlling the shape of the exponent:
        self.exponent = tf.Variable(exp, trainable=True, dtype=dtype)
        self.exponent2 = tf.Variable(0.02, trainable=True, dtype=dtype)


    @tf.function
    def call(self, inputs):
        # Mean subtract for all particles:
        if  self.mean_subtract:
            mean = tf.reduce_mean(inputs, axis=1)
            xinputs = inputs - mean[:,None,:]
        else:
            xinputs = inputs

        exponent_term1 = tf.reduce_sum((xinputs)**2, axis=(1,2))
        exponent_term2 = tf.reduce_sum((xinputs)**4, axis=(1,2))
        result = - self.exponent * exponent_term1 - self.exponent2*exponent_term2

        return tf.reshape(result, [-1,1])
