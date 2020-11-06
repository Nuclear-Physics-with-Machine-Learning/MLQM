import tensorflow as tf
import numpy

class GaussianBoundaryCondition(tf.keras.layers.Layer):
    """A simple module for applying an exponential boundary condition in N dimensions

    Note that the exponent is *inside* of the power of 2 in the exponent.
    This is to prevent divergence when it is trainable and goes negative.

    Extends:
        tf.keras.layers.Layer
    """

    def __init__(self, n : int, exp : float=0.1, trainable : bool=True):
        """Initializer

        Create a new exponentional boundary condition

        Arguments:
            n {int} -- Number of dimensions

        Keyword Arguments:
            exp {float} -- Starting value of exponents.  Must be broadcastable to the number of dimensions (default: {1.0})
            trainable {bool} -- Whether to allow the boundary condition to be trainable (default: {True})
        """
        tf.keras.layers.Layer.__init__(self)



        if n < 1:
            raise Exception("Dimension must be at least 1 for GaussianBoundaryCondition")

        # Use numpy to broadcast to the right dimension:
        exp = numpy.asarray(exp, dtype=numpy.float64)
        exp = numpy.broadcast_to(exp, (n,))

        # This is the parameter controlling the shape of the exponent:
        self.exponent = tf.Variable(exp, trainable=trainable, dtype=tf.float64)


    @tf.function
    def call(self, inputs):
        exponent_term = tf.reduce_sum((self.exponent * inputs)**2, axis=2)
        result = tf.exp(- (exponent_term) / 2.)
        return result
