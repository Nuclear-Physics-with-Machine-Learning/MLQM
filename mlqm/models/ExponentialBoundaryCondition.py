import tensorflow as tf
import numpy

class ExponentialBoundaryCondition(tf.keras.layers.Layer):
    """A simple module for applying an exponential boundary condition in N dimensions
    
    Note that the exponent is *inside* of the power of 2 in the exponent. 
    This is to prevent divergence when it is trainable and goes negative.
    
    Extends:
        tf.keras.layers.Layer
    """

    def __init__(self, n : int, nparticles : int, exp : float=0.1, trainable : bool=True):
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
            raise Exception("Dimension must be at least 1 for ExponentialBoundaryCondition")


        # This is the parameter controlling the shape of the exponent:
        self.exponent = tf.Variable(exp, trainable=trainable)



    def forward(self, inputs):
        
        # Reduce over the spatial dimension:
        r = tf.sqrt(tf.reduce_sum(inputs**2, dim=[2]) + 1e-8)
        # print(r.shape)
        exponent_term = tf.abs(self.exponent) * r / 2.
        # exponent_term = tf.abs(self.exponent) * r / 2.
        # print(exponent_term)
        # print(tf.sqrt(exponent_term))
        result = tf.exp(- exponent_term)
        # print(result)
        return result
        


