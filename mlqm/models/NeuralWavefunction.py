import numpy
import tensorflow as tf

#from .ExponentialBoundaryCondition import ExponentialBoundaryCondition

class NeuralWavefunction(tf.keras.models.Model):
    """Create a neural network eave function in N dimensions

    Boundary condition, if not supplied, is gaussian in every dimension

    Extends:
        tf.keras.models.Model
    """
    def __init__(self, ndim : int, nparticles: int, boundary_condition :tf.keras.layers.Layer = None):
        tf.keras.models.Model.__init__(self)

        self.ndim = ndim
        if self.ndim < 2 or self.ndim > 2:
           raise Exception("Dimension must be 2 for NeuralWavefunction")

        self.nparticles = nparticles

        # Create a boundary condition if needed:
#        if boundary_condition is None:
#            self.bc = ExponentialBoundaryCondition(self.ndim)
#        else:
#            self.bc = boundary_condition

        self.alpha = tf.Variable(0.1, dtype=tf.float64)
        self.beta  = tf.Variable(0.2, dtype=tf.float64)
        self.gamma = tf.Variable(30.0, dtype=tf.float64)


    def call(self, inputs):
        # This is expected to be exactly 2 dimensions.
        # shape is [walkers, particles, dim]

        a = inputs[:,:,0]
        b = inputs[:,:,1]
        c = a * b

        return -(self.alpha * a**2 + self.beta * b**2 + self.gamma * c)




    def n_parameters(self):
        return tf.reduce_sum( [ tf.reduce_prod(p.shape) for p in self.trainable_variables ])
