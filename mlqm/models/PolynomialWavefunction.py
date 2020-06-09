import tensorflow as tf
import numpy

from .ExponentialBoundaryCondition import ExponentialBoundaryCondition
    
class PolynomialWavefunction(tf.keras.models.Model):
    """Implememtation of a Polynomial wave funtion in N dimensions
    
    Create a polynomial, up to `degree` in every dimension `n`, fittable
    for optimization.

    Boundary condition, if not supplied, is gaussian like in every dimension.
    
    Extends:
        tf.keras.models.Model
    """

    def __init__(self,  n : int, nparticles : int, degree : int, boundary_condition :tf.keras.layers.Layer = None):
        """Initializer
        
        Create a polynomial wave function with exponential boundary condition
        
        Arguments:
            n {int} -- Dimension of the oscillator (1 <= n <= 3)
            nparticles {int} -- number of particles
            degree {int} -- Degree of the solution
            alpha {float} -- Alpha parameter (m * omega / hbar)
        
        Raises:
            Exception -- [description]
        """

        tf.keras.models.Model.__init__(self)
        
        self.n = n
        if self.n < 1 or self.n > 3: 
            raise Exception("Dimension must be 1, 2, or 3 for PolynomialWavefunction")

        if nparticles > 1:
            raise Exception("Polynomial wavefunction is only supported for one particle")

        # Use numpy to broadcast to the right dimension:
        degree = numpy.asarray(degree, dtype=numpy.int32)
        degree = numpy.broadcast_to(degree, (self.n,))

        # Degree of the polynomial:
        self.degree = degree
        
        if numpy.min(self.degree) < 0 or numpy.max(self.degree) > 4:
            raise Exception("Degree must be at least 0 in all dimensions")

        # Normalization:
        self.norm = 1.0
        

        # Craft the polynomial coefficients:

        # Add one to the degree since they start at "0"
        # Polynomial is of shape [degree, largest_dimension]
        self.polynomial = tf.Variable(
            initial_value = tf.random.normal(shape=(max(self.degree) +1 , self.n), dtype=tf.float32),
            trainable=True )

        # if boundary_condition is None:
        #     self.bc = ExponentialBoundaryCondition(self.n)
        # else:
        #     self.bc = boundary_condition



    def call(self, inputs, training=None):
        # Restrict to just one particle
        y = inputs[:,0,:]
        
        # Create the output tensor with the right shape, plus the constant term:
        polynomial_result = tf.zeros(y.shape)

        # This is a somewhat basic implementation:
        # Loop over degree:
        for d in range(max(self.degree) + 1):
            # Loop over dimension:

            # This is raising every element in the input to the d power (current degree)
            # This gets reduced by summing over all degrees for a fixed dimenions
            # Then they are reduced by multiplying over dimensions
            poly_term = y**d
            
            # Multiply every element (which is the dth power) by the appropriate 
            # coefficient in it's dimension
            res_vec = poly_term * self.polynomial[d]

            # Add this to the result:
            polynomial_result += res_vec

        # Multiply the results across dimensions at every point:
        polynomial_result = tf.reduce_prod(polynomial_result, axis=1)

        # boundary_condition = self.bc(y)

        # print(polynomial_result.shape)
        # print(boundary_condition.shape)
            
        return polynomial_result * self.norm
        # return boundary_condition * polynomial_result * self.norm

    
    def update_normalization(self, inputs, delta):
        # Inputs is expected to be a range of parameters along an x axis.
        value = self.call(inputs)
        
        print(value.shape)
        N = value ** 2


        N = tf.reduce_sum(N * delta)
        self.norm *= 1/tf.sqrt(N)

        return
    