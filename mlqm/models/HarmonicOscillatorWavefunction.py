import tensorflow as tf
import numpy

from .GaussianBoundaryCondition import GaussianBoundaryCondition


class HarmonicOscillatorWavefunction(tf.keras.layers.Layer):
    """Implememtation of the harmonic oscillator wave funtions

    Create a polynomial, up to `degree` in every dimension `n`, that is the
    exact solution to the harmonic oscillator wave function.

    Extends:
        tf.keras.layers.Layer
    """

    def __init__(self,  n : int, nparticles : int, degree : int, alpha : float, dtype=tf.float64):
        """Initializer

        Create a harmonic oscillator wave function

        Arguments:
            n {int} -- Dimension of the oscillator (1 <= n <= 3)
            nparticles {int} -- Number of particles
            degree {int} -- Degree of the solution (broadcastable to n)
            alpha {float} -- Alpha parameter (m * omega / hbar)

        Raises:
            Exception -- [description]
        """
        tf.keras.layers.Layer.__init__(self, dtype=dtype)

        self.n = n
        if self.n < 1 or self.n > 3:
            raise Exception("Dimension must be 1, 2, or 3 for HarmonicOscillatorWavefunction")

        if nparticles > 1:
            raise Exception("HarmonicOscillatorWavefunction is only for 1 particle for testing.")

        # Use numpy to broadcast to the right dimension:
        degree = numpy.asarray(degree, dtype=numpy.int32)
        degree = numpy.broadcast_to(degree, (self.n,))

        self.type = dtype

        # Degree of the polynomial:
        self.degree = degree

        if numpy.min(self.degree) < 0 or numpy.max(self.degree) > 4:
            raise Exception("Only the first 5 hermite polynomials are supported")

        alpha = numpy.asarray(alpha, dtype=numpy.int32)
        alpha = numpy.broadcast_to(alpha, (self.n,))
        self.alpha = alpha

        # Normalization:
        self.norm = numpy.power(self.alpha / numpy.pi, 0.25)
        self.norm = numpy.prod(self.norm)


        # Craft the polynomial coefficients:

        # Add one to the degree since they start at "0"
        # Polynomial is of shape [degree, largest_dimension]
        self.polynomial = numpy.zeros(shape=(max(self.degree) + 1, self.n), dtype=numpy.float64)
        #  Loop over the coefficents and set them:

        # Loop over dimension:
        self.polynomial_norm = numpy.zeros(shape=(self.n,), dtype=numpy.float64)
        for _n in range(self.n):
            # Loop over degree:
            _d = self.degree[_n]
            if _d == 0:
                self.polynomial[0][_n] = 1.0
            elif _d == 1:
                self.polynomial[1][_n] = 2.0
            elif _d == 2:
                self.polynomial[0][_n] = -2.0
                self.polynomial[2][_n] = 4.0
            elif _d == 3:
                self.polynomial[1][_n] = -12.0
                self.polynomial[3][_n] = 8.0
            elif _d == 4:
                self.polynomial[0][_n] = 12.0
                self.polynomial[2][_n] = -48.0
                self.polynomial[4][_n] = 16.0

            # Set the polynomial normalization as a function of the degree
            # For each dimension:
            self.polynomial_norm[_n] = 1.0 / numpy.sqrt(2**_d * numpy.math.factorial(_d))

        self.polynomial_norm = tf.convert_to_tensor(self.polynomial_norm, dtype=self.type)
        self.polynomial      = tf.convert_to_tensor(self.polynomial, dtype=self.type)


        self.exp = GaussianBoundaryCondition(
            n=self.n, exp=numpy.sqrt(self.alpha), trainable=False, dtype=self.type)


    @tf.function
    def call(self, inputs):

        # Slice the inputs to restrict to just one particle:
        y = inputs[:,0,:]

        # Create the output tensor with the right shape, plus the constant term:
        polynomial_result = tf.zeros(y.shape,dtype=self.type)

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

        # Again restrict the BC to just one particle:
        boundary_condition = self.exp(inputs)[:,0]

        total_normalization = self.norm * tf.reduce_prod(self.polynomial_norm)
        epsilon = 1e-16
        # Add epsilon here to prevent underflow
        wavefunction = tf.math.log(boundary_condition * polynomial_result * total_normalization + epsilon)

        return tf.reshape(wavefunction, [-1, 1])
