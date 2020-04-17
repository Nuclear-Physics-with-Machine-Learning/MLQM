import torch
import numpy

from .ExponentialBoundaryCondition import ExponentialBoundaryCondition


class HarmonicOscillator(torch.nn.Module):
    """Implememtation of the harmonic oscillator wave funtions
    
    [description]
    
    Extends:
        torch.nn.Module
    """

    def __init__(self,  n : int, degree : int, alpha : float):
        """Initializer
        
        Create a harmonic oscillator wave function
        
        Arguments:
            n {int} -- Dimension of the oscillator (1 <= n <= 3)
            degree {int} -- Degree of the solution (broadcastable to n)
            alpha {float} -- Alpha parameter (m * omega / hbar)
        
        Raises:
            Exception -- [description]
        """
        torch.nn.Module.__init__(self)
        
        self.n = n
        if self.n < 1 or self.n > 3: 
            raise Exception("Dimension must be 1, 2, or 3 for HarmonicOscillator")

        # Use numpy to broadcast to the right dimension:
        degree = numpy.asarray(degree, dtype=numpy.int32)
        degree = numpy.broadcast_to(degree, (self.n,))

        # Degree of the polynomial:
        self.degree = degree
        
        if numpy.min(self.degree) < 0 or numpy.max(self.degree) > 4:
            raise Exception("Only the first 5 hermite polynomials are supported")

        self.alpha = alpha
        
        # Normalization:
        self.norm  = numpy.power(self.alpha / numpy.pi, 0.25)
        

        # Craft the polynomial coefficients:

        # Add one to the degree since they start at "0"
        self.polynomial = torch.zeros(size=(max(self.degree) + 1, self.n))
        #  Loop over the coefficents and set them:

        print(self.polynomial)
        print(self.polynomial.shape)
        print(self.degree)       
        # Loop over dimension:
        for _n in range(self.n):
            # Loop over degree:
            _d = self.degree[_n]
            if _d == 0:
                self.polynomial[0][_n] = 1.0
            elif _d == 1:
                self.polynomial[0][_n] = 0.0
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


        print(self.polynomial)

        self.norm = torch.zeros(size=(self.n,))

        # Loop over dimension:
        for _n in range(self.n):
            print(_n)
            self.norm[_n] = 1.0 / numpy.sqrt(2**_n * numpy.math.factorial(_n))
    
        self.exp = ExponentialBoundaryCondition(n=self.n, exp=numpy.sqrt(self.alpha), trainable=False)
    
    def forward(self, inputs):
    
        y = inputs
        
        boundary_condition = self.exp(y)
        
        if self.n == 0:
            polynomial = 1
        elif self.n == 1:
            polynomial = y
        elif self.n == 2:
            polynomial = 2 * y**2 - 1
            
        return self.norm * boundary_condition * polynomial

