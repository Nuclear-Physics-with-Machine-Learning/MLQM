import torch
import numpy

from .ExponentialBoundaryCondition import ExponentialBoundaryCondition
    
class PolynomialWavefunction(torch.nn.Module):
    """Implememtation of a Polynomial wave funtion in N dimensions
    
    [description]
    
    Extends:
        torch.nn.Module
    """

    def __init__(self,  n : int, degree : int, alpha : float, boundary_condition :torch.nn.Module = None):
        """Initializer
        
        Create a polynomial wave function with exponential boundary condition
        
        Arguments:
            n {int} -- Dimension of the oscillator (1 <= n <= 3)
            degree {int} -- Degree of the solution
            alpha {float} -- Alpha parameter (m * omega / hbar)
        
        Raises:
            Exception -- [description]
        """
        torch.nn.Module.__init__(self)
        
        self.n = n
        if self.n < 1 or self.n > 3: 
            raise Exception("Dimension must be 1, 2, or 3 for PolynomialWavefunction")

        # Use numpy to broadcast to the right dimension:
        degree = numpy.asarray(degree, dtype=numpy.int32)
        degree = numpy.broadcast_to(degree, (self.n,))

        # Degree of the polynomial:
        self.degree = degree
        
        if numpy.min(self.degree) < 1:
            raise Exception("Degree must be at least 1 in all dimensions")

        self.alpha = alpha
        
        # Normalization:
        self.norm  = numpy.power(self.alpha / numpy.pi, 0.25)
        
        # Initialize the polynomial coefficients:

        self.polynomial = torch.ones(size=(max(self.degree), self.n))

        if boundary_condition is None:
            self.bc = ExponentialBoundaryCondition(self.n)
        else:
            self.bc = boundary_condition



    def forward(self, inputs):
        x = inputs

        # compute the boundary condition:
        boundary_condition = self.bc(x)






        poly = self.const + self.linear * x + self.quad * x**2

        # Multiply by a factor to enforce normalization and boundary conditions:
        # Note the exponent is within the power of 2 to make it's sign irrelevant
        x = self.norm *  poly * boundary_condition
        return x
    
    def update_normalization(self, inputs):
        # Inputs is expected to be a range of parameters along an x axis.
        with torch.no_grad():
            value = self.forward(inputs)
            N = value ** 2

            delta = inputs[1]-inputs[0]

            N = torch.sum(N) * delta
            self.norm *= 1/torch.sqrt(N)

            # The normalization condition is that the integral of the wavefunction squared
            # should be equal to 1 (probability sums to 1.)

        return
    
    def analytic_derivative(self, inputs):
        
        poly = self.const + self.linear * x + self.quad * x**2
        poly_prime = self.linear + 2 * self.quad * inputs
        exp = self.exp(x)
        
        res = exp * (-inputs) *poly + poly_prime * exp
        
        return self.norm * res
    
    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.zero_()