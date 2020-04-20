import torch
import numpy

from .ExponentialBoundaryCondition import ExponentialBoundaryCondition
    
class PolynomialWavefunction(torch.nn.Module):
    """Implememtation of a Polynomial wave funtion in N dimensions
    
    Create a polynomial, up to `degree` in every dimension `n`, fittable
    for optimization.

    Boundary condition, if not supplied, is gaussian like in every dimension.
    
    Extends:
        torch.nn.Module
    """

    def __init__(self,  n : int, degree : int, boundary_condition :torch.nn.Module = None):
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
        
        if numpy.min(self.degree) < 0 or numpy.max(self.degree) > 4:
            raise Exception("Degree must be at least 0 in all dimensions")

        # Normalization:
        self.norm = 1.0
        

        # Craft the polynomial coefficients:

        # Add one to the degree since they start at "0"
        # Polynomial is of shape [degree, largest_dimension]
        self.polynomial = torch.ones(size=(max(self.degree) +1 , self.n))

        if boundary_condition is None:
            self.bc = ExponentialBoundaryCondition(self.n)
        else:
            self.bc = boundary_condition



    def forward(self, inputs):
        y = inputs
        
        # Create the output tensor with the right shape, plus the constant term:
        polynomial_result = torch.zeros(inputs.shape)

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
        polynomial_result = torch.prod(polynomial_result, dim=1)

        boundary_condition = self.bc(y)

            
        return boundary_condition * polynomial_result * self.norm

    
    def update_normalization(self, inputs, delta):
        # Inputs is expected to be a range of parameters along an x axis.
        with torch.no_grad():
            value = self.forward(inputs)
            N = value ** 2


            N = torch.sum(N * delta)
            self.norm *= 1/torch.sqrt(N)

            # The normalization condition is that the integral of the wavefunction squared
            # should be equal to 1 (probability sums to 1.)

        return
    

    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.zero_()