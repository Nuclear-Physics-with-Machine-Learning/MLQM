import torch
import numpy

class ExponentialBoundaryCondition(torch.nn.Module):
    """A simple module for applying an exponential boundary condition in N dimensions
    
    Note that the exponent is *inside* of the power of 2 in the exponent. 
    This is to prevent divergence when it is trainable and goes negative.
    
    Extends:
        torch.nn.Module
    """

    def __init__(self, n : int, exp : float=1.0, trainable : bool=True):
        """Initializer
        
        Create a new exponentional boundary condition
        
        Arguments:
            n {int} -- Number of dimensions
        
        Keyword Arguments:
            exp {float} -- Starting value of exponents.  Must be broadcastable to the number of dimensions (default: {1.0})
            trainable {bool} -- Whether to allow the boundary condition to be trainable (default: {True})
        """
        torch.nn.Module.__init__(self)



        if n < 1: 
            raise Exception("Dimension must be at least 1 for ExponentialBoundaryCondition")

        # Use numpy to broadcast to the right dimension:
        exp = numpy.asarray(exp, dtype=numpy.float32)
        exp = numpy.broadcast_to(exp, (n,))

        # This is the parameter controlling the shape of the exponent:
        self.exponent = torch.nn.Parameter(torch.tensor(exp), requires_grad=trainable)



    def forward(self, inputs):
        
        return torch.exp(- (self.exponent * inputs) **2 / 2.)
        


