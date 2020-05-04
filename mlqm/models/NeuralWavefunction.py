import numpy
import torch

#from .ExponentialBoundaryCondition import ExponentialBoundaryCondition

class NeuralWavefunction(torch.nn.Module):
    """Create a neural network eave function in N dimensions
    
    Boundary condition, if not supplied, is gaussian in every dimension
    
    Extends:
        torch.nn.Module
    """
    def __init__(self, n : int, boundary_condition :torch.nn.Module = None):
        torch.nn.Module.__init__(self)
        
        self.n = n
        if self.n < 1 or self.n > 3: 
            raise Exception("Dimension must be 1, 2, or 3 for NeuralWavefunction")

        
        # Normalization:
        self.norm = 1.0
        
        # Create a boundary condition if needed:
#        if boundary_condition is None:
#            self.bc = ExponentialBoundaryCondition(self.n)
#        else:
#            self.bc = boundary_condition


        self.layer1 = torch.nn.Linear(self.n, 32)
        self.layer2 = torch.nn.Linear(32, 32)
        self.layer3 = torch.nn.Linear(32, 1)
    


    def forward(self, inputs):
        x = inputs
        x = self.layer1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        x = self.layer3(x)
        x = x.view([x.shape[0],])
       # boundary_condition = self.bc(inputs)
        result = x * torch.exp(-0.1*inputs.view([inputs.shape[0],])**2)

# Exact wave function
#        x = inputs
#        x = x.view([x.shape[0],])
#        result = torch.exp(-0.5*x**2)
        return result


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

