import numpy
import torch


class NeuralWavefunction(torch.nn.Module):
    
    def __init__(self):
        torch.nn.Module.__init__(self)
        
        self.layer1 = torch.nn.Linear(1,32)
        self.layer2 = torch.nn.Linear(32, 32)
        self.layer3 = torch.nn.Linear(32, 1)
    
        self.norm   = torch.Tensor([1.0])
        
        # This is an exponent for normalization:
        self.exp = ExponentialBoundaryCondition()





    def forward(self, inputs):
        x = inputs
        x = self.layer1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        x = self.layer3(x)

        boundary_condition = self.exp(x)


        return self.norm*x*boundary_condition
    
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
    
    
    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.zero_()