import numpy
import torch

#from .ExponentialBoundaryCondition import ExponentialBoundaryCondition

class NeuralWavefunction(torch.nn.Module):
    """Create a neural network eave function in N dimensions
    
    Boundary condition, if not supplied, is gaussian in every dimension
    
    Extends:
        torch.nn.Module
    """
    def __init__(self, n : int, npart: int, boundary_condition :torch.nn.Module = None):
        torch.nn.Module.__init__(self)
        
        self.n = n
#        if self.n < 1 or self.n > 3: 
#            raise Exception("Dimension must be 1, 2, or 3 for NeuralWavefunction")

        self.npart = npart
        
        # Create a boundary condition if needed:
#        if boundary_condition is None:
#            self.bc = ExponentialBoundaryCondition(self.n)
#        else:
#            self.bc = boundary_condition

        self.layer1 = torch.nn.Linear(self.n * self.npart, 128)
#        self.layer2 = torch.nn.Linear(16, 16)
        self.layer3 = torch.nn.Linear(128, 1)
#        self.layer4 = torch.nn.Linear(16, 16)
#        self.layer5 = torch.nn.Linear(16, 16)
#        self.layer6 = torch.nn.Linear(16, 1)

# Test solution psi = exp(-(a*x+b)**2/2)
#        self.layer1 = torch.nn.Linear(self.n, 1)
#        self.layer1.weight.data.fill_(0.7)
#        self.layer1.bias.data.fill_(0.1)

#        self.exponent = torch.nn.Parameter(2. * torch.ones(self.n), requires_grad=True)
#        self.shift = torch.nn.Parameter(0.1 * torch.ones(self.n), requires_grad=True)

    def forward(self, inputs):
        x = torch.flatten(inputs, start_dim = 1)
        x = self.layer1(x)
#        x = torch.relu(x)
        x = torch.nn.functional.softplus(x, beta=1.)
#        x = torch.celu(x, True)
#        x = torch.tanh(x)
#        x = torch.sigmoid(x)
#        x = self.layer2(x)
#        x = torch.sigmoid(x)
        x = self.layer3(x)
#        x = torch.exp(x)       
        x = x.view([x.shape[0],]) 
        x = x - torch.sum( 0.01 * inputs**4, dim=(1,2))
        result = x
#        result = torch.exp(x)       

# Exact wave function
#        x = inputs
#        x = inputs + self.shift
#        x = torch.sum(self.exponent * x**2, dim=1)
#        x = 1. * (x[:,0]**2 + 2*x[:,1]**2)
#        x = 1. * (x[:,0]**2 + 2*x[:,1]**2 + 3*x[:,0]*x[:,1])
        
#        result = torch.exp( - x / 2)
        return result

    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def count_parameters(self):
        self.np =sum(p.numel() for p in self.parameters() )
        self.npt =sum(p.numel() for p in self.parameters() if p.requires_grad)

    def flatten_params(self, parameters):
    #flattens all parameters into a single column vector. Returns the dictionary to recover them
    #:param: parameters: a generator or list of all the parameters
    #:return: a dictionary: {"params": [#params, 1],
    #"indices": [(start index, end index) for each param] **Note end index in uninclusive**
    
        l = [torch.flatten(p) for p in parameters]
        indices = []
        s = 0
        for p in l:
            size = p.shape[0]
            indices.append((s, s+size))
            s += size
        flat = torch.cat(l).view(-1, 1)
        return flat, indices


    def flatten_grad(self, parameters):
    #flattens all parameters into a single column vector. Returns the dictionary to recover them
    #:param: parameters: a generator or list of all the parameters
    #:return: a dictionary: {"params": [#params, 1],
    #"indices": [(start index, end index) for each param] **Note end index in uninclusive**
    
        l = [torch.flatten(p.grad) for p in parameters]
        indices = []
        s = 0
        for p in l:
            size = p.shape[0]
            indices.append((s, s+size))
            s += size
        flat = torch.cat(l).view(-1, 1)
        return flat, indices



    def recover_flattened(self, flat_params, indices, model):
        """
        Gives a list of recovered parameters from their flattened form
        :param flat_params: [#params, 1]
        :param indices: a list detaling the start and end index of each param [(start, end) for param]
        :param model: the model that gives the params with correct shapes
        :return: the params, reshaped to the ones in the model, with the same order as those in the model
        """
        l = [flat_params[s:e] for (s, e) in indices]
        for i, p in enumerate(model.parameters()):
            l[i] = l[i].view(*p.shape)
        return l


