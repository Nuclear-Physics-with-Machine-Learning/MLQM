import torch
import numpy


class MetropolisSampler(object):
    """Metropolis Sampler in N dimension
    
    Sample from N-D coordinates, using some initial probability distribution

    Relies on functional calls to sample on the fly with flexible distributions
    """
    def __init__(self, ndim : int, nwalkers : int, initializer : callable, init_params : iter ):
        '''Initialize a metropolis sampler
        
        Create a metropolis walker with `n` walkers.  Can use normal, uniform
        
        Arguments:
            ndim {int} -- Dimension
            nwalkers {int} -- Number of unique walkers
            initializer {callable} -- Function to call to initialize each walker
            init_params {iter} -- Parameters to pass to the initializer, unrolled automatically
        '''

        # Set the dimension:
        self.ndim = ndim

        # Set the number of walkers:
        self.nwalkers = nwalkers

        self.size = (self.nwalkers, self.ndim)

        self.walkers = initializer(*init_params, self.size)

        self.walkers.requires_grad_()

    def sample(self):
        '''Just return the current locations
        
        '''

        return self.walkers


    def kick(self, wavefunction : torch.nn.Module, kicker : callable, kicker_params : iter, device : str=None):
        """Sample points in N-d Space
        
        By default, samples points uniformly across all dimensions.
        Returns a torch tensor on the chosen device with gradients enabled.
        
        Keyword Arguments:
            device {str} -- What device to move data onto in torch (default: {None})
            kicker {callable} -- Function to call to create a kick for each walker
            kicker_params {iter} -- Parameters to pass to the kicker, unrolled automatically
        """

        # We need to compute the wave function twice:
        # Once for the original coordiate, and again for the kicked coordinates

        # Create a kick:
        kick = kicker(*kicker_params, self.size)

        with torch.no_grad():
            original = wavefunction(self.walkers)
            kicked   = wavefunction(self.walkers + kick)


            probability = torch.abs(kicked)**2 / torch.FloatTensor.abs(original)**2
            accept      = torch.ge(probability, torch.rand(size=[self.nwalkers]) )

            accept = accept.view([self.nwalkers, 1])

            self.walkers = torch.where(accept, self.walkers + kick, self.walkers)
            
            self.acceptance = torch.mean(accept.float())

        # Make sure the walkers compute gradients:
        self.walkers.requires_grad_()


        return self.acceptance




