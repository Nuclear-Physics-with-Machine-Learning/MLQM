import torch
import numpy

from mlqm import H_BAR

class Hydrogen(object):
    """Hydrogen Hamiltonian
    
    Implementation of the Hydrogen atom hamiltonian
    """

    def __init__(self, mu : float, e : float):

        object.__init__(self)


        self.mu = mu
        self.e  = e

        # Several objects get stored for referencing, if needed, after energy computation:
        self.pe = None
        self.ke = None

    def potential_energy(self, *, wavefunction=None, inputs=None, delta=None, w_of_x=None):
        """Return potential energy
        
        If the potential energy is already computed, and no arguments are supplied,
        return the cached value

        If all arguments are supplied, calculate and return the PE.

        Otherwise, exception
        
        Arguments:
            wavefunction {Wavefunction model} -- Callable wavefunction object
            inputs {torch.Tensor} -- Tensor of shape [N, dimension], must have graph enabled
            delta {torch.Tensor} -- Integral Computation 'dx'
            w_of_x {torch.Tensor} -- Optional, can use a cached forward pass of the wavefunction
        
        Returns:
            torch.Tensor - potential energy of shape [1]
        """

        if self.pe is not None:
            if wavefunction is None and inputs is None and delta is None:
                return self.pe
        
        if wavefunction is None or inputs is None or delta is None:
            raise Exception("Must provide all or none of wavefunction, inputs, AND delta to potential_energy computation")
        
        if w_of_x is None:
            w_of_x = wavefunction(inputs)

        r = torch.sqrt(torch.sum(inputs**2, dim=1))
        self.pe = - (self.e**2 ) * torch.sum((w_of_x**2  / (r+0.00001)) * delta )

        return self.pe

    def kinetic_energy(self, *, w_prime_dx=None, delta = None):
        """Return Kinetic energy
        
        If the potential energy is already computed, and no arguments are supplied,
        return the cached value

        If all arguments are supplied, calculate and return the PE.

        Otherwise, exception
        
        Arguments:
            w_of_x {torch.Tensor} -- Computed derivative of the wavefunction
        
        Returns:
            torch.Tensor - potential energy of shape [1]
        """

        if self.ke is not None:
            if w_prime_dx is None and delta is None:
                return self.ke
        self.ke = (H_BAR**2 / (2 * self.mu)) * torch.sum(w_prime_dx**2 * delta)
        return self.ke


    def energy(self, wavefunction, inputs, delta):
        """Compute the expectation valye of energy of the supplied wavefunction.
        
        Computes the integral of the wavefunction in this potential
        
        Arguments:
            wavefunction {Wavefunction model} -- Callable wavefunction object
            inputs {torch.Tensor} -- Tensor of shape [N, dimension], must have graph enabled
            delta {torch.Tensor} -- Integral Computation 'dx'
        
        Returns:
            torch.tensor - Energy of shape [1]
        """


        # This function takes the inputs
        # And computes the expectation value of the energy.
        
        # This is the value of the wave function:
        w_of_x = wavefunction(inputs)

        # Sum the wavefunction and call backwards to get the derivative of it with respect to x:
        grad_catalyst = torch.sum(w_of_x)
        grad_catalyst.backward(retain_graph = True)
        # This is the first derivative of the wave function:
        
        w_prime_dx = inputs.grad
        
        # Now we can compute integrals:
        normalization = torch.sum(w_of_x**2 * delta)
        
        # print("normalization: ", normalization)

        pe = self.potential_energy(wavefunction=wavefunction, inputs=inputs, delta=delta, w_of_x=w_of_x) 
        
        # print("pe: ", pe)

        ke = self.kinetic_energy(w_prime_dx=w_prime_dx, delta=delta)
        
        # print("ke: ", ke)

        energy = (pe + ke) / normalization
        
        return energy


