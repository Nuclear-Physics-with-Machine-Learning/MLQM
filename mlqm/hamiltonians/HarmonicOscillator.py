import torch
import numpy

from mlqm import H_BAR

class HarmonicOscillator(object):
    """Harmonic Oscillator Potential
    
    Implementation of the quantum harmonic oscillator hamiltonian
    """

    def __init__(self,  n : int, M : float, omega : float):

        object.__init__(self)


        self.n = n
        if self.n < 1 or self.n > 3: 
            raise Exception("Dimension must be 1, 2, or 3 for HarmonicOscillator")

        self.M = M

        self.omega = omega

        # Several objects get stored for referencing, if needed, after energy computation:
        self.pe = None
        self.ke = None
        self.ke_by_parts = None

    def potential_energy(self, *, wavefunction=None, inputs=None, w_of_x=None):
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
            if wavefunction is None and inputs is None:
                return self.pe
        
        if wavefunction is None or inputs is None:
            raise Exception("Must provide all or none of wavefunction, inputs, AND delta to potential_energy computation")
        
        if w_of_x is None:
            w_of_x = wavefunction(inputs)

        # x Squared needs to contract over spatial dimensions:
        x_squared = torch.sum(inputs**2, dim=-1)

        self.pe = (0.5 * self.M * self.omega**2 ) * w_of_x**2 * x_squared 

        return self.pe

    def kinetic_energy_by_parts(self, *, dw_dx=None, ):
        """Return Kinetic energy
        
        If the potential energy is already computed, and no arguments are supplied,
        return the cached value

        If all arguments are supplied, calculate and return the KE.

        Otherwise, exception
        
        Arguments:
            w_of_x {torch.Tensor} -- Computed derivative of the wavefunction
        
        Returns:
            torch.Tensor - potential energy of shape [1]
        """

        if self.ke is not None:
            if dw_dx is None:
                return self.ke

        # Contract d2_w_dx over spatial dimensions:
        self.ke_by_parts = (H_BAR**2 / (2 * self.M)) * torch.sum(dw_dx**2, dim=-1)
        return self.ke_by_parts


    def kinetic_energy(self, *, w_of_x, d2w_dx2):
        """Return Kinetic energy
        
        If the potential energy is already computed, and no arguments are supplied,
        return the cached value

        If all arguments are supplied, calculate and return the KE.

        Otherwise, exception
        
        Arguments:
            w_of_x {torch.Tensor} -- Computed derivative of the wavefunction
        
        Returns:
            torch.Tensor - potential energy of shape [1]
        """
        self.ke = -(H_BAR**2 / (2 * self.M)) * w_of_x * torch.sum(d2w_dx2, axis=-1)
        return self.ke


    def energy(self, wavefunction, inputs):
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
        # And computes the expectation value of the energy at each input point
        
        # This is the value of the wave function:
        w_of_x = wavefunction(inputs)

            
        grad_accum = torch.ones(len(inputs))

        dw_dx = torch.autograd.grad(
            outputs=w_of_x, 
            inputs=inputs, 
            grad_outputs = grad_accum,
            retain_graph=True, 
            create_graph=True)[0]

        grad_accum = torch.ones(inputs.shape)

        # Compute the second derivative:
        d2w_dx2 = torch.autograd.grad(
            outputs=dw_dx, 
            inputs=inputs, 
            grad_outputs = grad_accum,
            retain_graph=True, 
            create_graph=True)[0]

        # Potential energy depends only on the wavefunction
        pe = self.potential_energy(wavefunction=wavefunction, inputs=inputs, w_of_x=w_of_x) 
        
        # KE by parts needs only one derivative
        ke_by_parts = self.kinetic_energy_by_parts(dw_dx=dw_dx)
        
        # True, directly, uses the second derivative
        ke_direct = self.kinetic_energy(w_of_x=w_of_x, d2w_dx2 = d2w_dx2)

        # Total energy computations:
        energy = pe + ke_direct
        energy_by_parts = pe + ke_by_parts

        return energy, energy_by_parts


