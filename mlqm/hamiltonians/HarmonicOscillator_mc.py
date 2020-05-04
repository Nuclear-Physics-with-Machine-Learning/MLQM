import torch
import numpy

from mlqm import H_BAR

class HarmonicOscillator_mc(object):
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

    def potential_energy(self, *, wavefunction=None, inputs=None, w_of_x=None):
        "Returns potential energy"

        self.pe = ( 0.5 * self.M * self.omega**2 ) * torch.sum( inputs**2 )

        return self.pe

    def kinetic_energy(self, *, w_prime_dx=None):
        "Return Kinetic energy"

        self.ke = (H_BAR**2 / (2 * self.M)) * torch.sum(torch.pow(w_prime_dx,2))
        
        return self.ke


    def energy(self, wavefunction, inputs):
        """Compute the expectation valye of energy of the supplied wavefunction.
        
        Computes the integral of the wavefunction in this potential
        
        Arguments:
            wavefunction {Wavefunction model} -- Callable wavefunction object
            inputs {torch.Tensor} -- Tensor of shape [N, dimension], must have graph enabled
        
        Returns:
            torch.tensor - Energy of shape [1]
        """


        # This function takes the inputs
        # And computes the expectation value of the energy.
        
        # This is the value of the wave function:
        w_of_x = wavefunction(inputs)

        # Sum the wavefunction and call backwards to get the derivative of it with respect to x:
        log_psi = torch.log(w_of_x**2)
        grad_catalyst = torch.sum(log_psi)
        grad_catalyst.backward(retain_graph = True)

        # This is the first derivative of the logarithm of the wave function: dlog(psi)/dx = 1/psi dpsi/dx
        
        w_prime_dx = inputs.grad / 2.

        pe = self.potential_energy(wavefunction=wavefunction, inputs=inputs, w_of_x=w_of_x) 
        ke = self.kinetic_energy(w_prime_dx=w_prime_dx)

#        print("inputs=", inputs)

#        print("pe=", pe)
#        print("ke=", ke)
#        print("tot=", pe + ke)

        energy = pe + ke
        
        return energy



