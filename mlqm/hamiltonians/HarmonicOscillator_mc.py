import torch
import numpy

from mlqm import H_BAR


class HarmonicOscillator_mc(object):
    """Harmonic Oscillator Potential
    
    Implementation of the quantum harmonic oscillator hamiltonian
    """

    def __init__(self, M : float, omega : float, nwalk : int, ndim : int, npart : int):

        object.__init__(self)

        self.ndim = ndim
#        if self.ndim < 1 or self.ndim > 3: 
#            raise Exception("Dimension must be 1, 2, or 3 for HarmonicOscillator")

        self.M = M

        self.omega = omega

        self.nwalk = nwalk

        self.npart = npart

        # Several objects get stored for referencing, if needed, after energy computation:
        self.pe = None
        self.ke = None
        self.ke_jf= None

    def potential_energy(self, potential, inputs):
        "Returns potential energy"
        v_ij=torch.zeros(size=[self.nwalk,6])
        for i in range (self.npart):
            for j in range (i+1,self.npart):
                x_ij=inputs[:,i,:]-inputs[:,j,:]
                r_ij=torch.sqrt(torch.sum(x_ij**2,dim=1))
                v_ij+=potential.pionless(r_ij)
         
        #self.pe = ( 0.5 * self.M * self.omega**2 ) * torch.sum(inputs**2, dim=(1,2))
        self.pe=v_ij[:,0]
        return self.pe

    def kinetic_energy(self, wavefunction, inputs):
        "Returns kinetic energy"
      
        wavefunction.zero_grad()
        log_psi = wavefunction(inputs)
        vt = torch.ones(size=[self.nwalk])
        
        d_log_psi = torch.autograd.grad(log_psi, inputs, vt, create_graph=True, retain_graph=True)[0]
        d_psi = d_log_psi
        self.ke_jf = torch.sum(d_psi**2, (1,2)) / 2.        
        self.ke = 0
        for i in range (self.ndim):
            for j in range (self.npart):
                d_log_psi_ij = d_log_psi[:,j,i]
                d2_log_psi = torch.autograd.grad(d_log_psi_ij, inputs, vt, retain_graph=True)[0]
                d2_log_psi_ii_jj = d2_log_psi[:,j,i]
                d2_psi_ij = d2_log_psi_ii_jj + d_log_psi_ij**2
                self.ke -= d2_psi_ij / 2.

        return self.ke, self.ke_jf

    def energy(self, wavefunction, potential, inputs):
        """Compute the expectation valye of energy of the supplied wavefunction.
        
        Computes the integral of the wavefunction in this potential
        
        Arguments:
            wavefunction {Wavefunction model} -- Callable wavefunction object
            inputs {torch.Tensor} -- Tensor of shape [N, dimension], must have graph enabled
        
        Returns:
            torch.tensor - Energy of shape [1]
        """
      
        ke, ke_jf = self.kinetic_energy(wavefunction,inputs)
        pe= self.potential_energy(potential, inputs)
        
        energy_jf = ke_jf + pe

        energy = ke + pe

 
        return energy, energy_jf




