import torch
import numpy

from mlqm import H_BAR

class HarmonicOscillator_mc(object):
    """Harmonic Oscillator Potential
    
    Implementation of the quantum harmonic oscillator hamiltonian
    """

    def __init__(self,  n : int, M : float, omega : float, nwalk : int, ndim : int):

        object.__init__(self)

        self.n = n
        if self.n < 1 or self.n > 3: 
            raise Exception("Dimension must be 1, 2, or 3 for HarmonicOscillator")

        self.M = M

        self.omega = omega

        self.nwalk = nwalk

        self.ndim = ndim

        # Several objects get stored for referencing, if needed, after energy computation:
        self.pe = None
        self.ke = None

    def potential_energy_old(self, *, wavefunction=None, inputs=None, w_of_x=None):
        "Returns potential energy"

        self.pe = ( 0.5 * self.M * self.omega**2 ) * torch.sum( inputs**2 )

        return self.pe

    def kinetic_energy_old(self, *, w_prime_dx=None):
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


        # This function takes the coordinates as inputs and computes the expectation value of the energy.
        
        # This is the value of the wave function:
#        w_of_x = wavefunction(inputs)

        # This is the first derivative of the logarithm of the wave function: dlog(psi)/dx = 1/psi dpsi/dx
#        log_psi_2 = torch.sum(torch.log(w_of_x**2))
#        wavefunction.zero_grad()
#        log_psi_2.backward(retain_graph = True, create_graph = True)
#        w_prime_dx = inputs.grad / 2

#        pe_old = self.potential_energy_old(wavefunction=wavefunction, inputs=inputs, w_of_x=w_of_x) 
#        ke_old = self.kinetic_energy_old(w_prime_dx=w_prime_dx)

#        i = 0
#        for p in wavefunction.parameters():
#            i = i + 1
#            if (i==1): aa = p.data
#            if (i==2): bb = p.data


#        print()
#        print("nn wave function", torch.sum(w_of_x))
#        print("exact wave function", torch.sum(torch.exp(-0.5*(aa*torch.sum(inputs,1)+bb*torch.ones(self.nwalk))**2)))

#        inputs.grad.data.zero_()
        wavefunction.zero_grad()
        psi = wavefunction(inputs)
        log_psi = torch.log(psi)
#        print()
#        print("inputs", inputs)
#        print("log_psi", log_psi)
        vt = torch.ones(self.nwalk)

        
        d_log_psi = torch.autograd.grad(log_psi, inputs, vt, create_graph=True, retain_graph=True)[0]
        d_psi = d_log_psi
#        print("dlog_psi", d_log_psi)
        
        ke = 0
        for i in range (self.ndim):
            d_log_psi_i = d_log_psi[:,i]
#            print("d_log_psi_i",i, d_log_psi_i)
            d2_log_psi = torch.autograd.grad(d_log_psi_i, inputs, vt, retain_graph=True)[0]
            d2_log_psi_ii = d2_log_psi[:,i]
            d2_psi_i = d2_log_psi_ii + d_log_psi_i**2
#            print("d2_log_psi_ii",i, d2_log_psi_ii)
            ke -= d2_psi_i / 2.


#        d_log_psi = torch.autograd.grad(log_psi, inputs, vt, create_graph=True, retain_graph=True)
#        d_psi = d_log_psi
#        print("dlog_psi", d_log_psi)



#        d2_log_psi = torch.autograd.grad(d_log_psi, inputs, vtt, create_graph=False, retain_graph=False)[0]

#       print("d2log_psi", d2_log_psi)

#        exit()


#        d2_psi = d_log_psi**2 + d2_log_psi

#        ke = - torch.sum(d2_psi, 1) / 2.
        ke_jf = torch.sum(d_psi**2, 1) / 2.
        pe = ( 0.5 * self.M * self.omega**2 ) * torch.sum(inputs**2, 1)

        energy_jf = ke_jf + pe

        energy = ke + pe

        if (1 == 0) : 
            i = 0
            for p in wavefunction.parameters():
                i = i + 1
                if (i==1): aa = p
                if (i==2): bb = p


#            print()
#            print("input", inputs)

#            print()
#            print("aa=", aa)
#            print("bb=", bb)

            x0 = inputs
            x1 = x0 + bb
            x2 = torch.sum(aa * x1**2, dim=1)
            psi_exact = torch.exp( - x2 / 2)

            print()
            print("wave function", psi)
            print("exact wave function", psi_exact ) 

            print()
            print("derivative", d_psi )
            print("exact derivative", -aa * x1)

            print()
            print("squared derivative", d_psi**2)
            print("exact squared derivative", (-aa * x1)**2 )

            print()
            print("second derivative", d2_psi)
            print("exact second derivative", aa**2 * x1**2 - aa )


            print()
            print("kinetic energy", torch.sum(ke))
            print("exact kinetic energy",-torch.sum(aa**2 * x1**2 - aa)/2)
            print("jf kinetic energy", torch.sum(ke_jf))
            print("exact jf kinetic energy",torch.sum((-aa * x1)**2)/2)

            print()
            print("potential energy", pe)

            wavefunction.zero_grad()
            energy.backward(vt, retain_graph = True)

            dE_da = - aa * x1**2 + 0.5 

#            dE_da = aa * x1**2 
            
            dE_db = - aa**2 * x1  

# 2-d case
#            dE_da = self.ndim * dE_da
#            dE_db = self.ndim * dE_db
        
            print()
            i = 0
            for p in wavefunction.parameters():
                i = i + 1
                if (i==1): print("p grad inside=",i,torch.sum(p.grad), torch.sum(dE_da))
                if (i==2): print("p grad inside=",i,torch.sum(p.grad), torch.sum(dE_db))
#            exit()


        return energy, energy_jf




