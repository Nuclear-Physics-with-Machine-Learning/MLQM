import torch
import numpy as np

#from mlqm import H_BAR
#from mlqm import M

import tensorflow as tf
import numpy

import logging
logger = logging.getLogger()

from mlqm import H_BAR

class NuclearPotential(object):
    """Nuclear Physics Potential
    """

    def __init__(self, M : float, omega : float):

        object.__init__(self)

        self.M = M
        self.omega = omega

    @tf.function
    def pionless_2b(self, *, r_ij):
        pot_2b=torch.zeros(self.nwalk,6)
        vkr = 4.0
        v0r = -487.6128
        v0s = -17.5515
        x = vkr * r_ij
        vr = torch.exp(-x**2/4.0)
        pot_2b[:,0] = vr*v0r
        pot_2b[:,2] = vr*v0s
        return pot_2b

    @tf.function
    def pionless_3b(self, *,  r_ij):
        pot_3b = torch.zeros(self.nwalk)
        vkr = 4.0
        ar3b = np.sqrt(677.79890)
        x = vkr * r_ij
        vr = torch.exp(-x**2/4.0)
        pot_3b = vr * ar3b
        return pot_3b

    @tf.function
    def potential_energy(self, *, inputs,):
        """Return potential energy

        Calculate and return the PE.

        Arguments:
            inputs {tf.Tensor} -- Tensor of shape [N, dimension], must have graph enabled
        Returns:
            tf.Tensor - potential energy of shape [1]
        """

        # Potential calculation

        # Prepare buffers for the output:
        # (Walker shape is (self.nwalkers, self.nparticles, self.n) )
        nwalkers   = inputs.shape[0]
        nparticles = inputs.shape[1]

        v_ij = tf.zeros(size=[nwalkers,6])
        gr3b = tf.zeros(size=[nwalkers,nparticles])
        V_ijk = tf.zeros(size=[nwalkers])
        for i in range (nparticles-1):
            for j in range (i+1,nparticles):
                # 
                x_ij = inputs[:,i,:]-inputs[:,j,:]
                r_ij = torch.sqrt(torch.sum(x_ij**2,dim=1))
                v_ij += potential.pionless_2b(r_ij)
                if (self.npart > 2 ):
                   t_ij = potential.pionless_3b(r_ij)
                   gr3b[:,i] += t_ij
                   gr3b[:,j] += t_ij
                   V_ijk -= t_ij**2
        V_ijk += 0.5 * torch.sum(gr3b**2, dim = 1)
        self.pe = v_ij[:,0] + self.alpha * v_ij[:,2] + V_ijk


        return pe

    @tf.function
    def kinetic_energy_jf(self, *, dlogw_dx, M):
        """Return Kinetic energy

        Calculate and return the KE directly

        Otherwise, exception

        Arguments:
            logw_of_x {tf.Tensor} -- Computed derivative of the wavefunction

        Returns:
            tf.Tensor - potential energy of shape [1]
        """
        # < x | KE | psi > / < x | psi > =  1 / 2m [ < x | p | psi > / < x | psi >  = 1/2 w * x**2

        # Contract d2_w_dx over spatial dimensions and particles:
        ke_jf = (H_BAR**2 / (2 * M)) * tf.reduce_sum(dlogw_dx**2, axis=(1,2))

        return ke_jf

    @tf.function
    def kinetic_energy(self, *, KE_JF, d2logw_dx2, M):
        """Return Kinetic energy

        If the potential energy is already computed, and no arguments are supplied,
        return the cached value

        If all arguments are supplied, calculate and return the KE.

        Otherwise, exception

        Arguments:
            logw_of_x {tf.Tensor} -- Computed derivative of the wavefunction

        Returns:
            tf.Tensor - potential energy of shape [1]
        """
        # print("Enter ke call")

        # print("  d2logw_dx2.shape:", d2logw_dx2.shape)
        # print("  KE_JF.shape:", KE_JF.shape)
        ke = -(H_BAR**2 / (2 * M)) * tf.reduce_sum(d2logw_dx2, axis=(1,2))
        # print("  ke.shape:", ke.shape)
        ke = ke  - KE_JF
        # print("  ke.shape:", ke.shape)
        # print("Exit ke call")

        return ke

    @tf.function
    def energy(self, wavefunction, inputs):
        """Compute the expectation valye of energy of the supplied wavefunction.

        Computes the integral of the wavefunction in this potential

        Arguments:
            wavefunction {Wavefunction model} -- Callable wavefunction object
            inputs {tf.Tensor} -- Tensor of shape [N, dimension], must have graph enabled
            delta {tf.Tensor} -- Integral Computation 'dx'

        Returns:
            tf.tensor - Energy of shape [n_walkers]
        """


        # This function takes the inputs
        # And computes the expectation value of the energy at each input point


        # Turning off all tape watching except for the inputs:
        # Using the outer-most tape to watch the computation of the first derivative:
        with tf.GradientTape(persistent=True) as tape:
            # Use the inner tape to watch the computation of the wavefunction:
            tape.watch(inputs)
            with tf.GradientTape() as second_tape:
                second_tape.watch(inputs)
                logw_of_x = wavefunction(inputs, training=True)
            # Get the derivative of logw_of_x with respect to inputs
            dlogw_dx = second_tape.gradient(logw_of_x, inputs)

        # Get the derivative of dlogw_dx with respect to inputs (aka second derivative)

        # We have to extract the diagonal of the jacobian, which comes out with shape
        # [nwalkers, nparticles, dimension, nwalkers, nparticles, dimension]

        # This is the full hessian computation:
        d2logw_dx2 = tape.jacobian(dlogw_dx, inputs)

        # And this contracts:
        d2logw_dx2 = tf.einsum("wpdwpd->wpd",d2logw_dx2)


        # Potential energy depends only on the wavefunction
        pe = self.potential_energy(inputs=inputs, M = self.M, omega=self.omega)

        # KE by parts needs only one derivative
        ke_jf = self.kinetic_energy_jf(dlogw_dx=dlogw_dx, M=self.M)

        # True, directly, uses the second derivative
        ke_direct = self.kinetic_energy(KE_JF = ke_jf, d2logw_dx2 = d2logw_dx2, M=self.M)

        # print("pe.shape: ", pe.shape)
        # print("ke_jf.shape: ", ke_jf.shape)
        # print("ke_direct.shape: ", ke_direct.shape)

        # print("tf.reduce_mean(pe): ", tf.reduce_mean(pe))
        # print("tf.reduce_mean(ke_jf): ", tf.reduce_mean(ke_jf))
        # print("tf.reduce_mean(ke_direct): ", tf.reduce_mean(ke_direct))


        # Total energy computations:
        energy = tf.squeeze(pe+ke_direct)
        energy_jf = tf.squeeze(pe+ke_jf)

        # print("energy.shape:", energy.shape)
        # print("energy_jf.shape:", energy_jf.shape)

        return energy, energy_jf, ke_jf, ke_direct, pe




class NuclearPotential(object):

    def __init__(self,nwalk):
        self.nwalk=nwalk
        object.__init__(self)

    def pionless_2b(self, rr):
        pot_2b=torch.zeros(self.nwalk,6)
        vkr = 4.0
        v0r = -487.6128
        v0s = -17.5515
        x = vkr*rr
        vr = torch.exp(-x**2/4.0)
        pot_2b[:,0] = vr*v0r
        pot_2b[:,2] = vr*v0s
        return pot_2b

    def pionless_3b(self, rr):
        pot_3b = torch.zeros(self.nwalk)
        vkr = 4.0
        ar3b = np.sqrt(677.79890)
        x = vkr*rr
        vr = torch.exp(-x**2/4.0)
        pot_3b = vr * ar3b
        return pot_3b



