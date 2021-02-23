import tensorflow as tf
import numpy

import logging
logger = logging.getLogger()

from mlqm.hamiltonians import Hamiltonian

class HarmonicOscillator(Hamiltonian):
    """Harmonic Oscillator Potential

    Implementation of the quantum harmonic oscillator hamiltonian
    """

    def __init__(self, **kwargs):

        Hamiltonian.__init__(self, **kwargs)

        # Check the parameters have everything needed:
        for parameter in ["mass", "omega"]:
            if parameter not in self.parameters:
                raise KeyError(f"Parameter {parameter} not suppliled as keyword arg to HarmonicOscillator")


    @tf.function
    def potential_energy(self, *, inputs, M, omega):
        """Return potential energy

        Calculate and return the PE.

        Arguments:
            inputs {tf.Tensor} -- Tensor of shape [N, dimension], must have graph enabled
        Returns:
            tf.Tensor - potential energy of shape [1]
        """

        # Potential calculation
        # < x | H | psi > / < x | psi > = < x | 1/2 w * x**2  | psi > / < x | psi >  = 1/2 w * x**2
        # print("Enter pe call")

        # x Squared needs to contract over spatial dimensions:
        x_squared = tf.reduce_sum(inputs**2, axis=(1, 2))
        pe = (0.5 * M * omega**2 ) * x_squared

        return pe

    @tf.function
    def compute_energies(self, inputs, logw_of_x, dlogw_dx, d2logw_dx2):
        '''Compute PE, KE_JF, and KE_direct

        Harmonic Oscillator Energy Calculations

        Arguments:
            inputs {[type]} -- walker coordinates (shape is [nwalkers, nparticles, dimension])
            logw_of_x {[type]} -- computed wave function at each walker
            dlogw_dx {[type]} -- first derivative of wavefunction at each walker
            d2logw_dx2 {[type]} -- second derivative of wavefunction at each walker

        Raises:
            NotImplementedError -- [description]

        Returns:
            pe -- potential energy
            ke_jf -- JF Kinetic energy
            ke_direct -- 2nd deriv computation of potential energy
        '''

        # Potential energy depends only on the wavefunction
        pe = self.potential_energy(inputs=inputs, M = self.parameters["mass"], omega=self.parameters["omega"])

        # KE by parts needs only one derivative
        ke_jf = self.kinetic_energy_jf(dlogw_dx=dlogw_dx, M=self.parameters["mass"])

        # True, directly, uses the second derivative
        ke_direct = self.kinetic_energy(KE_JF = ke_jf, d2logw_dx2 = d2logw_dx2, M=self.parameters["mass"])

        return pe, ke_jf, ke_direct
