import tensorflow as tf
import numpy

import logging
logger = logging.getLogger()

from mlqm import H_BAR

class HarmonicOscillator(object):
    """Harmonic Oscillator Potential

    Implementation of the quantum harmonic oscillator hamiltonian
    """

    def __init__(self, M : float, omega : float):

        object.__init__(self)

        self.M = M
        self.omega = omega

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
        # print("  inputs.shape:", inputs.shape)
        x_squared = tf.reduce_sum(inputs**2, axis=(1, 2))
        # print("  x_squared.shape:", x_squared.shape)
        pe = (0.5 * M * omega**2 ) * x_squared
        # print("  pe.shape:", pe.shape)
        # print("Exit pe call")

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
        # print("Enter ke_jf call")
        # print("  dlogw_dx.shape: ", dlogw_dx.shape)
        ke_jf = (H_BAR**2 / (2 * M)) * tf.reduce_sum(dlogw_dx**2, axis=(1,2))
        # print("  ke_jf.shape: ", ke_jf.shape)
        # ke_jf = tf.reshape(ke_jf, [-1, 1])
        # print("  ke_jf.shape: ", ke_jf.shape)
        # print("Exit ke_jf call")
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
        with tf.GradientTape() as tape:
            # Use the inner tape to watch the computation of the second derivative:
            tape.watch(inputs)
            with tf.GradientTape() as second_tape:
                second_tape.watch(inputs)
                logw_of_x = wavefunction(inputs, training=True)
            # Get the derivative of logw_of_x with respect to inputs
            dlogw_dx = second_tape.gradient(logw_of_x, inputs)
        # Get the derivative of dlogw_dx with respect to inputs (aka second derivative)
        d2logw_dx2 = tape.gradient(dlogw_dx, inputs)

        # hessians = tf.hessians(logw_of_x, inputs)
        # d2_hessian = tf.reduce_sum(hessians[0], axis=(1,2,4,5))
        # d2logw_dx2 = tf.linalg.diag_part(d2_hessian)

        # print("logw_of_x.shape: ", logw_of_x.shape)
        # print("dlogw_dx.shape: ", dlogw_dx.shape)
        # print("d2logw_dx2.shape: ", d2logw_dx2.shape)

        # print("tf.reduce_mean(logw_of_x): ", tf.reduce_mean(logw_of_x))
        # print("tf.reduce_mean(dlogw_dx): ", tf.reduce_mean(dlogw_dx))
        # print("tf.reduce_mean(d2logw_dx2): ", tf.reduce_mean(d2logw_dx2))


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
