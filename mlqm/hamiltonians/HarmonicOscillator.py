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
        # x Squared needs to contract over spatial dimensions:
        x_squared = tf.reduce_sum(inputs**2, axis=(2))
        pe = (0.5 * M * omega**2 ) * x_squared
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

        # Contract d2_w_dx over spatial dimensions:
        ke_jf = (H_BAR**2 / (2 * M)) * tf.reduce_sum(dlogw_dx**2, axis=(1,2))
        return tf.reshape(ke_jf, [-1, 1])


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
        ke = -(H_BAR**2 / (2 * M)) * tf.reduce_sum(d2logw_dx2, axis=(1,2))
        return tf.reshape(ke, [-1, 1])  - KE_JF

    @tf.function
    def energy(self, wavefunction, inputs):
        """Compute the expectation valye of energy of the supplied wavefunction.

        Computes the integral of the wavefunction in this potential

        Arguments:
            wavefunction {Wavefunction model} -- Callable wavefunction object
            inputs {tf.Tensor} -- Tensor of shape [N, dimension], must have graph enabled
            delta {tf.Tensor} -- Integral Computation 'dx'

        Returns:
            tf.tensor - Energy of shape [1]
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


        # Potential energy depends only on the wavefunction
        pe = self.potential_energy(inputs=inputs, M = self.M, omega=self.omega)

        # KE by parts needs only one derivative
        ke_jf = self.kinetic_energy_jf(dlogw_dx=dlogw_dx, M=self.M)

        # True, directly, uses the second derivative
        ke_direct = self.kinetic_energy(KE_JF = ke_jf, d2logw_dx2 = d2logw_dx2, M=self.M)

        # Total energy computations:
        energy = tf.squeeze(pe + ke_direct)
        energy_jf = tf.squeeze(pe + ke_jf)

        return energy, energy_jf
