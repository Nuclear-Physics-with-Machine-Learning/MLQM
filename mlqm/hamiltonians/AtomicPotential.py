import tensorflow as tf
import numpy

from mlqm import H_BAR, ELECTRON_CHARGE

class AtomicPotential(object):
    """Atomic Hamiltonian

    Implementation of the atomic hamiltonian

    """

    def __init__(self, mu : float, Z : int):
        """Return potential energy

        If the potential energy is already computed, and no arguments are supplied,
        return the cached value

        If all arguments are supplied, calculate and return the PE.

        Otherwise, exception

        Arguments:
            mu {float} -- (Reduced) Mass of nucleus
            Z {int} -- Nuclear charge, aka number of electrons

        """

        object.__init__(self)

        self.mu = mu
        self.Z  = Z


    def potential_energy(self, *, inputs, Z):
        """Return potential energy

        If the potential energy is already computed, and no arguments are supplied,
        return the cached value

        If all arguments are supplied, calculate and return the PE.

        Otherwise, exception

        Arguments:
            inputs {tf.tensor} -- Tensor of shape [N, nparticles, dimension]
            Z {tf.tensor} -- Atomic number

        Returns:
            torch.Tensor - potential energy of shape [1]
        """

        # Potential energy is, for n particles, two pieces:
        # Compute r_i, where r_i = sqrt(sum(x_i^2, y_i^2, z_i^2)) (in 3D)
        # PE_1 = -(Z e^2)/(4 pi eps_0) * sum_i (1/r_i)
        # Second, compute r_ij, for all i != j, and then
        # PE_2 = -(e^2) / (4 pi eps_0) * sum_{i!=j} (1 / r_ij)
        # where r_ij = sqrt( [xi - xj]^2 + [yi - yj] ^2 + [zi - zj]^2)

        # Compute r
        # Square the coordinates and
        r = tf.math.sqrt(tf.reduce_sum(inputs**2, axis=2))
        print(r.shape)
        # This is the sum of 1/r for all particles with the nucleus:
        pe_1 = - (Z * ELECTRON_CHARGE**2 ) * tf.reduce_sum( 1. / (r + 1e-8), axis=1 )
        print(pe_1.shape)
        pe_2 = 0.
        return tf.reshape((pe_1 + pe_2), [-1, 1])


    # @tf.function
    # def potential_energy(self, *, inputs, M, omega):
    #     """Return potential energy
    #
    #     Calculate and return the PE.
    #
    #     Arguments:
    #         inputs {tf.Tensor} -- Tensor of shape [N, dimension], must have graph enabled
    #     Returns:
    #         tf.Tensor - potential energy of shape [1]
    #     """
    #
    #     # Potential calculation
    #     # < x | H | psi > / < x | psi > = < x | 1/2 w * x**2  | psi > / < x | psi >  = 1/2 w * x**2
    #     # x Squared needs to contract over spatial dimensions:
    #     x_squared = tf.reduce_sum(inputs**2, axis=(2))
    #     pe = (0.5 * M * omega**2 ) * x_squared
    #     return pe

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

        # print("d2logw_dx2.shape: ",( dlogw_dx**2).shape)
        # print("tf.reduce_sum(d2logw_dx2, axis=(2)).shape: ", tf.reduce_sum(dlogw_dx**2, axis=(2)).shape)
        # print("tf.reduce_sum(d2logw_dx2, axis=(1,2)).shape: ", tf.reduce_sum(dlogw_dx**2, axis=(1,2)).shape)

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
        # print("d2logw_dx2.shape: ", d2logw_dx2.shape)
        # print("tf.reduce_sum(d2logw_dx2, axis=(2)).shape: ", tf.reduce_sum(d2logw_dx2, axis=(2)).shape)
        # print("tf.reduce_sum(d2logw_dx2, axis=(1,2)).shape: ", tf.reduce_sum(d2logw_dx2, axis=(1,2)).shape)
        ke = -(H_BAR**2 / (2 * M)) * tf.reduce_sum(d2logw_dx2, axis=(1,2))
        return tf.reshape(ke, [-1, 1])  - KE_JF

    # def energy_internal(self, wavefunction, inputs)

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


        # Potential energy depends only on the wavefunction
        pe = self.potential_energy(inputs=inputs, Z=self.Z)

        # KE by parts needs only one derivative
        ke_jf = self.kinetic_energy_jf(dlogw_dx=dlogw_dx, M=self.mu)

        # True, directly, uses the second derivative
        ke_direct = self.kinetic_energy(KE_JF = ke_jf, d2logw_dx2 = d2logw_dx2, M=self.mu)

        # print(ke_jf)
        # print(ke_direct)

        # Total energy computations:
        energy = tf.squeeze(pe + ke_direct)
        energy_jf = tf.squeeze(pe + ke_jf)

        return energy, energy_jf
