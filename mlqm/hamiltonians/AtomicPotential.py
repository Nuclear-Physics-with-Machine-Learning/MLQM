import tensorflow as tf
import numpy

from mlqm import ELECTRON_CHARGE
from mlqm.hamiltonians import Hamiltonian

class AtomicPotential(Hamiltonian):
    """Atomic Hamiltonian

    Implementation of the atomic hamiltonian

    """

    def __init__(self, **kwargs):
        '''

        Arguments:
            mu {float} -- (Reduced) Mass of nucleus
            Z {int} -- Nuclear charge, aka number of electrons

        '''
        Hamiltonian.__init__(self, **kwargs)

        # Check the parameters have everything needed:
        for parameter in ["mass", "z"]:
            if parameter not in self.parameters:
                raise KeyError(f"Parameter {parameter} not suppliled as keyword arg to Atomic Potential")


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
        # Square the coordinates and sum for each walker
        r = tf.math.sqrt(tf.reduce_sum(inputs**2, axis=2))
        # This is the sum of 1/r for all particles with the nucleus:
        pe_1 = - (Z * ELECTRON_CHARGE**2 ) * tf.reduce_sum( 1. / (r + 1e-8), axis=1 )

        # This is the sum of 1/r for all particles with other particles.
        # n_particles = inputs.shape[1]
        # for i_particle in range(n_particles):
        #     centroid = inputs[:,i_particle,:]
        #
        #     r = tf.math.sqrt(tf.reduce_sum((inputs -centroid)**2, axis=2))
        #     pe_2 = -0.5* (ELECTRON_CHARGE**2 ) * tf.reduce_sum( 1. / (r + 1e-8), axis=1 )
        #     # Because this force is symmetric, I'm multiplying by 0.5 to prevent overflow
        pe_2 = 0.
        return pe_1 + pe_2


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
        pe = self.potential_energy(inputs=inputs, Z=self.parameters['z'])

        # KE by parts needs only one derivative
        ke_jf = self.kinetic_energy_jf(dlogw_dx=dlogw_dx, M=self.parameters["mass"])

        # True, directly, uses the second derivative
        ke_direct = self.kinetic_energy(KE_JF = ke_jf, d2logw_dx2 = d2logw_dx2, M=self.parameters["mass"])


        return pe, ke_jf, ke_direct
