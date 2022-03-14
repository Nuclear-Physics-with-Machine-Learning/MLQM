
import tensorflow as tf
import numpy

import logging
logger = logging.getLogger()

from mlqm import DEFAULT_TENSOR_TYPE
from mlqm.hamiltonians import Hamiltonian


class NuclearPotential(Hamiltonian):
    """Nuclear Physics Potential
    """

    def __init__(self, parameters):
        '''
        Arguments:
            mass {float} -- Nuclear mass, aka number of electrons

        '''
        Hamiltonian.__init__(self, parameters)


        # # Check the parameters have everything needed:
        # for parameter in ["mass"]:
        #     if parameter not in self.parameters:
        #         raise KeyError(f"Parameter {parameter} not suppliled as keyword arg to NuclearPotential")

        if 'vkr' in self.parameters:
            if self.parameters['vkr'] not in [2, 4, 6]:
                raise KeyError(f"Parameter vkr set to {self.parameters['vkr']} but must be 2, 4 or 6")
            self.vkr = tf.constant(self.parameters['vkr'], dtype = DEFAULT_TENSOR_TYPE)
        else:
            logger.info("Setting vkr to 4 in the nuclear potential by default.")
            self.vkr = tf.constant(4, dtype = DEFAULT_TENSOR_TYPE)

        if self.vkr == 2.0:
            self.v0r  = tf.constant(-133.3431,     dtype = DEFAULT_TENSOR_TYPE)
            self.v0s  = tf.constant(-9.0212,       dtype = DEFAULT_TENSOR_TYPE)
            self.ar3b = tf.constant(8.2757658256,  dtype = DEFAULT_TENSOR_TYPE)
            logger.info(f"Using vkr = {self.vkr}")
        elif self.vkr == 4.0:
            self.v0r  = tf.constant(-487.6128,     dtype = DEFAULT_TENSOR_TYPE)
            self.v0s  = tf.constant(-17.5515,      dtype = DEFAULT_TENSOR_TYPE)
            self.ar3b = tf.constant(26.0345712467, dtype = DEFAULT_TENSOR_TYPE)
            logger.info(f"Using vkr = {self.vkr}")
        elif self.vkr == 6.0:
            self.v0r  = tf.constant(-1064.5010,    dtype = DEFAULT_TENSOR_TYPE)
            self.v0s  = tf.constant(-26.0830,      dtype = DEFAULT_TENSOR_TYPE)
            self.ar3b = tf.constant(51.5038930567, dtype = DEFAULT_TENSOR_TYPE)
            logger.info(f"Using vkr = {self.vkr}")


        self.HBAR = tf.constant(197.327, dtype = DEFAULT_TENSOR_TYPE)

    # @tf.function(experimental_compile=False)
    def pionless_2b(self, *, r_ij):
        # print("r_ij: ", r_ij)
        x = self.vkr * r_ij
        # print("x: ", x)
        vr = tf.exp(-x**2/4.0)
        # print("vr: ", vr)
        # print("self.v0r: ", self.v0r)
        # print("self.v0s: ", self.v0s)
        return self.v0r*vr, self.v0s*vr

    @tf.function(experimental_compile=False)
    def pionless_3b(self, *,  r_ij, nwalkers):
        # pot_3b = tf.zeros(shape=(nwalkers), dtype=DEFAULT_TENSOR_TYPE)
        x = self.vkr * r_ij
        vr = tf.exp(-x**2/4.0)
        pot_3b = vr * self.ar3b
        return pot_3b

    @tf.function(experimental_compile=False)
    def potential_energy(self, *, inputs, spin, isospin):
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

        if nparticles == 2:
            alpha = 1.0
        elif nparticles > 2:
            alpha = -1.0

        # print("Alpha: ", alpha)

        # gr3b = tf.Variable(tf.zeros(shape=[nwalkers,nparticles], dtype=DEFAULT_TENSOR_TYPE))
        gr3b = [tf.zeros(shape=[nwalkers], dtype=DEFAULT_TENSOR_TYPE) for p in range(nparticles)]
        V_ijk = tf.zeros(shape=[nwalkers], dtype=DEFAULT_TENSOR_TYPE) # three body potential terms
        v_ij  = tf.zeros(shape=[nwalkers], dtype=DEFAULT_TENSOR_TYPE) # 2 body potential terms:
        for i in range (nparticles-1):
            for j in range (i+1,nparticles):
                #
                x_ij = inputs[:,i,:]-inputs[:,j,:]
                r_ij = tf.sqrt(tf.reduce_sum(x_ij**2,axis=1))
                vrr, vrs = self.pionless_2b(r_ij=r_ij)
                # v_ij += self.pionless_2b(r_ij=r_ij, nwalkers=nwalkers)
                v_ij += vrr + alpha * vrs
                if (nparticles > 2 ):
                   t_ij = self.pionless_3b(r_ij=r_ij, nwalkers=nwalkers)
                   gr3b[i] += t_ij
                   gr3b[j] += t_ij
                   # gr3b[i] = gr3b[:,i].assign(gr3b[:,i] + t_ij)
                   # gr3b = gr3b[:,j].assign(gr3b[:,j] + t_ij)
                   V_ijk -= t_ij**2
        # stack up gr3b:
        gr3b = tf.stack(gr3b, axis=1)
        V_ijk += 0.5 * tf.reduce_sum(gr3b**2, axis = 1)
        pe = v_ij + V_ijk

        # print(pe)
        return pe

    # @tf.function()
    # @tf.function(experimental_compile=False)
    def compute_energies(self, inputs, spin, isospin, w_of_x, dw_dx, d2w_dx2):
        '''Compute PE, KE_JF, and KE_direct

        Harmonic Oscillator Energy Calculations

        Arguments:
            inputs {[type]} -- walker coordinates (shape is [nwalkers, nparticles, dimension])
            w_of_x {[type]} -- computed wave function at each walker
            dw_dx {[type]} -- first derivative of wavefunction at each walker
            d2w_dx2 {[type]} -- second derivative of wavefunction at each walker

        Raises:
            NotImplementedError -- [description]

        Returns:
            pe -- potential energy
            ke_jf -- JF Kinetic energy
            ke_direct -- 2nd deriv computation of potential energy
        '''

        # Potential energy depends only on the wavefunction
        pe = self.potential_energy(inputs=inputs, spin=spin, isospin=isospin)

        # KE by parts needs only one derivative
        ke_jf = self.kinetic_energy_jf(
            w_of_x = w_of_x, dw_dx = dw_dx, M=self.parameters["mass"])

        # True, directly, uses the second derivative
        ke_direct = self.kinetic_energy(
            w_of_x = w_of_x, d2w_dx2 = d2w_dx2, M=self.parameters["mass"])

        return pe, ke_jf, ke_direct
