
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


        self.HBAR  = tf.constant(197.327,      dtype = DEFAULT_TENSOR_TYPE)
        self.alpha = tf.constant(1./137.03599, dtype = DEFAULT_TENSOR_TYPE)

        C01_dict = {
            'a' : -4.3852441,
            'b' : -5.72220536,
            'c' : -7.002509321,
            'd' : -8.22926713,
            'o' : -5.27518671,
        }
        C10_dict = {
            'a' : -8.00783936,
            'b' : -9.34392090,
            'c' : -10.7734100,
            'd' : -12.2993164,
            'o' : -7.04040080,
        }
        R0_dict = {
            'a' : 1.7,
            'b' : 1.9,
            'c' : 2.1,
            'd' : 2.3,
            'o' : 1.54592984,
        }
        R1_dict = {
            'a' : 1.5,
            'b' : 2.0,
            'c' : 2.5,
            'd' : 3.0,
            'o' : 1.83039397
        }

        # for the two body interactions:
        self.C01 = tf.constant(C01_dict[parameters.model], dtype=DEFAULT_TENSOR_TYPE)
        self.C10 = tf.constant(C10_dict[parameters.model], dtype=DEFAULT_TENSOR_TYPE)
        self.R0  = tf.constant(R0_dict[parameters.model],  dtype=DEFAULT_TENSOR_TYPE)
        self.R1  = tf.constant(R1_dict[parameters.model],  dtype=DEFAULT_TENSOR_TYPE)
        self.b   = tf.constant(4.27, dtype=DEFAULT_TENSOR_TYPE)
        self.c_pref = tf.constant(5.568327996831708,  dtype=DEFAULT_TENSOR_TYPE)

        # For the 3body interaction:
        self.R3  = tf.constant(1.0,  dtype=DEFAULT_TENSOR_TYPE)

        cE = 1.0786
        fpi = 92.4
        pi = 3.14159
        alpha_3body = (cE/ 1000. / (fpi)**4) * ((self.HBAR)**6 / (pi**3 * self.R3**6) )

        self.alpha_3body = tf.constant(tf.sqrt(alpha_3body), dtype = DEFAULT_TENSOR_TYPE)
        # self.ce3b = jnp.sqrt( self.ce3b / lamchi / fpi**4 * self.hc / jnp.pi**3 / self.R3**6 )

        # if self.vkr == 2.0:
        #     self.v0r  = tf.constant(-133.3431,     dtype = DEFAULT_TENSOR_TYPE)
        #     self.v0s  = tf.constant(-9.0212,       dtype = DEFAULT_TENSOR_TYPE)
        #     self.ar3b = tf.constant(8.2757658256,  dtype = DEFAULT_TENSOR_TYPE)
        #     logger.info(f"Using vkr = {self.vkr}")
        # elif self.vkr == 4.0:
        #     self.v0r  = tf.constant(-487.6128,     dtype = DEFAULT_TENSOR_TYPE)
        #     self.v0s  = tf.constant(-17.5515,      dtype = DEFAULT_TENSOR_TYPE)
        #     self.ar3b = tf.constant(26.0345712467, dtype = DEFAULT_TENSOR_TYPE)
        #     logger.info(f"Using vkr = {self.vkr}")
        # elif self.vkr == 6.0:
        #     self.v0r  = tf.constant(-1064.5010,    dtype = DEFAULT_TENSOR_TYPE)
        #     self.v0s  = tf.constant(-26.0830,      dtype = DEFAULT_TENSOR_TYPE)
        #     self.ar3b = tf.constant(51.5038930567, dtype = DEFAULT_TENSOR_TYPE)
        #     logger.info(f"Using vkr = {self.vkr}")

    @staticmethod
    def gen_possible_swaps(n_particles):
        '''
            # Create a list of all possible swaps.
            # With n particles, there are n_particles * (n_particles - 1) / 2 possible
            # swaps.  We can generate them all at once.
            # Note that swapping particles i and j is equal to swapping particles j and i.


            # Say, n = 4
            # 0 -> 1
            # 0 -> 2
            # 0 -> 3
            # 1 -> 2
            # 1 -> 3
            # 2 -> 3

        '''
        swap_i = []
        swap_j = []
        max_index = n_particles
        i = 0
        while i < n_particles:
            for j in range(i + 1, max_index):
                swap_i.append(i)
                swap_j.append(j)
            i += 1

        return tf.convert_to_tensor(swap_i), tf.convert_to_tensor(swap_j)



    def pionless_2b(self, r_ij):

        logger.info("pionless_2b")
        # These C functions are equation 2.7 from https://arxiv.org/pdf/2102.02327.pdf
        c0_r = (1./(self.c_pref * self.R0**3 )) * tf.exp(- tf.pow((r_ij / self.R0), 2))
        c1_r = (1./(self.c_pref * self.R1**3 )) * tf.exp(- tf.pow((r_ij / self.R1), 2))

        # Computing several functions here (A26 to A29 in https://arxiv.org/pdf/2102.02327.pdf):
        v_c         = self.HBAR * (3./16.) * (     self.C01 * c1_r +     self.C10 * c0_r)
        v_sigma     = self.HBAR * (1./16.) * (-3.* self.C01 * c1_r +     self.C10 * c0_r)
        v_tau       = self.HBAR * (1./16.) * (     self.C01 * c1_r - 3.* self.C10 * c0_r)
        v_sigma_tau = self.HBAR * -(1./16.) * (     self.C01 * c1_r +     self.C10 * c0_r)

        return v_c, v_sigma, v_tau, v_sigma_tau

    @tf.function()
    def pionless_3b(self, r_ij):
        logger.info("pionless_3b")
        x = r_ij / self.R3
        vr = tf.exp(-x**2)
        pot_3b = vr * self.alpha_3body
        return pot_3b


    @staticmethod
    def swap_by_index(pair, tensor):
        logger.debug("swap_by_index")
        shape = tensor.shape
        dim_0 = tf.range(shape[0])

        ones = tf.ones(shape=dim_0.shape, dtype=pair.dtype)

        copy_tensor = tf.identity(tensor)
        # dim_0 = tf.constant(range(tensor.shape[0]))
        index_i = tf.stack([dim_0, tf.gather(pair, 0)*ones], axis=1)
        index_j = tf.stack([dim_0, tf.gather(pair, 1)*ones], axis=1)

        # Replace this with gather if too slow:
        s_i = tensor[:,tf.gather(pair, 0)]
        s_j = tensor[:,tf.gather(pair, 1)]

        copy_tensor = tf.tensor_scatter_nd_update(copy_tensor, index_i, s_j)
        copy_tensor = tf.tensor_scatter_nd_update(copy_tensor, index_j, s_i)

        return copy_tensor

    @tf.function()
    def potential_em(self, r_ij):
        logger.info("potential_em")
        r_m = tf.maximum(r_ij, 0.0001)
        br  = self.b * r_m
        f_coul = 1 - (1 + (11./16.)*br + (3./16)*tf.pow(br,2) + (1./48)*tf.pow(br,3))*tf.exp(-br)
        return self.alpha * self.HBAR * f_coul / r_m

    # @tf.function(experimental_relax_shapes=True)
    def potential_pairwise(self, w_of_x, inputs, spin, isospin, pair):
        logger.info("Potential Pairwise")
        # Difference in ij coordinates:
        i = tf.gather(pair, 0); j = tf.gather(pair, 1);
        x_ij = tf.gather(inputs, i, axis=1) - tf.gather(inputs, j, axis=1)
        # Take the magnitude of that difference across dimensions
        r_ij = tf.sqrt(tf.reduce_sum(x_ij**2,axis=1))



        # Compute the Vrr and Vrs terms for this pair of particles:
        v_c, v_sigma, v_tau, v_sigma_tau = self.pionless_2b(r_ij=r_ij)

        ##
        ## TODO: ADD V_EM
        ##
        v_em = self.potential_em(r_ij=r_ij)

        # Now, we need to exchange the spin and isospin of this pair of particles

        swapped_spin    = self.swap_by_index(pair, tensor=spin)
        swapped_isospin = self.swap_by_index(pair, tensor=isospin)


        # Compute the wavefunction under all these exchanges:
        w_of_x_swap_spin    = self.wavefunction(inputs, swapped_spin, isospin)
        w_of_x_swap_isospin = self.wavefunction(inputs, spin,         swapped_isospin)
        w_of_x_swap_both    = self.wavefunction(inputs, swapped_spin, swapped_isospin)

        # Now compute several ratios:
        ratio_swapped_spin    = w_of_x_swap_spin    / w_of_x
        ratio_swapped_isospin = w_of_x_swap_isospin / w_of_x
        ratio_swapped_both    = w_of_x_swap_both    / w_of_x

        spin_factor     = tf.reshape(2*ratio_swapped_spin - 1,      (-1,))
        isospin_factor  = tf.reshape(2*ratio_swapped_isospin - 1,   (-1,))
        both_factor     = tf.reshape(4*ratio_swapped_both - 2*ratio_swapped_spin - 2*ratio_swapped_isospin + 1,  (-1,))


        # Em force only applies to protons, so apply that:
        proton = (1./4)*(1 + isospin[:,i])*(1 + isospin[:,j])

        # We accumulate the pairwise interaction of these two nucleons:
        v_ij = v_c
        v_ij += v_sigma     * spin_factor
        v_ij += v_tau       * isospin_factor
        v_ij += v_sigma_tau * both_factor
        v_ij += v_em        * proton

        t_ij = self.pionless_3b(r_ij=r_ij)
        logger.info("DONE - Potential Pairwise")

        return v_ij, t_ij


    @tf.function()
    def potential_energy(self, *, wavefunction, inputs, spin, isospin):
        """Return potential energy

        Calculate and return the PE.

        Arguments:
            inputs {tf.Tensor} -- Tensor of shape [N, dimension], must have graph enabled
        Returns:
            tf.Tensor - potential energy of shape [1]
        """

        logger.info("potential_energy")

        # Potential calculation

        # Prepare buffers for the output:
        # (Walker shape is (self.nwalkers, self.nparticles, self.n) )
        nwalkers   = inputs.shape[0]
        nparticles = inputs.shape[1]

        if nparticles == 2:
            alpha = 1.0
        elif nparticles > 2:
            alpha = -1.0

        # print(spin)
        # print(isospin)


        # The potential is ultimately just _one_ number per walker.
        # But, to properly compute the 3-body term, we use a per-particle
        # accumulater (gr3b) for each walker.
        gr3b = [tf.zeros(shape=[nwalkers], dtype=DEFAULT_TENSOR_TYPE) for p in range(nparticles)]
        V_ijk = tf.zeros(shape=[nwalkers], dtype=DEFAULT_TENSOR_TYPE) # three body potential terms
        v_ij  = tf.zeros(shape=[nwalkers], dtype=DEFAULT_TENSOR_TYPE) # 2 body potential terms:

        w_of_x = wavefunction(inputs, spin, isospin)

        # We need to flatten this loop.

        # Here we compute the pair-wise interaction terms
        for pair in self.swaps:

            this_v_ij, t_ij = self.potential_pairwise(w_of_x, inputs, spin, isospin, pair)
            v_ij += this_v_ij

            if (nparticles > 2 ):
                i = tf.gather(pair, 0)
                j = tf.gather(pair, 1)
                # Compute the 3 particle component which runs cyclically
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

    def compile_functions(self, inputs, spin, isospin, wavefunction):
        '''
        Replace important function calls with JIT'd versions all at once so
        they don't get traced many times.
        '''
        logger.info("Compiling hamiltonian functions!")

        # First, generate all the swaps:
        n_particles = spin.shape[-1]
        self.swaps = self.gen_possible_swaps(n_particles)

        example_pair = self.swaps[0]

        spec = [
            tf.TensorSpec(example_pair.shape, example_pair.dtype),
            tf.TensorSpec(spin.shape, spin.dtype),
        ]

        # This function only ever swaps spin/isospin, so replace it safely:
        self.swap_by_index = tf.function(self.swap_by_index, input_signature=spec, jit_compile=True)

        #  Call it once to compile it
        self.swap_by_index(example_pair, spin)

        # pionless_2b(self, r_ij)
        x_ij = tf.gather(inputs, 0, axis=1) - tf.gather(inputs, 1, axis=1)
        r_ij = tf.sqrt(tf.reduce_sum(x_ij**2,axis=1))

        spec = [
            tf.TensorSpec(r_ij.shape, r_ij.dtype),
        ]
        self.pionless_2b = tf.function(self.pionless_2b, input_signature = spec, jit_compile=True)
        _ = self.pionless_2b(r_ij)

        self.potential_em =  tf.function(self.potential_em, input_signature = spec, jit_compile=True)
        _ = self.potential_em(r_ij)

        self.pionless_3b =  tf.function(self.pionless_3b, input_signature = spec, jit_compile=True)
        _ = self.pionless_3b(r_ij)



        # Simulate the wavefunction so we can trace the pair-wise potential:
        self.wavefunction = wavefunction
        w_of_x = self.wavefunction(inputs, spin, isospin)

        spec = [
            tf.TensorSpec(w_of_x.shape, w_of_x.dtype),
            tf.TensorSpec(inputs.shape, inputs.dtype),
            tf.TensorSpec(spin.shape, spin.dtype),
            tf.TensorSpec(isospin.shape, isospin.dtype),
            tf.TensorSpec(example_pair.shape, example_pair.dtype),
        ]
        self.potential_pairwise = tf.function(self.potential_pairwise, input_signature=spec, jit_compile=True)

        # Now, compile the pairwise potential:
        _ = self.potential_pairwise(w_of_x, inputs, spin, isospin, example_pair)

        logger.info("Finished compiling hamiltonian functions.")


    @tf.function()
    # @tf.function(experimental_compile=False)
    def compute_energies(self, wavefunction, inputs, spin, isospin, w_of_x, dw_dx, d2w_dx2):
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
        logger.info("compute_energies")

        # Potential energy depends only on the wavefunction
        pe = self.potential_energy(wavefunction = wavefunction, inputs=inputs, spin=spin, isospin=isospin)

        # KE by parts needs only one derivative
        ke_jf = self.kinetic_energy_jf(
            w_of_x = w_of_x, dw_dx = dw_dx, M=self.parameters["mass"])

        # True, directly, uses the second derivative
        ke_direct = self.kinetic_energy(
            w_of_x = w_of_x, d2w_dx2 = d2w_dx2, M=self.parameters["mass"])

        return pe, ke_jf, ke_direct
