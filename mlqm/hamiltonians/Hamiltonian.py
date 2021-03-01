import tensorflow as tf
import numpy

import logging
logger = logging.getLogger()

from mlqm import DEFAULT_TENSOR_TYPE

class Hamiltonian(object):
    """Harmonic Oscillator Potential

    Implementation of the quantum harmonic oscillator hamiltonian
    """

    def __init__(self, **kwargs):
        ''' Initialize the Hamiltonian

        The derived class will check parameters, but this converts all of them to floats
        and scores as TF Constants.

        '''
        object.__init__(self)
        self.parameters = {}
        # Cast them all to tf constants:
        for key in kwargs:
            self.parameters[key] = tf.constant(float(kwargs[key]),dtype=DEFAULT_TENSOR_TYPE)

        self.HBAR = tf.constant(1.0, dtype = DEFAULT_TENSOR_TYPE)

    @tf.function
    def potential_energy(self, *, inputs):
        """Return potential energy

        Calculate and return the PE.

        Arguments:
            inputs {tf.Tensor} -- Tensor of shape [N, dimension], must have graph enabled
        Returns:
            tf.Tensor - potential energy of shape [1]
        """

        raise NotImplementedError("Hamiltonian classes should implement this function")

    @tf.function
    def kinetic_energy_jf(self, *, dlogw_dx, M):
        """Return Kinetic energy

        Calculate and return the KE directly

        Otherwise, exception

        Arguments:
            dlogw_of_x/dx {tf.Tensor} -- Computed derivative of the wavefunction

        Returns:
            tf.Tensor - kinetic energy (JF) of shape [1]
        """
        # < x | KE | psi > / < x | psi > =  1 / 2m [ < x | p | psi > / < x | psi >  = 1/2 w * x**2

        # Contract d2_w_dx over spatial dimensions and particles:
        ke_jf = (self.HBAR**2 / (2 * M)) * tf.reduce_sum(dlogw_dx**2, axis=(1,2))

        return ke_jf

    @tf.function
    def kinetic_energy(self, *, KE_JF : tf.Tensor, d2logw_dx2 : tf.Tensor, M):
        """Return Kinetic energy


        If all arguments are supplied, calculate and return the KE.

        Arguments:
            d2logw_dx2 {tf.Tensor} -- Computed second derivative of the wavefunction
            KE_JF {tf.Tensor} -- JF computation of the kinetic energy

        Returns:
            tf.Tensor - potential energy of shape [1]
        """

        ke = -(self.HBAR**2 / (2 * M)) * tf.reduce_sum(d2logw_dx2, axis=(1,2))
        ke = ke  - KE_JF

        return ke

    @tf.function
    def compute_derivatives(self, wavefunction : tf.keras.models.Model, inputs : tf.Tensor):


        # Turning off all tape watching except for the inputs:
        # Using the outer-most tape to watch the computation of the first derivative:
        with tf.GradientTape() as tape:
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
        d2logw_dx2 = tape.batch_jacobian(dlogw_dx, inputs)

        # And this contracts:
        d2logw_dx2 = tf.einsum("wpdpd->wpd",d2logw_dx2)

        return logw_of_x, dlogw_dx, d2logw_dx2

    '''
    This whole section is correct but a bad implementation

    @tf.function
    def derivative_single_walker(self, wavefunction, walker):

        # Using the outer-most tape to watch the computation of the first derivative:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(walker)
            # Use the inner tape to watch the computation of the wavefunction:
            with tf.GradientTape(persistent=True) as second_tape:
                second_tape.watch(walker)
                logw_of_x = wavefunction(walker, training=True)
            # Get the derivative of logw_of_x with respect to inputs
            dlogw_dx = second_tape.gradient(logw_of_x, walker)

        d2logw_dx2 = tape.jacobian(dlogw_dx, walker)
        d2logw_dx2 = tf.einsum("wpdwpd->wpd",d2logw_dx2)

        return logw_of_x, dlogw_dx, d2logw_dx2

    @tf.function
    def compute_derivatives(self, wavefunction : tf.keras.models.Model, inputs : tf.Tensor):

        output_shape = inputs.shape
        inputs = tf.split(inputs, output_shape[0], axis=0)

        logw_of_x, dlogw_dx, d2logw_dx2 = zip(*(self.derivative_single_walker(wavefunction, i) for i in inputs))
        # Get the derivative of dlogw_dx with respect to inputs
        # (aka second derivative)

        # We have to extract the diagonal of the jacobian,
         # which comes out with shape
        # [nwalkers, nparticles, dimension, nwalkers, nparticles, dimension]

        # For a fixed number of particles and dimension,
        # this memory usage grows as nwalkers**2
        # BUT, the jacobian is block diagonal: if you
        # access the block [i,:,:,j,:,:] it is all zero unless i == j.

        # In this implementation, we're computing the jacobian PER walker and
        # only with respect to it's own inputs.  So, the jacobians are all
        # shaped like [1, npart, ndim, 1, npart, ndim] which grows linearly
        # with the number of walkers instead of quadratically.

        #restack everything:
        logw_of_x  = tf.reshape(tf.concat(logw_of_x, axis=0), (output_shape[0], 1))
        dlogw_dx   = tf.reshape(tf.concat(dlogw_dx, axis=0), output_shape)
        d2logw_dx2 = tf.reshape(tf.concat(d2logw_dx2, axis=0), output_shape)


        return logw_of_x, dlogw_dx, d2logw_dx2
    '''

    @tf.function
    def compute_energies(self, inputs, logw_of_x, dlogw_dx, d2logw_dx2):
        '''Compute PE, KE_JF, and KE_direct

        Placeholder for a user to implement their calculation of the energies.

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

        raise NotImplementedError("Please implement this function in the derived class.")

        # Needs to return like this
        return pe, ke_jf, ke_direct
        # return None

    @tf.function
    def energy(self, wavefunction : tf.keras.models.Model, inputs : tf.Tensor):
        """Compute the expectation value of energy of the supplied wavefunction.

        Computes the integral of the wavefunction in this potential

        Arguments:
            wavefunction {Wavefunction model} -- Callable wavefunction object
            inputs {tf.Tensor} -- Tensor of shape [nwalkers, nparticles, dimension]

        Returns:
            tf.tensor - energy of shape [n_walkers]
            tf.tensor - energy_jf of shape [n_walkers]
            tf.tensor - ke_jf of shape [n_walkers]
            tf.tensor - ke_direct of shape [n_walkers]
            tf.tensor - pe of shape [n_walkers]
        """


        # This function takes the inputs
        # And computes the expectation value of the energy at each input point

        logw_of_x, dlogw_dx, d2logw_dx2 = self.compute_derivatives(wavefunction, inputs)

        pe, ke_jf, ke_direct = self.compute_energies(inputs, logw_of_x, dlogw_dx, d2logw_dx2)

        # Total energy computations:
        energy = tf.squeeze(pe+ke_direct)
        energy_jf = tf.squeeze(pe+ke_jf)

        return energy, energy_jf, ke_jf, ke_direct, pe, logw_of_x
