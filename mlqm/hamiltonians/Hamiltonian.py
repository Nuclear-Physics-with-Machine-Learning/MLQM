import jax.numpy as numpy
from functools import partial
from jax import jit, vmap
from jax import grad, jacfwd

import logging
logger = logging.getLogger()

from mlqm import DEFAULT_TENSOR_TYPE

HBAR = 1.0

# @jit
# def kinetic_energy_jf(w_of_x, dw_dx, M):
#     """Return Kinetic energy
#
#     Calculate and return the KE directly
#
#     Otherwise, exception
#
#     Arguments:
#         w_of_x {DeviceArray} -- Computed derivative of the wavefunction
#         dw/dx {DeviceArray} -- Computed derivative of the wavefunction
#         M {float} -- Mass
#
#     Returns:
#         Device Array - kinetic energy (JF) of shape [1]
#     """
#     # < x | KE | psi > / < x | psi > =  1 / 2m [ < x | p | psi > / < x | psi >  = 1/2 w * x**2
#
#
#
#     internal_arg = dw_dx / tf.reshape(w_of_x, (-1,1,1))
#
#     # Contract d2_w_dx over spatial dimensions and particles:
#     ke_jf = (self.HBAR**2 / (2 * M)) * tf.reduce_sum(internal_arg**2, axis=(1,2))
#
#
#     return ke_jf
#
# @tf.function
# def kinetic_energy(self, *, w_of_x : tf.Tensor, d2w_dx2 : tf.Tensor, M):
#     """Return Kinetic energy
#
#
#     If all arguments are supplied, calculate and return the KE.
#
#     Arguments:
#         d2w_dx2 {tf.Tensor} -- Computed second derivative of the wavefunction
#         KE_JF {tf.Tensor} -- JF computation of the kinetic energy
#
#     Returns:
#         tf.Tensor - potential energy of shape [1]
#     """
#
#
#     inverse_w = tf.reshape(1/(w_of_x), (-1,1) )
#     # Only reduce over the spatial dimension here:
#     summed_d2 = tf.reduce_sum(d2w_dx2, axis=(2))
#
#     ke = -(self.HBAR**2 / (2 * M)) * tf.reduce_sum(inverse_w * summed_d2, axis=1)
#
#     return ke

# @partial(jit, static_argnums=(0,))
def compute_derivatives_single(wavefunction, params, x, spin, isospin):
    '''
    Compute the value, as well as first and second derivatives of function
    '''

    n_particles = x.shape[0]
    n_dim = x.shape[1]

    # l = lambda p, _x : wavefunction.apply(p, _x, spin, isospin)

    w_of_x = wavefunction.apply(params, x, spin, isospin)
    # gradient function:
    g_f = grad(wavefunction.apply, argnums=1)

    # Calculate gradient:
    dw_dx = g_f(params, x, spin, isospin)
    # jacobian function:
    J_fn = jacfwd(g_f, argnums=1)

    # Calculate Jacobian:
    J = J_fn(params, x, spin, isospin)

    # This selects out the elements of the jacobian that we actually want:
    selection = numpy.eye(n_particles*n_dim).reshape(
        (n_particles, n_dim, n_particles, n_dim)
    )
    d2w_dx2 = (J*selection).sum((2,3))

    return w_of_x, dw_dx, d2w_dx2


compute_derivatives = vmap(compute_derivatives_single, in_axes = [None, None, 0,0,0])
# partial(jit(, static_argnums=0))
# @tf.function
# def compute_energies(self, wavefunction, inputs, spin, isospin, w_of_x, dw_dx, d2w_dx2):
#     '''Compute PE, KE_JF, and KE_direct
#
#     Placeholder for a user to implement their calculation of the energies.
#
#     Arguments:
#         inputs {[type]} -- walker coordinates (shape is [nwalkers, nparticles, dimension])
#         w_of_x {[type]} -- computed wave function at each walker
#         dw_dx {[type]} -- first derivative of wavefunction at each walker
#         d2w_dx2 {[type]} -- second derivative of wavefunction at each walker
#
#     Raises:
#         NotImplementedError -- [description]
#
#     Returns:
#         pe -- potential energy
#         ke_jf -- JF Kinetic energy
#         ke_direct -- 2nd deriv computation of potential energy
#     '''
#
#     raise NotImplementedError("Please implement this function in the derived class.")
#
#     # Needs to return like this
#     return pe, ke_jf, ke_direct
#     # return None
#
# def compile_functions(self, inputs, spin, isospin):
#     raise NotImplementedError("Please implement this function in the derived class.")
#
# @tf.function
# def energy(self,
#     wavefunction : tf.keras.models.Model,
#     inputs       : tf.Tensor,
#     spin         : tf.Tensor,
#     isospin      : tf.Tensor = None):
#     """Compute the expectation value of energy of the supplied wavefunction.
#
#     Computes the integral of the wavefunction in this potential
#
#     Arguments:
#         wavefunction {Wavefunction model} -- Callable wavefunction object
#         inputs {tf.Tensor} -- Tensor of shape [nwalkers, nparticles, dimension]
#
#     Returns:
#         tf.tensor - energy of shape [n_walkers]
#         tf.tensor - energy_jf of shape [n_walkers]
#         tf.tensor - ke_jf of shape [n_walkers]
#         tf.tensor - ke_direct of shape [n_walkers]
#         tf.tensor - pe of shape [n_walkers]
#     """
#
#     logger.info("energy")
#
#     # This function takes the inputs
#     # And computes the expectation value of the energy at each input point
#
#     w_of_x, dw_dx, d2w_dx2 = \
#         self.compute_derivatives(wavefunction, inputs, spin, isospin)
#
#     pe, ke_jf, ke_direct = self.compute_energies(
#         wavefunction, inputs, spin, isospin, w_of_x, dw_dx, d2w_dx2)
#
#     # Total energy computations:
#     energy    = tf.squeeze(pe+ke_direct)
#     energy_jf = tf.squeeze(pe+ke_jf)
#
#     return energy, energy_jf, ke_jf, ke_direct, pe, w_of_x
