import os, sys, pathlib
from dataclasses import dataclass, field
import numpy
import pytest

import tensorflow as tf

from omegaconf import OmegaConf

# Add the mlqm path:
current_dir = pathlib.Path(__file__).parent.resolve()
init = pathlib.Path("__init__.py")

while current_dir != current_dir.root:

    if (current_dir / init).is_file():
        current_dir = current_dir.parent
    else:
        break

# Here, we break and add the directory to the path:
sys.path.insert(0,str(current_dir))

from mlqm import DEFAULT_TENSOR_TYPE
tf.keras.backend.set_floatx(DEFAULT_TENSOR_TYPE)

# For the wavefunction:
from mlqm.models import ManyBodyWavefunction
from mlqm.config import ManyBodyCfg

# For the hamiltonian:
from mlqm.hamiltonians import Hamiltonian
from mlqm.config import NuclearHamiltonian as H_config

# Generate fake particles
#
# nwalkers = 4
# nparticles = 2
# ndim = 3

def generate_inputs(nwalkers, nparticles, ndim, n_spin_up, n_protons):

    inputs = numpy.random.uniform(size=[nwalkers, nparticles, ndim])


    # Note that we initialize with NUMPY for ease of indexing and shuffling
    spin_walkers = numpy.zeros(shape=(nwalkers, nparticles)) - 1
    for i in range(n_spin_up):
        if i < spin_walkers.shape[-1]:
            spin_walkers[:,i] += 2

    # Shuffle the spin up particles on each axis:

    # How to compute many permutations at once?
    #  Answer from https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
    # Bottom line: gen random numbers for each axis, sort only in that axis,
    # and apply the permutations
    idx = numpy.random.rand(*spin_walkers.shape).argsort(axis=1)
    spin_walkers = numpy.take_along_axis(spin_walkers, idx, axis=1)

    # Note that we initialize with NUMPY for ease of indexing and shuffling
    isospin_walkers = numpy.zeros(shape=(nwalkers, nparticles)) - 1
    for i in range(n_protons):
        if i < isospin_walkers.shape[-1]:
            isospin_walkers[:,i] += 2

    # Shuffle the spin up particles on each axis:

    # How to compute many permutations at once?
    #  Answer from https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
    # Bottom line: gen random numbers for each axis, sort only in that axis,
    # and apply the permutations
    idx = numpy.random.rand(*isospin_walkers.shape).argsort(axis=1)
    isospin_walkers = numpy.take_along_axis(isospin_walkers, idx, axis=1)

    return inputs, spin_walkers, isospin_walkers

def swap_particles(walkers, spin, isospin, i, j):
    # Switch two particles, i != j:


    walkers[:, [i,j], :] = walkers[:, [j,i], :]

    spin[:,[i,j]] = spin[:,[j,i]]

    isospin[:,[i,j]] = isospin[:,[j,i]]

    return walkers, spin, isospin


# class analytic_model(tf.keras.models.Model):
#
#     def __init__(self, ndim, nparticles):
#         tf.keras.models.Model.__init__(self)
#
#         p = tf.random.uniform(shape=(ndim,),dtype=DEFAULT_TENSOR_TYPE)
#         self.ndim = ndim
#         self.nparticles = nparticles
#         self.parameters = tf.Variable(p)
#
#     def __call__(self, inputs, spin, isospin, training=False):
#
#         # We ignore the spin and isospin.
#         mean = tf.reduce_mean(inputs, axis=1)
#         xinputs = inputs - mean[:,None,:]
#
#
#         inputs2 = tf.pow(xinputs, 2)
#         dot = - inputs2 * self.parameters
#
#         return tf.exp(tf.reduce_sum(dot, axis=(1,2)))
#
#     def first_derivative(self, inputs):
#
#         vals = self.__call__(inputs, None, None)
#         first_deriv = -2*inputs*vals*self.parameters
#
#         return first_deriv
#
# @pytest.mark.parametrize('nwalkers', [10])
# @pytest.mark.parametrize('nparticles', [2,3,4,5])
# @pytest.mark.parametrize('ndim', [1,2,3])
# def test_hamiltonian_analytic(nwalkers, nparticles, ndim):
#
#
#     w = analytic_model(ndim, nparticles)
#
#
#     inputs, spins, isospins = generate_inputs(nwalkers, nparticles, ndim, 2, 1)
#
#     #mean subtract:
#     xinputs = tf.convert_to_tensor(
#         inputs - numpy.reshape(numpy.mean(inputs, axis=1), (nwalkers, 1, ndim)),
#         dtype = DEFAULT_TENSOR_TYPE)
#
#     w_local = w(inputs, spins, isospins)
#
#     hc = OmegaConf.structured(H_config())
#
#     h = Hamiltonian(hc)
#
#     w_of_x, dw_dx, d2w_dx2 = h.compute_derivatives(w, xinputs, spins, isospins)
#
#     print("First deriv analytic: ", w.first_derivative(inputs))
#     print("First deriv numerical: ", dw_dx)
#
#     dw_dx = dw_dx.numpy()
#     d2w_dx2 = d2w_dx2.numpy()
#
#     # Make sure we've got the same values of the wavefunction:
#     assert ((w_local - w_of_x).numpy() < 1e-8).all()
#
#     # We proceed to check the derivatives with finte difference methods.
#
#     # Need a "difference" term:
#     kick = numpy.zeros(shape = inputs.shape)
#     kick_size = 1e-4
#
#     # First and second order derivative.  Check it for each dimension.
#     for i_dim in range(ndim):
#
#         # select a random particle to kick:
#         i_kicked_particle = numpy.random.choice(range(nparticles), size=(nwalkers,))
#         print(i_kicked_particle)
#         # i_kicked_particle = 0
#         this_kick = numpy.copy(kick)
#         # Have to create an index for walkers to slice:
#         walkers = numpy.arange(nwalkers)
#         # Only applying to particle 0
#         this_kick[walkers,i_kicked_particle,i_dim] += kick_size
#         kicked_up_input = xinputs + \
#             tf.convert_to_tensor(this_kick, dtype=DEFAULT_TENSOR_TYPE)
#
#         kicked_double_up_input = xinputs + \
#             tf.convert_to_tensor(2*this_kick, dtype=DEFAULT_TENSOR_TYPE)
#         # # Mean subtract:
#         # up_xinputs = kicked_up_input - \
#         #     numpy.reshape(numpy.mean(kicked_up_input, axis=1), (nwalkers, 1, ndim))
#
#         kicked_down_input = xinputs - \
#             tf.convert_to_tensor(this_kick, dtype=DEFAULT_TENSOR_TYPE)
#
#         kicked_double_down_input = xinputs - \
#             tf.convert_to_tensor(2*this_kick, dtype=DEFAULT_TENSOR_TYPE)
#         # down_xinputs = kicked_down_input - \
#         #     numpy.reshape(numpy.mean(kicked_down_input, axis=1), (nwalkers, 1, ndim))
#
#
#         # In this case, *because* there is a mean subtraction,
#         # we will calculate a derivate for the first particle only.
#         # The derivatives for the other particles based on this kick will be
#         # flipped sign.
#
#         # The total kick will actually be kick_size / nparticles, because of
#         # the effect of mean subtraction
#
#         # Differences:
#         w_up = w(kicked_up_input, spins, isospins)
#         w_down = w(kicked_down_input, spins, isospins)
#         w_up_up = w(kicked_double_up_input, spins, isospins)
#         w_down_down = w(kicked_double_down_input, spins, isospins)
#
#
#
#         # Use numpy to make slicing easier
#         w_prime_fd = tf.reshape((w_up - w_down) / (2*kick_size), (nwalkers,)).numpy()
#         # What about the second derivative?
#
#         # https://math.stackexchange.com/questions/3756717/finite-differences-second-derivative-as-successive-application-of-the-first-deri
#         # This gives precision of O(kick**4)
#         w_prime_prime_num = -w_down_down + 16*w_down - 30* w_of_x + 16 * w_up - w_up_up
#         w_prime_prime_fd = tf.reshape(w_prime_prime_num/ (12 * kick_size**2), (nwalkers,)).numpy()
#
#
#
#         # Now, verify everything is correct.
#         # print("dw_dx: ", dw_dx)
#          # slice to just the dimension we're moving, all walkers
#         target = dw_dx[walkers,i_kicked_particle,i_dim]
#         second_target = d2w_dx2[walkers,i_kicked_particle, i_dim]
#
#         # print("target: ", target)
#         # print("w_prime_fd: ", w_prime_fd)
#         # The tolerance on the second order first derivative is net_kick**2
#         # print("First difference: ", (w_prime_fd - target) )
#         # print("Target tolerance: ", kick_size**2)
#         assert( numpy.abs(w_prime_fd - target) < kick_size **2).all()
#
#         print("second_target: ", second_target)
#         print("w_prime_prime_fd: ", w_prime_prime_fd)
#         print("2nd difference: ", w_prime_prime_fd - second_target)
#         assert (numpy.abs(w_prime_prime_fd - second_target) < kick_size ).all()

        #

@pytest.mark.parametrize('nwalkers', [10])
@pytest.mark.parametrize('nparticles', [2,3,4,5])
@pytest.mark.parametrize('ndim', [1,2,3])
@pytest.mark.parametrize('spin', [True])
@pytest.mark.parametrize('iso_spin', [True])
def test_hamiltonian(nwalkers, nparticles, ndim, spin, iso_spin):

    n_spin_up = 2
    n_protons = 1
    c = ManyBodyCfg()

    c = OmegaConf.structured(c)
    w = ManyBodyWavefunction(ndim, nparticles, c,
        n_spin_up = n_spin_up, n_protons = n_protons,
        use_spin = True, use_isospin = True
    )



    inputs, spins, isospins = generate_inputs(nwalkers, nparticles, ndim, n_spin_up, n_protons)

    #mean subtract:
    xinputs = tf.convert_to_tensor(
        inputs - numpy.reshape(numpy.mean(inputs, axis=1), (nwalkers, 1, ndim)))

    w_local = w(inputs, spins, isospins)

    hc = OmegaConf.structured(H_config())

    h = Hamiltonian(hc)

    w_of_x, dw_dx, d2w_dx2 = h.compute_derivatives(w, xinputs, spins, isospins)

    dw_dx = dw_dx.numpy()
    d2w_dx2 = d2w_dx2.numpy()

    # Make sure we've got the same values of the wavefunction:
    assert ((w_local - w_of_x).numpy() < 1e-8).all()

    # We proceed to check the derivatives with finte difference methods.

    # Need a "difference" term:
    kick = numpy.zeros(shape = inputs.shape)
    kick_size = 1e-4

    # First and second order derivative.  Check it for each dimension.
    for i_dim in range(ndim):

        # select a random particle to kick:
        i_kicked_particle = numpy.random.choice(range(nparticles), size=(nwalkers,))
        print(i_kicked_particle)
        # i_kicked_particle = 0
        this_kick = numpy.copy(kick)
        # Have to create an index for walkers to slice:
        walkers = numpy.arange(nwalkers)
        # Only applying to particle 0
        this_kick[walkers,i_kicked_particle,i_dim] += kick_size
        kicked_up_input = xinputs + \
            tf.convert_to_tensor(this_kick, dtype=DEFAULT_TENSOR_TYPE)

        kicked_double_up_input = xinputs + \
            tf.convert_to_tensor(2*this_kick, dtype=DEFAULT_TENSOR_TYPE)
        # # Mean subtract:
        # up_xinputs = kicked_up_input - \
        #     numpy.reshape(numpy.mean(kicked_up_input, axis=1), (nwalkers, 1, ndim))

        kicked_down_input = xinputs - \
            tf.convert_to_tensor(this_kick, dtype=DEFAULT_TENSOR_TYPE)

        kicked_double_down_input = xinputs - \
            tf.convert_to_tensor(2*this_kick, dtype=DEFAULT_TENSOR_TYPE)
        # down_xinputs = kicked_down_input - \
        #     numpy.reshape(numpy.mean(kicked_down_input, axis=1), (nwalkers, 1, ndim))


        # In this case, *because* there is a mean subtraction,
        # we will calculate a derivate for the first particle only.
        # The derivatives for the other particles based on this kick will be
        # flipped sign.

        # The total kick will actually be kick_size / nparticles, because of
        # the effect of mean subtraction

        # Differences:
        w_up = w(kicked_up_input, spins, isospins)
        w_down = w(kicked_down_input, spins, isospins)
        w_up_up = w(kicked_double_up_input, spins, isospins)
        w_down_down = w(kicked_double_down_input, spins, isospins)



        # Use numpy to make slicing easier
        w_prime_fd = tf.reshape((w_up - w_down) / (2*kick_size), (nwalkers,)).numpy()
        # What about the second derivative?

        # https://math.stackexchange.com/questions/3756717/finite-differences-second-derivative-as-successive-application-of-the-first-deri
        # This gives precision of O(kick**4)
        w_prime_prime_num = -w_down_down + 16*w_down - 30* w_of_x + 16 * w_up - w_up_up
        w_prime_prime_fd = tf.reshape(w_prime_prime_num/ (12 * kick_size**2), (nwalkers,)).numpy()



        # Now, verify everything is correct.
        # print("dw_dx: ", dw_dx)
         # slice to just the dimension we're moving, all walkers
        target = dw_dx[walkers,i_kicked_particle,i_dim]
        second_target = d2w_dx2[walkers,i_kicked_particle, i_dim]

        # print("target: ", target)
        # print("w_prime_fd: ", w_prime_fd)
        # The tolerance on the second order first derivative is net_kick**2
        # print("First difference: ", (w_prime_fd - target) )
        # print("Target tolerance: ", kick_size**2)
        assert( numpy.abs(w_prime_fd - target) < kick_size **2).all()

        print("second_target: ", second_target)
        print("w_prime_prime_fd: ", w_prime_prime_fd)
        print("2nd difference: ", w_prime_prime_fd - second_target)
        assert (numpy.abs(w_prime_prime_fd - second_target) < kick_size ).all()

        #



if __name__ == "__main__":
    test_hamiltonian_analytic(3,2,3)
    # test_hamiltonian(1,2,3,False, False)
