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


from mlqm.models import ManyBodyWavefunction
from mlqm import DEFAULT_TENSOR_TYPE
tf.keras.backend.set_floatx(DEFAULT_TENSOR_TYPE)
from mlqm.config import ManyBodyCfg


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



# @pytest.mark.parametrize('nwalkers', [10])
# @pytest.mark.parametrize('nparticles', [2])
# @pytest.mark.parametrize('ndim', [3])
# @pytest.mark.parametrize('n_spin_up', [2])
# @pytest.mark.parametrize('n_protons', [1])
# def test_isospin_swap(nwalkers, nparticles, ndim, n_spin_up, n_protons):


#     c = ManyBodyCfg()

#     c = OmegaConf.structured(c)
#     w = ManyBodyWavefunction(ndim, nparticles, c,
#         n_spin_up = n_spin_up, n_protons = n_protons,
#         use_spin = True, use_isospin = True
#     )
#     inputs, spins, isospins = generate_inputs(nwalkers, nparticles, ndim, n_spin_up, n_protons)
#     # print("inputs: ", inputs)
#     # print("spins: ", spins)
#     # print("isospins: ", isospins)
#     a = w(inputs, spins, isospins).numpy()
#     # print("a: ", a)

#     i , j = numpy.random.choice(range(nparticles), size=2, replace=False)

#     _, _, isospins = swap_particles(numpy.copy(inputs), numpy.copy(spins), isospins, i, j)

#     # print(spins)

#     a_prime = w(inputs, spins, isospins).numpy()
#     print("isospin 2*a_prime / a: ", 2*a_prime / a)
#     # print("a_prime: ", a)
#     # By switching two particles, we should have inverted the sign.
#     assert (a != 0).all()
#     assert (a_prime != 0).all()
#     assert (2*a_prime / a - 1 == -3 ).all()




if __name__ == "__main__":
    test_isospin_slater(2,2,3, 2,1)