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


# def test_wavefunction_spatial_slater(nwalkers, nparticles, ndim, n_spin_up, n_protons):
#
# @pytest.mark.parametrize('nwalkers', [1, 2, 10])
# @pytest.mark.parametrize('nparticles', [2,3,4])
# @pytest.mark.parametrize('ndim', [3])
# @pytest.mark.parametrize('n_spin_up', [1,2])
# @pytest.mark.parametrize('n_protons', [1,2])
# def test_wavefunction_slater(nwalkers, nparticles, ndim, n_spin_up, n_protons):
#
#     c = ManyBodyCfg()
#
#     c = OmegaConf.structured(c)
#     w = ManyBodyWavefunction(ndim, nparticles, c,
#         n_spin_up = n_spin_up, n_protons = n_protons,
#         use_spin = True, use_isospin = True
#     )
#
#     inputs, spins, isospins = generate_inputs(nwalkers, nparticles, ndim, n_spin_up, n_protons)
#     print("inputs: ", inputs)
#     print("spins: ", spins)
#     print("isospins: ", isospins)
#     a = w.construct_slater_matrix(inputs, spins, isospins)
#     print(a)
#
#     det = tf.linalg.det(a)
#
#     print(det)
#
#     # The determinant can be 0 in cases where the states are not allowed.
#     # assert (det.numpy() !=0).all()
#
#     # We've checked antisymmetry in another test but check it again here
#
#     i , j = numpy.random.choice(range(nparticles), size=2, replace=False)
#     inputs, spins, isospins = swap_particles(inputs, spins, isospins, i, j)
#
#
#
#     swapped_a = w.construct_slater_matrix(inputs, spins, isospins)
#     swapped_det = tf.linalg.det(swapped_a)
#     diff = (swapped_det + det).numpy()
#     assert (diff < 1e-8 ).all()


if __name__ == "__main__":

    test_wavefunction_slater(2,2,3,2,1,)
    test_wavefunction_slater(2,3,3,2,1,)
    test_wavefunction_slater(2,4,3,2,1,)
