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


    walkers[:,[i,j],:] = walkers[:, [j,i], :]

    spin[:,[i,j]] = spin[:,[j,i]]

    isospin[:,[i,j]] = isospin[:,[j,i]]

    return walkers, spin, isospin


@pytest.mark.parametrize('nwalkers', [1, 4, 10])
@pytest.mark.parametrize('nparticles', [2,3,4])
@pytest.mark.parametrize('ndim', [1,2,3])
def test_wavefunction_spatial_slater(nwalkers, nparticles, ndim):

    c = ManyBodyCfg()

    c = OmegaConf.structured(c)
    w = ManyBodyWavefunction(ndim, nparticles, c, n_spin_up = 2, n_protons = 1)

    inputs, _, _ = generate_inputs(nwalkers, nparticles, ndim, 2, 1)
    # print(inputs)

    # mean subtract:
    xinputs = inputs - numpy.reshape(numpy.mean(inputs, axis=1), (nwalkers, 1, ndim))
    a = w.compute_spatial_slater(inputs).numpy()

    print("a: ", a)

    # If we change one particle, it should change just one column of the matrix.
    for i_particle in range(nparticles):
        new_inputs = inputs.copy()
        new_inputs[:,i_particle,:] += 1
        # print("inputs - new_inputs: ", inputs - new_inputs)
        a_prime = w.compute_spatial_slater(new_inputs).numpy()

        # print("a_prime: ", a_prime)

        diff = a - a_prime
        # print(f"diff[:, {i_particle}, :]: ", diff[:, i_particle, :])
        # print(f"diff: ", diff)
        assert (diff[:,i_particle,:] != 0).all()

        diff = numpy.delete(diff, i_particle, 1)
        # print(f"numpy.delete(diff, {i_particle}, 1): ", diff )

        assert (diff == 0).all()

@pytest.mark.parametrize('nwalkers', [1, 4, 10])
@pytest.mark.parametrize('nparticles', [2,3])
@pytest.mark.parametrize('ndim', [1,2,3])
def test_spin_slater(nwalkers, nparticles, ndim, n_spin_up, n_protons):

    c = ManyBodyCfg()

    c = OmegaConf.structured(c)
    w = ManyBodyWavefunction(ndim, nparticles, c, n_spin_up, n_protons)

    _, spins, _ = generate_inputs(nwalkers, nparticles, ndim, n_spin_up, n_protons)
    # print("spins: ", spins)
    a = w.compute_spin_slater(spins).numpy()
    # print("a: ", a)

    # print(numpy.linalg.det(a))

    # The slater determinant for spins should

@pytest.mark.parametrize('nwalkers', [1, 4, 10])
@pytest.mark.parametrize('nparticles', [2,3])
@pytest.mark.parametrize('ndim', [1,2,3])
@pytest.mark.parametrize('n_spin_up',[1])
@pytest.mark.parametrize('n_protons',[1])
def test_isospin_slater(nwalkers, nparticles, ndim, n_spin_up, n_protons):

    c = ManyBodyCfg()

    c = OmegaConf.structured(c)
    w = ManyBodyWavefunction(ndim, nparticles, c, n_spin_up, n_protons)
    # print(w.isospin_spinor_2d)
    _, _, isospins = generate_inputs(nwalkers, nparticles, ndim, n_spin_up, n_protons)
    # print("isospins: ", isospins)
    a = w.compute_isospin_slater(isospins).numpy()
    # print("isospin_slater: ", a)

    # print(numpy.linalg.det(a))

@pytest.mark.parametrize('nwalkers', [10])
@pytest.mark.parametrize('nparticles', [3])
@pytest.mark.parametrize('ndim', [3])
def test_wavefunction_slater(nwalkers, nparticles, ndim, n_spin_up, n_protons):

    c = ManyBodyCfg()

    c = OmegaConf.structured(c)
    w = ManyBodyWavefunction(ndim, nparticles, c, n_spin_up = n_spin_up, n_protons = n_protons)

    inputs, spins, isospins = generate_inputs(nwalkers, nparticles, ndim, n_spin_up, n_protons)
    print("inputs: ", inputs)
    print("spins: ", spins)
    print("isospins: ", isospins)
    a = w.construct_slater_matrix(inputs, spins, isospins)
    print(a)

    sign, logdet = tf.linalg.logdet(a)

    det = sign * tf.exp(logdet)
    print(det)

    assert (det.numpy() !=0).all()


@pytest.mark.parametrize('nwalkers', [10])
@pytest.mark.parametrize('nparticles', [3])
@pytest.mark.parametrize('ndim', [3])
def test_wavefunction_asymmetry(nwalkers, nparticles, ndim):

    c = ManyBodyCfg()

    c = OmegaConf.structured(c)
    w = ManyBodyWavefunction(ndim, nparticles, c, n_spin_up = n_spin_up, n_protons = n_protons)

    inputs, spins, isospins = generate_inputs(nwalkers, nparticles, ndim, n_spin_up, n_protons)
    print("inputs: ", inputs)
    print("spins: ", spins)
    print("isospins: ", isospins)
    a = w(inputs, spins, isospins).numpy()
    print("a: ", a)

    i , j = numpy.random.choice(range(nparticles), size=2, replace=False)

    inputs, spins, isospins = swap_particles(inputs, spins, isospins, i, j)

    a_prime = w(inputs, spins, isospins).numpy()
    print("a_prime: ", a)
    # By switching two particles, we should have inverted the sign.
    assert (a != 0).all()
    assert (a_prime != 0).all()
    assert (a + a_prime == 0 ).all()

#
# print(inputs, spins, isospins)
# print(a)
# print(w.construct_slater_matrix(inputs, spins, isospins))
#
# # Make a swap:
#
# print(inputs, spins, isospins)
# a = w(inputs, spins, isospins)
# print(a)
# print(w.construct_slater_matrix(inputs, spins, isospins))

if __name__ == "__main__":
    test_wavefunction_slater(2,2,3,2,1)
    # test_wavefunction_asymmetry(2,3,3)
    # test_spin_slater(2,2,3, 2,1)
    # test_isospin_slater(2,2,3, 2,1)
