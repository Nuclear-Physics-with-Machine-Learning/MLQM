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

@pytest.mark.parametrize('nwalkers', [10])
@pytest.mark.parametrize('nparticles', [2,3,4])
@pytest.mark.parametrize('ndim', [1,2,3])
@pytest.mark.parametrize('n_spin_up', [1,2])
@pytest.mark.parametrize('n_protons', [1,2])
@pytest.mark.parametrize('func', ['compute_spatial_slater', 'compute_spin_slater', 'compute_isospin_slater'])
def test_wavefunction_slater_component(nwalkers, nparticles, ndim, n_spin_up, n_protons, func):


    # Create a config:
    c = ManyBodyCfg()

    # Cast to structured config:
    c = OmegaConf.structured(c)
    # Init the wavefunction object:
    w = ManyBodyWavefunction(ndim, nparticles, c,
        n_spin_up = n_spin_up, n_protons = n_protons,
        use_spin = True, use_isospin = True
    )

    # Create inputs:
    inputs, spin, isospin = generate_inputs(nwalkers, nparticles, ndim, n_spin_up, n_protons)
    # print(inputs)

    # mean subtract:
    xinputs = inputs - numpy.reshape(numpy.mean(inputs, axis=1), (nwalkers, 1, ndim))

    def make_computation(_inputs, _spin, _isospin, w, func):

        if func == "compute_spatial_slater":
            # Compute the SPATIAL slater only:
            _a = w.compute_spatial_slater(_inputs).numpy()
        elif func == "compute_spin_slater":
            # Spin only
            _a = w.compute_spin_slater(_spin).numpy()
        elif func == "compute_isospin_slater":
            # Isospin only
            _a = w.compute_isospin_slater(_isospin).numpy()
        return _a

    a = make_computation(xinputs, spin, isospin, w, func)
    print("a: ", a)

    def kick_inputs(_inputs, _spin, _isospin, func):

        if func == "compute_spatial_slater":
            # Compute the SPATIAL slater only:
            _inputs = _inputs.copy()
            _inputs[:,i_particle,:] += 1
        elif func == "compute_spin_slater":
            # Spin only
            _spin = _spin.copy()
            _spin[:,i_particle] += 1
        elif func == "compute_isospin_slater":
            # Isospin only
            _isospin = _isospin.copy()
            _isospin[:,i_particle] += 1
        return _inputs, _spin, _isospin

    # If we change one particle, it should change just one column of the matrix.
    for i_particle in range(nparticles):

        print(xinputs)
        new_inputs, new_spin, new_isospin = kick_inputs(xinputs, spin, isospin, func)
        print(new_inputs)
        # print("inputs - new_inputs: ", inputs - new_inputs)
        # a_prime = w.__getattribute__(func)(new_inputs).numpy()
        a_prime = make_computation(new_inputs, new_spin, new_isospin, w, func)
        print(a_prime)

        # print("a_prime: ", a_prime)

        diff = a - a_prime
        # print(f"diff[:, {i_particle}, :]: ", diff[:, i_particle, :])
        assert (diff[:,:,i_particle] != 0).all()

        diff = numpy.delete(diff, i_particle, 2)
        # print(f"numpy.delete(diff, {i_particle}, 1): ", diff )

        assert (diff == 0).all()

    # if we swap two particles, any two, the spatial slater should exchange two columns.
    # Practically, the determinant should change sign.
    i , j = numpy.random.choice(range(nparticles), size=2, replace=False)
    xinputs, spin, isospin = swap_particles(xinputs, spin, isospin, i, j)

    original_det = numpy.linalg.det(a)
    swapped_a = make_computation(xinputs, spin, isospin, w, func)

    # The index of the slater matrix should be [walker, state, particle, ]
    print(a[0,:,i])
    print(swapped_a[0,:,j])
    print("Swapped a: ", swapped_a)
    new_det = numpy.linalg.det(swapped_a)
    print("original_det: ", original_det)
    print("new_det: ", new_det)
    print("original_det + new_det: ", original_det + new_det)

    # Assert 0 within machine tolerance:
    assert  (original_det + new_det < 1e-8).all()




if __name__ == "__main__":
    test_wavefunction_slater_component(2,2,3,2,1, "compute_spatial_slater")
    test_wavefunction_slater_component(2,3,3,2,1, "compute_spatial_slater")
    test_wavefunction_slater_component(2,4,3,2,1, "compute_spatial_slater")
    test_wavefunction_slater_component(2,2,3,2,1, "compute_spin_slater")
    test_wavefunction_slater_component(2,3,3,2,1, "compute_spin_slater")
    test_wavefunction_slater_component(2,4,3,2,1, "compute_spin_slater")
    test_wavefunction_slater_component(2,2,3,2,1, "compute_isospin_slater")
    test_wavefunction_slater_component(2,3,3,2,1, "compute_isospin_slater")
    test_wavefunction_slater_component(2,4,3,2,1, "compute_isospin_slater")
