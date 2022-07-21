from jax import random
import jax.numpy as numpy

import sys, os
import pathlib
from dataclasses import dataclass, field
import pytest

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

from mlqm.models import initialize_wavefunction
from mlqm.samplers import MetropolisSampler, kick

from mlqm.config import ManyBodyCfg, Sampler
from omegaconf import OmegaConf


from jax.config import config; config.update("jax_enable_x64", True)
from jax import vmap

def swap_particles(walkers, spin, isospin, ij):
    # Switch two particles, i != j:


    new_walkers = walkers.at[:,ij[0],:].set(walkers[:,ij[1],:])
    new_walkers = new_walkers.at[:,ij[1],:].set(walkers[:,ij[0],:])

    new_spin = spin.at[:,ij[0]].set(spin[:,ij[1]])
    new_spin = new_spin.at[:,ij[1]].set(spin[:,ij[0]])

    new_isospin = isospin.at[:,ij[0]].set(isospin[:,ij[1]])
    new_isospin = new_isospin.at[:,ij[1]].set(isospin[:,ij[0]])

    return new_walkers, new_spin, new_isospin



# def test_wavefunction_spatial_slater(nwalkers, nparticles, ndim, n_spin_up, n_protons):

@pytest.mark.parametrize('nwalkers', [10])
@pytest.mark.parametrize('nparticles', [2])
@pytest.mark.parametrize('ndim', [1,2,3])
@pytest.mark.parametrize('n_spin_up', [1,2])
@pytest.mark.parametrize('n_protons', [1,2])
@pytest.mark.parametrize('func', ['compute_spatial_slater', 'compute_spin_slater', 'compute_isospin_slater'])
def test_wavefunction_slater_component(nwalkers, nparticles, ndim, n_spin_up, n_protons, func):

    # Create the sampler config:
    sampler_config = Sampler()

    sampler_config.n_walkers_per_observation = nwalkers
    sampler_config.n_concurrent_obs_per_rank = 1
    sampler_config.n_particles  = nparticles
    sampler_config.n_dim = ndim
    sampler_config.n_spin_up = n_spin_up
    sampler_config.n_protons = n_protons


    # Initialize the sampler:
    key = random.PRNGKey(0)
    key, subkey = random.split(key)

    sampler = MetropolisSampler(
        sampler_config,
        subkey,
        "float64"
        )
    x, spin, isospin = sampler.sample()


    # Create the wavefunction:
    key, subkey = random.split(key)

    c = ManyBodyCfg
    c = OmegaConf.structured(c)

    wavefunction, parameters = initialize_wavefunction(
        x, spin, isospin, subkey, sampler_config, c)


    a = wavefunction.apply(parameters, x, spin, isospin)


    # mean subtract:
    xinputs = x - numpy.reshape(numpy.mean(x, axis=1), (nwalkers, 1, ndim))
    xinputs = x - x.mean(axis=0)
    import pdb; pdb.set_trace()

    def make_computation(_inputs, _spin, _isospin, w, func):

        if func == "compute_spatial_slater":
            # Compute the SPATIAL slater only:
            _a = vmap(w.compute_spatial_slater(_inputs), in_axes=[None,0])
        elif func == "compute_spin_slater":
            # Spin only
            _a = vmap(w.compute_spin_slater(_spin), in_axes=[None,0])
        elif func == "compute_isospin_slater":
            # Isospin only
            _a = vmap(w.compute_isospin_slater(_isospin), in_axes=[None,0])
        return _a

    a = make_computation(xinputs, spin, isospin, wavefunction, func)
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
        print(f"diff[:, {i_particle}, :]: ", diff[:, i_particle, :])
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
    test_wavefunction_slater_component(3,2,3,2,1, "compute_spatial_slater")
    # test_wavefunction_slater_component(2,3,3,2,1, "compute_spatial_slater")
    # test_wavefunction_slater_component(2,4,3,2,1, "compute_spatial_slater")
    # test_wavefunction_slater_component(2,2,3,2,1, "compute_spin_slater")
    # test_wavefunction_slater_component(2,3,3,2,1, "compute_spin_slater")
    # test_wavefunction_slater_component(2,4,3,2,1, "compute_spin_slater")
    # test_wavefunction_slater_component(2,2,3,2,1, "compute_isospin_slater")
    # test_wavefunction_slater_component(2,3,3,2,1, "compute_isospin_slater")
    # test_wavefunction_slater_component(2,4,3,2,1, "compute_isospin_slater")
