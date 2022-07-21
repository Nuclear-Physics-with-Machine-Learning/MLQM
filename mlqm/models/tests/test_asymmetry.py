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


def swap_particles(walkers, spin, isospin, ij):
    # Switch two particles, i != j:


    new_walkers = walkers.at[:,ij[0],:].set(walkers[:,ij[1],:])
    new_walkers = new_walkers.at[:,ij[1],:].set(walkers[:,ij[0],:])

    new_spin = spin.at[:,ij[0]].set(spin[:,ij[1]])
    new_spin = new_spin.at[:,ij[1]].set(spin[:,ij[0]])

    new_isospin = isospin.at[:,ij[0]].set(isospin[:,ij[1]])
    new_isospin = new_isospin.at[:,ij[1]].set(isospin[:,ij[0]])

    return new_walkers, new_spin, new_isospin



@pytest.mark.parametrize('nwalkers', [10])
@pytest.mark.parametrize('nparticles', [2,3,4])
@pytest.mark.parametrize('ndim', [3])
@pytest.mark.parametrize('n_spin_up', [1,2])
@pytest.mark.parametrize('n_protons', [1,2])
def test_wavefunction_asymmetry(nwalkers, nparticles, ndim, n_spin_up, n_protons):

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


    a = wavefunction.apply_walkers(parameters, x, spin, isospin)

    print("a: ", a)
    key, subkey = random.split(key)
    ij = random.choice(subkey, nparticles, shape=(2,), replace=False)

    print(ij)

    x_swapped, spin_swapped, isospin_swapped \
        = swap_particles(x, spin, isospin, ij)


    a_swapped = wavefunction.apply_walkers(parameters, x_swapped, spin_swapped, isospin_swapped)

    print("a_swapped: ", a_swapped)
    # for i in range(10):
    #     a_prime = w(inputs, spins, isospins).numpy()
    # By switching two particles, we should have inverted the sign.
    assert (a + a_swapped < 1e-8 ).all()


if __name__ == "__main__":
    # test_wavefunction_asymmetry(2,2,3,2,1)
    test_wavefunction_asymmetry(3,3,3,2,2)
