from jax import random
import jax.numpy as numpy
import time

import os, sys
mlqm_dir = os.path.dirname(os.path.abspath(__file__))
mlqm_dir = os.path.dirname(mlqm_dir)
sys.path.insert(0,mlqm_dir)

from mlqm.models import initialize_wavefunction
from mlqm.samplers import MetropolisSampler, kick

from mlqm.config import ManyBodyCfg, Sampler
from omegaconf import OmegaConf
from jax import vmap, jit

# from mlqm import DEFAULT_TENSOR_TYPE

# Create the sampler config:
sampler_config = Sampler()

sampler_config.n_walkers_per_observation = 1000
sampler_config.n_concurrent_obs_per_rank = 1
sampler_config.n_particles  = 4
sampler_config.n_dim = 3
sampler_config.n_spin_up = 1
sampler_config.n_protons = 1


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
c.mean_subtract=False
c = OmegaConf.structured(c)

wavefunction, parameters = initialize_wavefunction(
    x, spin, isospin, subkey, sampler_config, c)

w_local = wavefunction.apply_walkers(parameters, x, spin, isospin)


from mlqm.hamiltonians import compute_derivatives




start = time.time()
w_of_x, dw_dx, d2w_dx2 = compute_derivatives(wavefunction, parameters, x, spin, isospin)

print(f"Time for jitting and run: {time.time() - start:.3f}")

print("Begin benchmark measurements")

times = []
for i in range(10):

    start = time.time()
    x, spin, isospin = sampler.sample()
    w_of_x, dw_dx, d2w_dx2 = compute_derivatives(wavefunction, parameters, x, spin, isospin)
    w_of_x.block_until_ready()
    t = time.time() - start
    # print(f"Time for just run: {t:.3f}")
    times.append(t)

times = numpy.asarray(times)
print(times)

print(f"Metropolis Mean time for {sampler_config.n_void_steps} kicks: {numpy.mean(times):.4f}  +/- {numpy.std(times):.4f}")

#
# print(f"Time for wavefunction: {time.time() - start:.3f}")
