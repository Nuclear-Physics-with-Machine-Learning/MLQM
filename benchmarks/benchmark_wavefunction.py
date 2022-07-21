from jax import random
import jax.numpy as numpy

import os, sys
mlqm_dir = os.path.dirname(os.path.abspath(__file__))
mlqm_dir = os.path.dirname(mlqm_dir)
sys.path.insert(0,mlqm_dir)

from mlqm.models import initialize_wavefunction
from mlqm.samplers import MetropolisSampler, kick

from mlqm.config import ManyBodyCfg, Sampler
from omegaconf import OmegaConf

# from mlqm import DEFAULT_TENSOR_TYPE

sampler_config = Sampler()

sampler_config.n_walkers_per_observation = 1000
sampler_config.n_void_steps = 250
sampler_config.n_particles  = 16

sampler_config.n_concurrent_obs_per_rank = 1


key = random.PRNGKey(0)
key, subkey = random.split(key)

sampler = MetropolisSampler(
    sampler_config,
    subkey,
    "float64"
    )
x, spin, isospin = sampler.sample()


key, subkey = random.split(key)

c = ManyBodyCfg
c = OmegaConf.structured(c)
wavefunction, parameters = initialize_wavefunction(x, spin, isospin, subkey, sampler_config, c)

import time
start = time.time()
key, subkey = random.split(key)
x, spin, isospin = sampler.sample()
x = kick(wavefunction, parameters, x, spin, isospin, subkey, sampler_config.n_void_steps)
sampler.update(x, spin, isospin)
print(f"Time for jitting and run: {time.time() - start:.3f}")

start = time.time()
x, spin, isospin = sampler.sample()
x = kick(wavefunction, parameters, x, spin, isospin, subkey, sampler_config.n_void_steps)
sampler.update(x, spin, isospin)

print(f"Time for just run: {time.time() - start:.3f}")


start = time.time()
x, spin, isospin = sampler.sample()
x = kick(wavefunction, parameters, x, spin, isospin, subkey, sampler_config.n_void_steps)
sampler.update(x, spin, isospin)

print(f"Time for second run: {time.time() - start:.3f}")


start = time.time()
print(x.shape)
x, spin, isospin = sampler.sample()
w_of_x = wavefunction.apply(parameters, x, spin, isospin)
print(f"Time for wavefunction: {time.time() - start:.3f}")


