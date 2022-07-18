from jax import random
import jax.numpy as numpy


from mlqm.models import initialize_correlator
from mlqm.samplers import MetropolisSampler, kick

from mlqm.config import DeepSetsCfg, Sampler
from omegaconf import OmegaConf

# from mlqm import DEFAULT_TENSOR_TYPE

sampler_config = Sampler()
print(sampler_config)

sampler_config.n_walkers_per_observation = 3
sampler_config.n_void_steps = 250
sampler_config.n_particles  = 2

sampler_config.n_concurrent_obs_per_rank = 1


key = random.PRNGKey(0)
key, subkey = random.split(key)

sampler = MetropolisSampler(
    sampler_config,
    subkey,
    "float64"
    )
x, _, _ = sampler.sample()


key, subkey = random.split(key)

print(x)
c = DeepSetsCfg
c = OmegaConf.structured(c)
wavefunction, parameters = initialize_correlator(x, subkey, c)

# sampler.set_wavefunction(wavefunction)


import time
start = time.time()
key, subkey = random.split(key)
x, _, _ = sampler.sample()
x = kick(wavefunction, parameters, x, subkey, sampler_config.n_void_steps)
sampler.update(x)
print(f"Time for jitting and run: {time.time() - start:.3f}")

start = time.time()
x, _, _ = sampler.sample()
x = kick(wavefunction, parameters, x, subkey, sampler_config.n_void_steps)
sampler.update(x)

print(f"Time for just run: {time.time() - start:.3f}")


start = time.time()
x, _, _ = sampler.sample()
x = kick(wavefunction, parameters, x, subkey, sampler_config.n_void_steps)
sampler.update(x)

print(f"Time for second run: {time.time() - start:.3f}")


start = time.time()
print(x.shape)
x, _, _ = sampler.sample()
w_of_x = wavefunction.apply(parameters, x)
print(f"Time for wavefunction: {time.time() - start:.3f}")

