import tensorflow as tf

from mlqm.models import DeepSetsCorrelator
from mlqm.samplers import MetropolisSampler

from mlqm.config import DeepSetsCfg
from omegaconf import OmegaConf

from mlqm import DEFAULT_TENSOR_TYPE
tf.keras.backend.set_floatx(DEFAULT_TENSOR_TYPE)

n_walkers = 2000
n_kicks   = 250
n_particles = 16
sampler = MetropolisSampler(
        n           = 3,
        nwalkers    = n_walkers,
        nparticles  = n_particles,
        initializer = tf.random.normal,
        init_params = {"mean": 0.0, "stddev" : 0.2},
        n_spin_up   = 2,
        n_protons   = 1,
        use_spin    = False,
        use_isospin = False,
        dtype       = tf.float64)


c = DeepSetsCfg
c = OmegaConf.structured(c)
print(c)
wavefunction = DeepSetsCorrelator(ndim=3,nparticles=n_particles,configuration=c)

x, _, _ = sampler.sample()

import time
start = time.time()
sampler.kick(wavefunction, 
    kicker        = tf.random.normal,
    kicker_params = {"mean":0.0, "stddev": 0.2},
    nkicks        = n_kicks)
print(f"Time for jitting and run: {time.time() - start:.3f}")

start = time.time()
sampler.kick(wavefunction, 
    kicker        = tf.random.normal,
    kicker_params = {"mean":0.0, "stddev": 0.2},
    nkicks        = n_kicks)
print(f"Time for just run: {time.time() - start:.3f}")


start = time.time()
sampler.kick(wavefunction, 
    kicker        = tf.random.normal,
    kicker_params = {"mean":0.0, "stddev": 0.2},
    nkicks        = n_kicks)
print(f"Time for second run: {time.time() - start:.3f}")


start = time.time()
w_of_x = wavefunction(x)
print(f"Time for wavefunction: {time.time() - start:.3f}")
