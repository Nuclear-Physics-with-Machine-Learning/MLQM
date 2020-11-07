import sys, os
import pathlib
import configparser
import time

import argparse
import signal
import pickle


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf


# Add the local folder to the import path:
mlqm_dir = os.path.dirname(os.path.abspath(__file__))
# mlqm_dir = os.path.dirname(mlqm_dir)
sys.path.insert(0,mlqm_dir)

from mlqm.samplers     import Estimator
from mlqm.optimization import Optimizer
from mlqm import DEFAULT_TENSOR_TYPE

tf.keras.backend.set_floatx(DEFAULT_TENSOR_TYPE)


from mlqm.samplers import MetropolisSampler

dimension  = 1
nparticles = 2
n_checks   = 10

# As an optimization, we increase the number of walkers by nobservations
sampler = MetropolisSampler(
    n           = dimension,
    nparticles  = nparticles,
    nwalkers    = 100,
    initializer = tf.random.normal,
    init_params = {"mean": 0.0, "stddev" : 0.2},
    dtype       = DEFAULT_TENSOR_TYPE)



from mlqm.hamiltonians import HarmonicOscillator
hamiltonian = HarmonicOscillator(
    M           = 1.0,
    omega       = 1.0,
)


from mlqm.models import DeepSetsWavefunction
wavefunction = DeepSetsWavefunction(dimension, nparticles, mean_subtract=False)

for i_check in range(n_checks):
    x = sampler.sample()
    # break up 2 into individual particles:
    split = [x[:,i,:] for i in range(nparticles)]

    # put it back together, could randomize with more particles:
    x_swapped = tf.stack(split,axis=1)

    untrained1 = wavefunction(x)
    untrained2 = wavefunction(x_swapped)
    assert all(tf.equal(untrained1, untrained2))

model_path = "log/HarmonicOscillator/1D/2particles/debug/harmonic_oscillator.model"
wavefunction.load_weights(model_path)

for i_check in range(n_checks):
    x = sampler.sample()
    # break up 2 into individual particles:
    split = [x[:,i,:] for i in range(nparticles)]

    # put it back together, could randomize with more particles:
    x_swapped = tf.stack(split,axis=1)

    trained1 = wavefunction(x)
    trained2 = wavefunction(x_swapped)
    assert all(tf.equal(trained1, trained2))

print(f"Symmetry is asserted in {dimension}D using DeepSet Wavefunction")
