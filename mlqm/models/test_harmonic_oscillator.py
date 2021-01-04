import pytest

import tensorflow as tf
import numpy

from . import HarmonicOscillatorWavefunction
from ..samplers import CartesianSampler

@pytest.mark.parametrize('dimension', [1,2,3])
@pytest.mark.parametrize('nparticles', [1,2])
def test_create_harmonic_oscillator(dimension, nparticles):

    # For each dimension, randomly pick a degree
    degree = [ numpy.random.randint(0,4) for d in range(dimension)]

    ho_w = HarmonicOscillatorWavefunction(dimension, nparticles, degree, alpha=1.0)

    assert True

@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('nparticles', [1,2])
def test_run_harmonic_oscillator(dimension, nparticles):

    # For each dimension, randomly pick a degree
    degree = [ numpy.random.randint(0,4) for d in range(dimension)]
    ho_w = HarmonicOscillatorWavefunction(dimension, nparticles, degree, alpha=1.0)


    delta = 0.5

    sampler = CartesianSampler(dimension, delta=delta, mins=-10, maxes=10)

    x = sampler.sample()

    wavefunction = ho_w(x)


    assert tf.abs(tf.reduce_sum(wavefunction**2) * delta**dimension - 1.0) < 0.01
