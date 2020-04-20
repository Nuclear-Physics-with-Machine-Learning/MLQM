import pytest

import torch
import numpy

from . import HarmonicOscillator
from ..samplers.CartesianSampler import CartesianSampler

@pytest.mark.parametrize('dimension', [1,2,3])
def test_create_harmonic_oscillator(dimension):

    # For each dimension, randomly pick a degree
    degree = [ numpy.random.randint(0,4) for d in range(dimension)]

    ho = HarmonicOscillator.HarmonicOscillator(dimension, degree, alpha=1.0)

    assert True

@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_run_harmonic_oscillator(dimension):

    # For each dimension, randomly pick a degree
    degree = [ numpy.random.randint(0,4) for d in range(dimension)]
    ho = HarmonicOscillator.HarmonicOscillator(dimension, degree, alpha=1.0)


    delta = 0.5

    sampler = CartesianSampler(dimension, delta=delta, mins=-10, maxes=10)

    x = sampler.sample()

    wavefunction = ho(x)


    assert torch.abs(torch.sum(wavefunction**2) * delta**dimension - 1.0) < 0.01
