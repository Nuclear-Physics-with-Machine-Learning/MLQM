import pytest

import torch
import numpy

from . import HarmonicOscillator


@pytest.mark.parametrize('dimension', [1,2,3])
def test_create_harmonic_oscillator(dimension):

    # For each dimension, randomly pick a degree
    degree = [ numpy.random.randint(0,4) for d in range(dimension)]

    ho = HarmonicOscillator.HarmonicOscillator(dimension, degree, alpha=1.0)

    assert True

@pytest.mark.parametrize('dimension', [1,2,3])
def test_run_harmonic_oscillator(dimension):

    # For each dimension, randomly pick a degree
    degree = [ numpy.random.randint(0,4) for d in range(dimension)]

    ho = HarmonicOscillator.HarmonicOscillator(dimension, degree, alpha=1.0)


    # Create a set of input data:
    # This is preparing input data that is differentiable:
    delta = 0.1

    dim_coordinates = [ numpy.arange(-10, 10, delta, dtype=numpy.float32) for i in range(dimension) ]
    _ = _x.reshape((_x.shape[0], 1))

    x = torch.autograd.Variable(torch.tensor(_x), requires_grad = True)

