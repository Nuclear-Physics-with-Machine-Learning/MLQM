import pytest

import torch
import numpy

from . import NeuralWavefunction
from ..samplers.CartesianSampler import CartesianSampler


@pytest.mark.parametrize('dimension', [1,2,3])
def test_create_polynomial(dimension):

    # For each dimension, randomly pick a degree

    nn_w = NeuralWavefunction.NeuralWavefunction(dimension)

    assert True


@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_run_polynomial(dimension):

    # For each dimension, randomly pick a degree

    nn_w = NeuralWavefunction.NeuralWavefunction(dimension)


    delta = 2.0

    sampler = CartesianSampler(dimension, delta=delta, mins=-10, maxes=10)

    x = sampler.sample()

    nn_w.update_normalization(x, delta=delta**dimension)

    wavefunction = nn_w(x)


    assert torch.abs(torch.sum(wavefunction**2) * delta**dimension - 1.0) < 0.01



if __name__ == "__main__":


    test_create_polynomial(dimension = 1, degree=3)