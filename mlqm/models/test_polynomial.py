import pytest

import torch
import numpy

from . import PolynomialWavefunction
from ..samplers.CartesianSampler import CartesianSampler


@pytest.mark.parametrize('dimension', [1,2,3])
def test_create_polynomial(dimension):

    # For each dimension, randomly pick a degree
    degree = [ numpy.random.randint(1,4) for d in range(dimension)]

    poly_w = PolynomialWavefunction(dimension, degree)

    assert True


@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_run_polynomial(dimension):

    # For each dimension, randomly pick a degree
    degree = [ numpy.random.randint(0,4) for d in range(dimension)]
    poly_w = PolynomialWavefunction(dimension, degree)


    delta = 0.5

    sampler = CartesianSampler(dimension, delta=delta, mins=-10, maxes=10)

    x = sampler.sample()

    poly_w.update_normalization(x, delta=delta**dimension)

    wavefunction = poly_w(x)


    assert torch.abs(torch.sum(wavefunction**2) * delta**dimension - 1.0) < 0.01



if __name__ == "__main__":


    test_create_polynomial(dimension = 1, degree=3)