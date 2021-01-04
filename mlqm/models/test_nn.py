import pytest

import tensorflow as tf
import numpy

from . import NeuralWavefunction
from ..samplers import CartesianSampler


@pytest.mark.parametrize('dimension', [1,2,3])
def test_create_polynomial(dimension):

    # For each dimension, randomly pick a degree

    nn_w = NeuralWavefunction(dimension)

    assert True


@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_run_polynomial(dimension):

    # For each dimension, randomly pick a degree

    nn_w = NeuralWavefunction(dimension)


    delta = 2.0

    sampler = CartesianSampler(dimension, delta=delta, mins=-10, maxes=10)

    x = sampler.sample()

    nn_w.update_normalization(x, delta=delta**dimension)

    wavefunction = nn_w(x)


    assert tf.abs(tf.reduce_sum(wavefunction**2) * delta**dimension - 1.0) < 0.01



if __name__ == "__main__":


    test_create_polynomial(dimension = 1, degree=3)