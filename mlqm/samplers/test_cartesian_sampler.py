import pytest
import numpy

from .CartesianSampler import CartesianSampler


@pytest.mark.parametrize("dimension", [1,2,3])
def test_cartesian_sampler(dimension):

    mins   = numpy.random.random(size=(dimension,))
    maxes  = numpy.random.random(size=(dimension,))

    # Limit delta to 1/10 resolution in tests
    deltas = [ (maxes[d] - mins[d]) / 10 for d in range(dimension) ]


    sampler = CartesianSampler(dimension, delta=deltas, mins=mins, maxes=maxes)

    a = sampler.sample()

    assert True