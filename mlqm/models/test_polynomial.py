import pytest

import torch
import numpy

from . import PolynomialWavefunction


@pytest.mark.parametrize('dimension', [1,2,3])
def test_create_polynomial(dimension):

    # For each dimension, randomly pick a degree
    degree = [ numpy.random.randint(1,4) for d in range(dimension)]

    ho = PolynomialWavefunction.PolynomialWavefunction(dimension, degree, alpha=1.0)

    assert True


if __name__ == "__main__":
    test_create_polynomial(dimension = 1, degree=3)