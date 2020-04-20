import pytest

import torch
import numpy

from .. import H_BAR
from . import HarmonicOscillator
from ..models   import HarmonicOscillatorWavefunction
from ..samplers import CartesianSampler

@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_run_harmonic_oscillator(dimension):

    m = 1
    omega = 1
    alpha = m * omega / H_BAR

    # For each dimension, randomly pick a degree
    degree = [ numpy.random.randint(0,4) for d in range(dimension)]
    ho_w = HarmonicOscillatorWavefunction(dimension, degree, alpha=alpha)


    delta = 0.5

    sampler = CartesianSampler(dimension, delta=delta, mins=-10, maxes=10)

    x = sampler.sample()

    ho = HarmonicOscillator(dimension, m, omega)

    measured_energy = ho.energy(ho_w, x, delta)

    # The energy should be hbar * omega *(sum(degree) + 0.5*dimension)

    assert abs(measured_energy - H_BAR*omega * (numpy.sum(degree) + 0.5*dimension)) < 0.1
