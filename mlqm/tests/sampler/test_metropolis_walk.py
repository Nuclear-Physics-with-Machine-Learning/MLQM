from mlqm import models, samplers
import numpy
import math

import tensorflow as tf
import pytest

@pytest.mark.parametrize('dimension', [1])
@pytest.mark.parametrize('nwalkers', [500,])
@pytest.mark.parametrize('nparticles', [1])
@pytest.mark.parametrize('n_thermalize', [500])
def test_metropolis_walk(dimension, nwalkers, nparticles, n_thermalize):


    alpha = 0.9

    kicker = tf.random.normal
    kicker_params = {"mean": 0.0, "stddev" : 0.4}

    # Create a sampler: 
    sampler = samplers.MetropolisSampler(
        n           = dimension,
        nwalkers    = nwalkers,
        nparticles  = nparticles,
        initializer = kicker,
        init_params = kicker_params,
        dtype       = tf.float64)


    # Use a Gaussian Wavefunction:
    wavefunction = models.GaussianBoundaryCondition(n = dimension, exp=math.sqrt(alpha), trainable=False, dtype=tf.float64)

    x = sampler.sample()

    # Run void steps to thermalize the walkers to this wavefunction:
    acceptance = sampler.kick(wavefunction, kicker, kicker_params, nkicks=n_thermalize)
    x = sampler.sample()


    y = wavefunction(x)
    integral = tf.reduce_mean(y)

    # # For a gaussian wave function, it is expected the integral (average of y values)
    # # Should converge to approximately log(sqrt(pi/alpha)), where alpha is the exponential parameter.

    # # The convergence error should be approximately sqrt(n_walkers), statistical uncertainty.

    # uncert  = 1./ math.sqrt(nwalkers)                                                                                                                                                                                                 
    # print(uncert)
 
    # analytic_integral = math.log(math.pow(math.pi / (alpha), dimension * 0.5))

    # print("analytic_integral: ", analytic_integral)
    # print("integral: ", integral)

    # We ought to be able to see that the sampled positions are, in all dimensions, roughly following
    # the gaussian.

    x = x.numpy()
    sigma = math.pow(1./ (4*alpha), 0.5)

    # Here, the goal is to see if we've sampled properly.
    for i_particle in range(nparticles):
        positions = x[:,i_particle,:]
        print(positions.shape)
        for i_dim in range(dimension):
            where_up   = positions[:,i_dim] < sigma
            where_down = positions[:,i_dim] > -sigma

            all_where = numpy.logical_and(where_up, where_down)

            frac = numpy.sum(all_where.astype(numpy.float32)) / nwalkers
            print(frac)


    assert(False)