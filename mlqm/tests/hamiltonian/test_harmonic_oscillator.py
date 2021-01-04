from mlqm import models, samplers, hamiltonians

import tensorflow as tf
import numpy
import pytest

@pytest.mark.parametrize('dimension', [1,2,3])
@pytest.mark.parametrize('nwalkers', [25,50])
@pytest.mark.parametrize('degree', [0,1,2])
@pytest.mark.parametrize('n_thermalize', [100])
def test_metropolis_walk(dimension, nwalkers, degree, n_thermalize, dtype=tf.float64):

    mass = 1.0
    omega = 1.0

    kicker = tf.random.normal
    kicker_params = {"mean": 0.0, "stddev" : 0.4}

    # Create a sampler: 
    sampler = samplers.MetropolisSampler(
        n           = dimension,
        nwalkers    = nwalkers,
        nparticles  = 1,
        initializer = kicker,
        init_params = kicker_params,
        dtype       = dtype)

    x = sampler.sample()


    # Use a Gaussian Wavefunction:
    wavefunction = models.HarmonicOscillatorWavefunction(
        n = dimension, 
        nparticles=1, 
        degree = degree, 
        alpha = omega, 
        dtype=dtype)

    # Initialize the wavefunction:
    wavefunction.call(x)



    # Run void steps to thermalize the walkers to this wavefunction:
    acceptance = sampler.kick(wavefunction, kicker, kicker_params, nkicks=n_thermalize)

    # Get the new parameters
    x = sampler.sample()



    hamiltonian = hamiltonians.HarmonicOscillator(mass=tf.constant(mass), omega=tf.constant(omega))


    energy, energy_jf, ke_jf, ke_direct, pe = hamiltonian.energy(wavefunction, x)

    mean_energy = tf.reduce_mean(energy)
    theory_energy = ( omega) * (degree + 0.5) * (dimension)
    assert(numpy.abs(mean_energy - theory_energy ) < 0.01) 

    # # Here, the goal is to see if we've sampled properly.
    # for i_particle in range(nparticles):
    #     positions = x[:,i_particle,:]
    #     print(positions.shape)
    #     for i_dim in range(dimension):
    #         print(tf.reduce_mean(positions[:,i_dim]))

def main():
    dimension    =  2
    nwalkers     =  25
    degree       =  0
    n_thermalize =  100
    test_metropolis_walk(dimension, nwalkers, degree, n_thermalize, dtype=tf.float64)


if __name__ == '__main__':
    main()
