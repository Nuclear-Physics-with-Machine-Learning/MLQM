from jax import random
import jax.numpy as numpy


import time
import sys, os
import pathlib
from dataclasses import dataclass, field
import pytest

# Add the mlqm path:
current_dir = pathlib.Path(__file__).parent.resolve()
init = pathlib.Path("__init__.py")

while current_dir != current_dir.root:

    if (current_dir / init).is_file():
        current_dir = current_dir.parent
    else:
        break

# Here, we break and add the directory to the path:
sys.path.insert(0,str(current_dir))

from mlqm.models import initialize_wavefunction
from mlqm.samplers import MetropolisSampler, kick

from mlqm.config import ManyBodyCfg, Sampler
from omegaconf import OmegaConf


from jax.config import config; config.update("jax_enable_x64", True)


@pytest.mark.parametrize('seed', [0, time.time()])
@pytest.mark.parametrize('nwalkers', [10])
@pytest.mark.parametrize('nparticles', [2,3,4,5])
@pytest.mark.parametrize('ndim', [1,2,3])
@pytest.mark.parametrize('n_spin_up', [1,2])
@pytest.mark.parametrize('n_protons', [1,2])
def test_hamiltonian(seed, nwalkers, nparticles, ndim, n_spin_up, n_protons):

    # Create the sampler config:
    sampler_config = Sampler()

    sampler_config.n_walkers_per_observation = nwalkers
    sampler_config.n_concurrent_obs_per_rank = 1
    sampler_config.n_particles  = nparticles
    sampler_config.n_dim = ndim
    sampler_config.n_spin_up = n_spin_up
    sampler_config.n_protons = n_protons


    # Initialize the sampler:
    key = random.PRNGKey(int(seed))
    key, subkey = random.split(key)

    sampler = MetropolisSampler(
        sampler_config,
        subkey,
        "float64"
        )
    x, spin, isospin = sampler.sample()


    # Create the wavefunction:
    key, subkey = random.split(key)

    c = ManyBodyCfg
    c.mean_subtract=False
    c = OmegaConf.structured(c)

    wavefunction, parameters = initialize_wavefunction(
        x, spin, isospin, subkey, sampler_config, c)

    w_local = wavefunction.apply_walkers(parameters, x, spin, isospin)


    from mlqm.hamiltonians import compute_derivatives, compute_derivatives_single
    w_of_x, dw_dx, d2w_dx2 = compute_derivatives_single(wavefunction, parameters, x[0], spin[0], isospin[0])

    w_of_x, dw_dx, d2w_dx2 = compute_derivatives(wavefunction, parameters, x, spin, isospin)

    # Make sure we've got the same values of the wavefunction:
    assert (w_local - w_of_x < 1e-8).all()

    # We proceed to check the derivatives with finte difference methods.

    # Need a "difference" term:
    kick = numpy.zeros(shape = x.shape)
    kick_size = 1e-4

    # First and second order derivative.  Check it for each dimension.
    for i_dim in range(ndim):

        # select a random particle to kick:
        key, subkey = random.split(key)
        i_kicked_particle = random.choice(subkey, nparticles, shape=(nwalkers,))
        # i_kicked_particle = 0
        this_kick = numpy.copy(kick)
        # Have to create an index for walkers to slice:
        walkers = numpy.arange(nwalkers)
        # Only applying to particle 0
        this_kick = this_kick.at[walkers,i_kicked_particle,i_dim].add(kick_size)
        kicked_up_input = x + this_kick

        kicked_double_up_input = x + 2*this_kick
        # # Mean subtract:
        # up_x = kicked_up_input - \
        #     numpy.reshape(numpy.mean(kicked_up_input, axis=1), (nwalkers, 1, ndim))

        kicked_down_input = x - this_kick

        kicked_double_down_input = x - 2*this_kick
        # down_x = kicked_down_input - \
        #     numpy.reshape(numpy.mean(kicked_down_input, axis=1), (nwalkers, 1, ndim))


        # In this case, *because* there is a mean subtraction,
        # we will calculate a derivate for the first particle only.
        # The derivatives for the other particles based on this kick will be
        # flipped sign.

        # The total kick will actually be kick_size / nparticles, because of
        # the effect of mean subtraction

        # Differences:
        w_up = wavefunction.apply_walkers(parameters, kicked_up_input, spin, isospin)
        w_down = wavefunction.apply_walkers(parameters, kicked_down_input, spin, isospin)
        w_up_up = wavefunction.apply_walkers(parameters, kicked_double_up_input, spin, isospin)
        w_down_down = wavefunction.apply_walkers(parameters, kicked_double_down_input, spin, isospin)



        # Use numpy to make slicing easier
        w_prime_fd = (w_up - w_down) / (2*kick_size)
        print(w_prime_fd.shape)
        # What about the second derivative?

        # https://math.stackexchange.com/questions/3756717/finite-differences-second-derivative-as-successive-application-of-the-first-deri
        # This gives precision of O(kick**4)
        w_prime_prime_num = -w_down_down + 16*w_down - 30* w_of_x + 16 * w_up - w_up_up
        w_prime_prime_fd = w_prime_prime_num/ (12 * kick_size**2)



        # Now, verify everything is correct.
        # print("dw_dx: ", dw_dx)
         # slice to just the dimension we're moving, all walkers
        target = dw_dx[walkers,i_kicked_particle,i_dim]
        second_target = d2w_dx2[walkers,i_kicked_particle, i_dim]

        # print("target: ", target)
        # print("w_prime_fd: ", w_prime_fd)
        # The tolerance on the second order first derivative is net_kick**2
        # print("First difference: ", (w_prime_fd - target) )
        # print("Target tolerance: ", kick_size**2)
        assert( numpy.abs(w_prime_fd - target) < kick_size **2).all()

        print("second_target: ", second_target)
        print("w_prime_prime_fd: ", w_prime_prime_fd)
        print("2nd difference: ", w_prime_prime_fd - second_target)
        assert (numpy.abs(w_prime_prime_fd - second_target) < kick_size ).all()

@pytest.mark.parametrize('nwalkers', [10])
@pytest.mark.parametrize('nparticles', [2,3,4,5])
@pytest.mark.parametrize('ndim', [1,2,3])
@pytest.mark.parametrize('spin', [True])
@pytest.mark.parametrize('iso_spin', [True])
def test_energies(nwalkers, nparticles, ndim, spin, iso_spin):

    n_spin_up = 2
    n_protons = 1
    c = ManyBodyCfg()

    c = OmegaConf.structured(c)
    w = ManyBodyWavefunction(ndim, nparticles, c,
        n_spin_up = n_spin_up, n_protons = n_protons,
        use_spin = spin, use_isospin = iso_spin
    )



    inputs, spins, isospins = generate_inputs(nwalkers, nparticles, ndim, n_spin_up, n_protons)

    hc = OmegaConf.structured(H_config())

    h = NuclearPotential(hc)

    inputs = tf.convert_to_tensor(inputs)
    spins = tf.convert_to_tensor(spins)
    isospins = tf.convert_to_tensor(isospins)

    start = time.time()
    energy, energy_jf, ke_jf, ke_direct, pe, w_of_x = h.energy(w, inputs, spins, isospins)

    print(f"First run: {time.time() - start:.3f}")

    start = time.time()
    energy, energy_jf, ke_jf, ke_direct, pe, w_of_x = h.energy(w, inputs, spins, isospins)

    print(f"Second run: {time.time() - start:.3f}")

    print(ke_direct)

    assert (ke_direct >= 0).numpy().all()
    # assert (pe <= 0).numpy().all()

if __name__ == "__main__":
    test_hamiltonian(100,3,2,3,1,1)
    # test_energies(10,2,3,True, True)
