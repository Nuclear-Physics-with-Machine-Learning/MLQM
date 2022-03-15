import pathlib
import time, sys

import tensorflow as tf
import numpy

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


from mlqm.samplers import MetropolisSampler

n_walkers  = 5
n_dim      = 3
nparticles = 3
nkicks     = 2

sampler = MetropolisSampler(
    n           = n_dim,
    nwalkers    = n_walkers,
    nparticles  = nparticles,
    initializer = tf.random.normal,
    init_params = {"mean": 0.0, "stddev" : 0.2},
    n_spin_up   = 2,
    n_protons   = 1,
    use_spin    = True,
    use_isospin = True,
    dtype       = tf.float64
)


class TrivialWavefunction:

    def __init__(self, sigma = 1.0, mu=2.0):
        self.sigma = tf.constant(sigma, dtype=tf.float64)
        self.mu    = tf.constant(mu, dtype=tf.float64)

    def __call__(self, inputs, spin=None, isospins=None):

        # Take the inputs and square them, centered on mu:
        sq = tf.pow(inputs - tf.reshape(self.mu, (-1,1,1 )), 2)
        # Reduce over all dimensions that aren't the primary one:

        reduced = tf.reduce_sum(sq, axis=(1,2))

        return tf.reshape(tf.exp(-0.5*reduced / self.sigma), (-1,1))


wavefunction = TrivialWavefunction()

kicker = tf.random.normal
kick_args = {"mean": 0.0, "stddev" : 0.6}

x, spin, isospin = sampler.sample()

start_values = wavefunction(x, spin, isospin)

print("Mean x: ", tf.reduce_mean(x))
print("Var x: ", tf.math.reduce_std(x))

acceptance = sampler.kick(wavefunction, kicker, kick_args, nkicks=1)
start = time.time()

acceptance = sampler.kick(wavefunction, kicker, kick_args, nkicks=nkicks)
print(f"time for {nkicks} kicks: ", time.time() - start)

print("Acceptance: ", acceptance)

x, spin, isospin = sampler.sample()
print("End Mean x: ", tf.reduce_mean(x))
print("End Var x: ", tf.math.reduce_std(x))
