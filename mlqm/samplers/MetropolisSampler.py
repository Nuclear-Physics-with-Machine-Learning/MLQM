import tensorflow as tf
import numpy


class MetropolisSampler(object):
    """Metropolis Sampler in N dimension

    Sample from N-D coordinates, using some initial probability distribution

    Relies on functional calls to sample on the fly with flexible distributions
    """
    def __init__(self,
        n           : int,
        nwalkers    : int,
        nparticles  : int,
        initializer : callable,
        init_params : iter ):
        '''Initialize a metropolis sampler

        Create a metropolis walker with `n` walkers.  Can use normal, uniform

        Arguments:
            n {int} -- Dimension
            nwalkers {int} -- Number of unique walkers
            initializer {callable} -- Function to call to initialize each walker
            init_params {iter} -- Parameters to pass to the initializer, unrolled automatically
        '''

        # Set the dimension:
        self.n = n

        # Set the number of walkers:
        self.nwalkers = nwalkers

        # Set the number of particles:
        self.nparticles = nparticles

        self.size = (self.nwalkers, self.nparticles, self.n)

        #  Run the initalize to get the first locations:
        self.walkers = initializer(shape=self.size, **init_params)

    def sample(self):
        '''Just return the current locations

        '''
        # Make sure to wrap in tf.Variable for back prop calculations
        return  self.walkers

    def kick(self,
        wavefunction : tf.keras.models.Model,
        kicker : callable,
        kicker_params : iter,
        nkicks : int ):
        '''Wrapper for a compiled kick function via tensorflow.

        This fills in the compiled function with the size and the walkers.

        Arguments:
            wavefunction {tf.keras.models.Model} -- The wavefunction used for the metropolis walk
            kicker {callable} -- A callable function for generating kicks
            kicker_params {iter} -- Arguments to the kicker function.
        '''
        # for i in range(nkicks):
        walkers, acceptance = self.internal_kicker(self.size, self.walkers, wavefunction, kicker, kicker_params, tf.constant(nkicks))

        # Update the walkers:
        self.walkers = walkers

        # Send back the acceptance:
        return acceptance

    @tf.function
    def internal_kicker(self,
        shape,
        walkers,
        wavefunction : tf.keras.models.Model,
        kicker : callable,
        kicker_params : iter,
        nkicks : tf.constant):
        """Sample points in N-d Space

        By default, samples points uniformly across all dimensions.
        Returns a torch tensor on the chosen device with gradients enabled.

        Keyword Arguments:
            kicker {callable} -- Function to call to create a kick for each walker
            kicker_params {iter} -- Parameters to pass to the kicker, unrolled automatically
        """

        # We need to compute the wave function twice:
        # Once for the original coordiate, and again for the kicked coordinates
        acceptance = tf.convert_to_tensor(0.0)
        for i_kick in tf.range(nkicks):
            # Create a kick:
            kick = kicker(shape=shape, **kicker_params)

            # Compute the values of the wave function, which should be of shape
            # [nwalkers, 1]
            original = wavefunction(walkers)
            kicked   = wavefunction(walkers + kick)


            # Probability is the ratio of kicked **2 to original
            probability = 2 * (kicked - original)
            # Acceptance is whether the probability for that walker is greater than
            # a random number between [0, 1).
            # Pull the random numbers and create a boolean array
            accept      = probability >  tf.math.log(tf.random.uniform(shape=[shape[0],1]))

            # We need to broadcast accept to match the right shape
            # Needs to come out to the shape [nwalkers, nparticles, ndim]
            accept = tf.tile(accept, [1,tf.reduce_prod(shape[1:])])
            accept = tf.reshape(accept, shape)
            walkers = tf.where(accept, walkers + kick, walkers)

            acceptance = tf.reduce_mean(tf.cast(accept, tf.float32))

        return walkers, acceptance
