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
        self.walkers = tf.Variable(initializer(shape=self.size, **init_params), trainable=True)

    def sample(self):
        '''Just return the current locations
        
        '''
        # Make sure to wrap in tf.Variable for back prop calculations
        return tf.Variable(lambda : self.walkers, trainable = True)

    def kick(self, 
        wavefunction : tf.keras.models.Model, 
        kicker : callable, 
        kicker_params : iter):
        """Sample points in N-d Space
        
        By default, samples points uniformly across all dimensions.
        Returns a torch tensor on the chosen device with gradients enabled.
        
        Keyword Arguments:
            kicker {callable} -- Function to call to create a kick for each walker
            kicker_params {iter} -- Parameters to pass to the kicker, unrolled automatically
        """

        # We need to compute the wave function twice:
        # Once for the original coordiate, and again for the kicked coordinates

        # Create a kick:
        kick = kicker(shape=self.size, **kicker_params)


        # Compute the values of the wave function, which should be of shape
        # [nwalkers, 1]
        original = wavefunction(self.walkers)
        kicked   = wavefunction(self.walkers + kick)

        # Probability is the ratio of kicked **2 to original
        probability = tf.abs(kicked)**2 / tf.abs(original)**2

        # Acceptance is whether the probability for that walker is greater than
        # a random number between [0, 1).
        # Pull the random numbers and create a boolean array
        accept      = probability >  tf.random.uniform(shape=[self.nwalkers,1]) 


        self.walkers = tf.where(accept, self.walkers + kick, self.walkers)
        
        self.acceptance = tf.reduce_mean(tf.cast(accept, tf.float32))

        return self.acceptance





