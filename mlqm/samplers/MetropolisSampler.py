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
        init_params : iter ,
        dtype       = tf.float64):
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

        self.dtype = dtype

        #  Run the initalize to get the first locations:
        self.walkers = initializer(shape=self.size, **init_params, dtype=dtype)
        self.walker_history = []

    def sample(self):
        '''Just return the current locations

        '''
        # Make sure to wrap in tf.Variable for back prop calculations
        # Before returning, append the current walkers to the walker history:
        
        self.walker_history.append(self.walkers)
        return  self.walkers

    def get_all_walkers(self):
        return self.walker_history

    def reset_history(self):
        self.walker_history = []

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
        walkers, acceptance = self.internal_kicker(
            self.size, self.walkers, wavefunction, kicker, kicker_params, tf.constant(nkicks), dtype=self.dtype)

        # Update the walkers:
        self.walkers = walkers

        # Send back the acceptance:
        return acceptance

    @tf.function(experimental_compile=False)
    # @profile
    def internal_kicker(self,
        shape,
        walkers,
        wavefunction : tf.keras.models.Model,
        kicker : callable,
        kicker_params : iter,
        nkicks : tf.constant,
        dtype):
        """Sample points in N-d Space

        By default, samples points uniformly across all dimensions.
        Returns a torch tensor on the chosen device with gradients enabled.

        Keyword Arguments:
            kicker {callable} -- Function to call to create a kick for each walker
            kicker_params {iter} -- Parameters to pass to the kicker, unrolled automatically
        """

        # Drop the model to reduced precision for this:
        # params = wavefunction.parameters()
        # print(params)

        # reduced_wf = tf.cast(wavefunction, dtype=self.dtype)
        # wavefunction.cast(self.dtype)


        # We need to compute the wave function twice:
        # Once for the original coordiate, and again for the kicked coordinates
        acceptance = tf.convert_to_tensor(0.0, dtype=dtype)
        # Calculate the current wavefunction value:
        current_wavefunction = wavefunction(walkers)

        # Generate a long set of random number from which we will pull:
        random_numbers = tf.math.log(tf.random.uniform(shape = [nkicks,shape[0],1], dtype=dtype))

        # Generate a long list of kicks:
        # print(shape)
        kicks = kicker(shape=[nkicks, *shape], **kicker_params, dtype=dtype)
        # print(kicks.shape)

        # Adding spin:
        #  A meaningful metropolis move is to pick a pair and exchange the spin
        #  ONly one pair gets swapped at a time
        #  Change the isospin of a pair as well.
        #  The spin coordinate is 2 dimensions per particle: spin and isospin (each up/down)
        #  

        # Computing modulus square of wavefunction in new vs old coordinates
        #  - this kicks randomly with a guassian, and has an acceptance probaility
        # However, what we can do instead is to add a drift term
        # Instead of kicking with a random gaussian, we compute the derivative 
        # with respect to X.
        # Multiply it by sigma^2
        # Then, clip the drift so it is not too large.
        # New coordinates are the old + gaussian + drift
        # Acceptance is ratio of modulus squared wavefunction IF the move is symmetric
        # So need to weight the modulus with a drift reweighting term.


        # Spin typically thermalizes first.
        # Fewer spin configurations allowed due to total spin conservation
        # 

        for i_kick in tf.range(nkicks):
            # Create a kick:
            kick = kicks[i_kick]
            # kick = kicker(shape=shape, **kicker_params, dtype=dtype)
            kicked = walkers + kick

            # Compute the values of the wave function, which should be of shape
            # [nwalkers, 1]
            kicked_wavefunction   = wavefunction(kicked)


            # Probability is the ratio of kicked **2 to original
            probability = 2 * (kicked_wavefunction - current_wavefunction)
            # Acceptance is whether the probability for that walker is greater than
            # a random number between [0, 1).
            # Pull the random numbers and create a boolean array
            # accept      = probability >  tf.random.uniform(shape=[shape[0],1])
            accept      = probability >  random_numbers[i_kick]
            # accept      = probability >  tf.math.log(tf.random.uniform(shape=[shape[0],1]))

            # Grab the kicked wavefunction in the places it is new, to speed up metropolis:
            current_wavefunction = tf.where(accept, kicked_wavefunction, current_wavefunction)

            # We need to broadcast accept to match the right shape
            # Needs to come out to the shape [nwalkers, nparticles, ndim]
            accept = tf.tile(accept, [1,tf.reduce_prod(shape[1:])])
            accept = tf.reshape(accept, shape)
            walkers = tf.where(accept, kicked, walkers)

            acceptance = tf.reduce_mean(tf.cast(accept, dtype))

        return walkers, acceptance
