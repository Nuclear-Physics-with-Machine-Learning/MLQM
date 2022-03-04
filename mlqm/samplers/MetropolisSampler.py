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
        init_params : iter,
        n_spin_up   : int = 0,
        n_protons   : int = 0,
        dtype       : tf.dtypes = tf.float64):
        '''Initialize a metropolis sampler

        Create a metropolis walker with `n` walkers.  Can use normal, uniform

        Arguments:
            n {int} -- Dimension
            nwalkers {int} -- Number of unique walkers
            nparticles {int} -- Total number of particles per walker
            initializer {callable} -- Function to call to initialize each walker
            init_params {iter} -- Parameters to pass to the initializer, unrolled automatically
            dtype {tf.dtypes} -- data format type for the walkers
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

        self.use_spin = False
        self.use_isospin = False

        self.possible_swap_pairs = self.gen_possible_swaps(self.nparticles)


        #  Initialize the spin if needed:
        if n_spin_up != 0:
            self.use_spin = True
            self.spin_size = (self.nwalkers, self.nparticles)


            #  Run the initalize to get the first locations:
            #  The initializer sets a random number of particles in each walker
            #  to the spin up state in order to create a total z sping as specified.

            # Note that we initialize with NUMPY for ease of indexing and shuffling
            init_walkers = numpy.zeros(shape=self.spin_size) - 1
            for i in range(n_spin_up):
                init_walkers[:,i] += 2

            # Shuffle the spin up particles on each axis:

            # How to compute many permutations at once?
            #  Answer from https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
            # Bottom line: gen random numbers for each axis, sort only in that axis,
            # and apply the permutations
            idx = numpy.random.rand(*init_walkers.shape).argsort(axis=1)
            init_walkers = numpy.take_along_axis(init_walkers, idx, axis=1)

            self.spin_walkers = tf.Variable(init_walkers)
            self.spin_walker_history = []

        #  Initialize the spin if needed:
        if n_protons != 0:
            self.use_isospin = True
            self.isospin_size = (self.nwalkers, self.nparticles)


            #  Run the initalize to get the first locations:
            #  The initializer sets a random number of particles in each walker
            #  to the spin up state in order to create a total z sping as specified.

            # Note that we initialize with NUMPY for ease of indexing and shuffling
            init_walkers = numpy.zeros(shape=self.isospin_size) -1
            for i in range(n_protons):
                init_walkers[:,i] += 2

            # Shuffle the spin up particles on each axis:

            # How to compute many permutations at once?
            #  Answer from https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
            # Bottom line: gen random numbers for each axis, sort only in that axis,
            # and apply the permutations
            idx = numpy.random.rand(*init_walkers.shape).argsort(axis=1)
            init_walkers = numpy.take_along_axis(init_walkers, idx, axis=1)

            self.isospin_walkers = tf.Variable(init_walkers)
            self.isospin_walker_history = []


    def gen_possible_swaps(self, n_particles):
        '''
            # Create a list of all possible swaps.
            # With n particles, there are n_particles * (n_particles - 1) / 2 possible
            # swaps.  We can generate them all at once.
            # Note that swapping particles i and j is equal to swapping particles j and i.


            # Say, n = 4
            # 0 -> 1
            # 0 -> 2
            # 0 -> 3
            # 1 -> 2
            # 1 -> 3
            # 2 -> 3

        '''
        swap_i = []
        swap_j = []
        max_index = n_particles
        i = 0
        while i < n_particles:
            for j in range(i + 1, max_index):
                swap_i.append(i)
                swap_j.append(j)
            i += 1

        return tf.convert_to_tensor(swap_i), tf.convert_to_tensor(swap_j)


    def sample(self):
        '''Just return the current locations

        '''
        # Make sure to wrap in tf.Variable for back prop calculations
        # Before returning, append the current walkers to the walker history:

        self.walker_history.append(self.walkers)
        self.spin_walker_history.append(self.spin_walkers)
        self.isospin_walker_history.append(self.isospin_walkers)
        return  self.walkers, self.spin_walkers, self.isospin_walkers

    def get_all_walkers(self):
        return self.walker_history, self.spin_walker_history, self.isospin_walker_history

    def reset_history(self):
        self.walker_history         = []
        self.spin_walker_history    = []
        self.isospin_walker_history = []

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
        walkers, spin_walkers, isospin_walkers, acceptance = self.internal_kicker(
            self.size, self.walkers, self.spin_walkers, self.isospin_walkers,
            wavefunction,
            kicker, kicker_params, tf.constant(nkicks), dtype=self.dtype)

        # Update the walkers:
        self.walkers = walkers
        self.spin_walkers = spin_walkers
        self.isospin_walkers = isospin_walkers
        # Send back the acceptance:
        return acceptance

    # @tf.function(experimental_compile=False)
    # @profile
    @tf.function
    def internal_kicker(self,
        shape,
        walkers,
        spin_walkers,
        isospin_walkers,
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
        current_wavefunction = wavefunction(
            walkers, spin_walkers, isospin_walkers)

        # Generate a long set of random number from which we will pull:
        random_numbers = tf.random.uniform(shape = [nkicks,shape[0],1], dtype=dtype)
        # random_numbers = tf.math.log(
        #     tf.random.uniform(shape = [nkicks,shape[0],1], dtype=dtype))

        # Generate a long list of kicks:
        kicks = kicker(shape=[nkicks, *shape], **kicker_params, dtype=dtype)


        def generate_swap_first_and_second(nkicks,nwalkers,swap_index_i, swap_index_j):


            # Generate k random numbers in the range (0, swap_index_i.shape[0])
            spin_swap_indexes = tf.random.uniform(
                shape = (nkicks, shape[0]),
                minval = 0,
                maxval = swap_index_i.shape[0],
                dtype = tf.int32
            )

            spin_swap_indexes_f = tf.gather(swap_index_i, spin_swap_indexes)
            spin_swap_indexes_s = tf.gather(swap_index_j, spin_swap_indexes)
            return spin_swap_indexes_f, spin_swap_indexes_s

        spin_swap_indexes_f, spin_swap_indexes_s = \
            generate_swap_first_and_second(nkicks, shape[0], *self.possible_swap_pairs)

        isospin_swap_indexes_f, isospin_swap_indexes_s = \
            generate_swap_first_and_second(nkicks, shape[0], *self.possible_swap_pairs)

        first_index = tf.range(shape[0])
        #

        # Adding spin:
        #  A meaningful metropolis move is to pick a pair and exchange the spin
        #  ONly one pair gets swapped at a time
        #  Change the isospin of a pair as well.
        #  The spin coordinate is 2 dimensions per particle: spin and isospin (each up/down)
        #

        # Computing modulus squa f wavefunction in new vs old coordinates
        #  - this kicks randomly with a guassian, and has an acceptance probaility
        # However, what we can do instead is to add a drift term
        # Instead of kicking with a random gaussian, we compute the derivative
        # with respect to X.
        # Multiply it by sigma^2
        # Then, clip the drift so it is not too large.
        # New coordinates are the old + gaussian + drift
        # Acceptance is ratio of modulus sq d wavefunction IF the move is symmetric
        # So need to weight the modulus with a drift reweighting term.


        # Spin typically thermalizes first.
        # Fewer spin configurations allowed due to total spin conservation
        #

        for i_kick in tf.range(nkicks):

            # Get kicked spin coordinates
            kicked_spins = self.swap_random_indexes_opt(
                spin_walkers, first_index,
                spin_swap_indexes_f[i_kick], spin_swap_indexes_s[i_kick])
            kicked_isospins = self.swap_random_indexes_opt(
                isospin_walkers, first_index,
                isospin_swap_indexes_f[i_kick], isospin_swap_indexes_s[i_kick])
            # kicked_spins = spin_walkers
            # kicked_isospins = isospin_walkers


            # Create a kick:
            kick = kicks[i_kick]
            # kick = kicker(shape=shape, **kicker_params, dtype=dtype)
            kicked = walkers + kick

            # Compute the values of the wave function, which should be of shape
            # [nwalkers, 1]
            kicked_wavefunction = wavefunction(kicked,
                kicked_spins, kicked_isospins)


            # Probability is the ratio of kicked **2 to original
            probability = tf.math.pow(kicked_wavefunction, 2) / tf.math.pow(current_wavefunction, 2)
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
            spatial_accept = tf.tile(accept, [1,tf.reduce_prod(shape[1:])])
            spatial_accept = tf.reshape(spatial_accept, shape)
            walkers = tf.where(spatial_accept, kicked, walkers)


            spin_accept = tf.tile(accept, [1,tf.reduce_prod(shape[1:-1])])
            spin_walkers = tf.where(spin_accept,kicked_spins,spin_walkers )
            isospin_walkers = tf.where(spin_accept,kicked_isospins,isospin_walkers )


            acceptance = tf.reduce_mean(tf.cast(accept, dtype))

        return walkers, spin_walkers, isospin_walkers, acceptance

    # @profile
    @tf.function
    def swap_random_indexes_opt(self, input_tensor, first_index, swap_indexes_f, swap_indexes_s):
        '''
        Pick two indexes, per row, and swap the values
        '''

        # First thing to do is generate a set of pairs of indexes, for every row.

        # First, select indexes:
        second_swap_indexes = tf.stack([first_index, swap_indexes_f], axis=-1)
        first_swap_indexes  = tf.stack([first_index, swap_indexes_s], axis=-1)

        # Gather the values:
        first_index_value  = tf.gather_nd(input_tensor, first_swap_indexes)
        second_index_value = tf.gather_nd(input_tensor, second_swap_indexes)


        # Now, have to _set_ the new indexes
        swapped_tensor = tf.tensor_scatter_nd_update(input_tensor, first_swap_indexes, second_index_value)
        swapped_tensor = tf.tensor_scatter_nd_update(swapped_tensor, second_swap_indexes, first_index_value)

        return swapped_tensor
