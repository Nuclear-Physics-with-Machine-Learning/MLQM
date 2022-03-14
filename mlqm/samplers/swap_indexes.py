import time

import tensorflow as tf
import numpy

# @profile
def swap_loop_basic(n_swaps, input_tensor):
    @profile
    @tf.function
    def swap_random_indexes(input_tensor):
        '''
        Pick two indexes, per row, and swap the values
        '''

        # First thing to do is generate a set of pairs of indexes, for every row.

        first_index = tf.range(input_tensor.shape[0])

        # First, select indexes:
        rands = tf.random.uniform(shape=input_tensor.shape)
        swap_indexes = tf.math.top_k(rands, 2, sorted=False).indices
        second_swap_indexes = tf.stack([first_index, swap_indexes[:,1]], axis=-1)

        l = [first_index, swap_indexes[:,0]]
        first_swap_indexes  = tf.stack(l, axis=-1)

        # Gather the values:
        first_index_value  = tf.gather_nd(input_tensor, first_swap_indexes)
        second_index_value = tf.gather_nd(input_tensor, second_swap_indexes)


        # Now, have to _set_ the new indexes
        swapped_tensor = tf.tensor_scatter_nd_update(input_tensor, first_swap_indexes, second_index_value)
        swapped_tensor = tf.tensor_scatter_nd_update(swapped_tensor, second_swap_indexes, first_index_value)

        return swapped_tensor

    for i in range(n_swaps):
        input_tensor = swap_random_indexes(input_tensor)


# @profile
def swap_loop_opt(n_swaps, input_tensor):

    @profile
    @tf.function
    def swap_random_indexes_opt(input_tensor, first_index, swap_indexes):
        '''
        Pick two indexes, per row, and swap the values
        '''

        # First thing to do is generate a set of pairs of indexes, for every row.

        # First, select indexes:
        second_swap_indexes = tf.stack([first_index, swap_indexes[:,1]], axis=-1)

        first_swap_indexes  = tf.stack([first_index, swap_indexes[:,0]], axis=-1)

        # Gather the values:
        first_index_value  = tf.gather_nd(input_tensor, first_swap_indexes)
        second_index_value = tf.gather_nd(input_tensor, second_swap_indexes)


        # Now, have to _set_ the new indexes
        swapped_tensor = tf.tensor_scatter_nd_update(input_tensor, first_swap_indexes, second_index_value)
        swapped_tensor = tf.tensor_scatter_nd_update(swapped_tensor, second_swap_indexes, first_index_value)

        return swapped_tensor


    @tf.function
    def gen_swaps(n_swaps, shape_ref):
        rands = tf.random.uniform(shape=(n_swaps, * shape_ref.shape))
        swap_indexes = tf.math.top_k(rands, 2, sorted=False).indices
        return swap_indexes

    swap_indexes = gen_swaps(n_swaps, input_tensor)

    # print(swap_indexes)

    # Above gives fully random swaps while excluding self-swaps.
    # This gives fully random swaps while allowing self-swaps
    # swap_indexes = tf.random.uniform(
    #     shape = (n_swaps, *input_tensor.shape),
    #     minval = 0,
    #     maxval = input_tensor.shape[-1],
    #     dtype = tf.int32
    # )


    first_index = tf.range(input_tensor.shape[0])
    for i in range(n_swaps):
        input_tensor = swap_random_indexes_opt(input_tensor, first_index, swap_indexes[i])


# @profile
def swap_loop_ale(n_swaps, input_tensor, possible_swap_pairs):

    # @profile
    # @tf.function
    def swap_random_indexes_opt(input_tensor, first_index, swap_indexes_f, swap_indexes_s):
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

    # print(input_tensor.shape)

    swap_index_i = possible_swap_pairs[0]
    swap_index_j = possible_swap_pairs[1]

    # Generate k random numbers in the range (0, swap_index_i.shape[0])
    swap_indexes = tf.random.uniform(
        shape = (n_swaps, input_tensor.shape[0]),
        minval = 0,
        maxval = swap_index_i.shape[0],
        dtype = tf.int32
    )

    print(swap_indexes[0])

    #
    # # @tf.function
    # def gen_swaps(n_swaps, shape_ref):
    #
    #
    #     rands = tf.random.uniform(shape=(n_swaps, * shape_ref.shape))
    #     swap_indexes = tf.math.top_k(rands, 2, sorted=False).indices
    #     return swap_indexes

    swap_indexes_f = tf.gather(swap_index_i, swap_indexes)
    swap_indexes_s = tf.gather(swap_index_j, swap_indexes)

    # def gen_swaps(n_swaps, shape_ref):
    #
    #
    #     rands = tf.random.uniform(shape=(n_swaps, * shape_ref.shape))
    #     swap_indexes = tf.math.top_k(rands, 2, sorted=False).indices
    #     return swap_indexes
    # swap_indexes = gen_swaps(n_swaps, input_tensor)
    # print(swap_indexes.shape)

    # Above gives fully random swaps while excluding self-swaps.
    # This gives fully random swaps while allowing self-swaps
    # swap_indexes = tf.random.uniform(
    #     shape = (n_swaps, *input_tensor.shape),
    #     minval = 0,
    #     maxval = input_tensor.shape[-1],
    #     dtype = tf.int32
    # )


    first_index = tf.range(input_tensor.shape[0])
    for i in range(n_swaps):
        input_tensor = swap_random_indexes_opt(
            input_tensor, first_index,
            swap_indexes_f[i], swap_indexes_s[i])

def gen_possible_swaps(n_particles):
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

if __name__ == "__main__":



    n_walkers   = 5000
    n_particles = 2
    z_spin      = 1
    n_swaps     = 200
    size = (n_walkers, n_particles)
    # Note that we initialize with NUMPY for ease of indexing and shuffling
    init_walkers = numpy.zeros(shape=size)
    for i in range(z_spin):
        init_walkers[:,i] += 1



    # Shuffle the spin up particles on each axis:

    # How to compute many permutations at once?
    #  Answer from https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
    # Bottom line: gen random numbers for each axis, sort only in that axis,
    # and apply the permutations
    idx = numpy.random.rand(*init_walkers.shape).argsort(axis=1)
    init_walkers = numpy.take_along_axis(init_walkers, idx, axis=1)

    possible_swap_pairs = gen_possible_swaps(n_particles)


    a = tf.Variable(init_walkers)

    start = time.time()

    swap_loop_basic(1, a)
    swap_loop_opt(1, a)
    swap_loop_ale(1, a, possible_swap_pairs)

    print(f"Done first in {time.time() - start :.3f}")

    start = time.time()

    swap_loop_basic(n_swaps, a)

        # a = swap_random_indexes(a, swap_indexes[i])

    # print(a)
    print(f"Done basic in {time.time() - start:.3f}")
    start = time.time()


    swap_loop_opt(n_swaps, a)



    print(f"Done optimized in {time.time() - start:.3f}")
    start = time.time()

    swap_loop_ale(n_swaps, a, possible_swap_pairs)

    print(f"Done ale in {time.time() - start:.3f}")
    # print(a)
