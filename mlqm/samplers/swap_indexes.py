import tensorflow as tf
import numpy

def swap_random_indexes(input_tensor):  
    '''
    Pick two indexes, per row, and swap the values
    '''

    # First thing to do is generate a set of pairs of indexes, for every row.

    first_index = tf.constant(tf.range(input_tensor.shape[0]))

    # First, select indexes:
    swap_indexes = tf.math.top_k(tf.random.uniform(shape=input_tensor.shape), 2, sorted=False).indices


    first_swap_indexes  = tf.stack([first_index, swap_indexes[:,0]], axis=-1)
    second_swap_indexes = tf.stack([first_index, swap_indexes[:,1]], axis=-1)

    # Gather the values:
    first_index_value  = tf.gather_nd(input_tensor, first_swap_indexes)
    second_index_value = tf.gather_nd(input_tensor, second_swap_indexes)


    # Now, have to _set_ the new indexes
    swapped_tensor = tf.tensor_scatter_nd_update(input_tensor, first_swap_indexes, second_index_value)
    swapped_tensor = tf.tensor_scatter_nd_update(swapped_tensor, second_swap_indexes, first_index_value)
    
    return swapped_tensor



if __name__ == "__main__":
    n_walkers   = 5
    n_particles = 2
    z_spin      = 1
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


    a = tf.Variable(init_walkers)

    print(a)

    a = swap_random_indexes(a)
    print(a)