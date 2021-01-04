import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import numpy
import tensorflow as tf

numpy.random.seed(0)

class model(tf.keras.models.Model):

    def __init__(self, layer_weights):
        tf.keras.models.Model.__init__(self)

        self._layers = []
        for i in range(len(layer_weights)):
            init = tf.constant_initializer(layer_weights[i])
            self._layers.append(
                tf.keras.layers.Dense(layer_weights[i].shape[-1],
                    kernel_initializer=init, use_bias=False))


    @tf.function
    def call(self, x):

        for i in range(len(self._layers)):
            x = self._layers[i](x)
        return x

    def count_parameters(self):
        return numpy.sum([ numpy.prod(p.shape) for p in self.trainable_variables ] )

@tf.function
def compute_jacobians(input_vector, M):
    with tf.GradientTape() as tape:
        output = M(input_vector)

    jacobian = tape.jacobian(output, M.trainable_variables, parallel_iterations=None)

    param_shapes = [ j.shape[1:] for j in jacobian ]
    flat_shape = [[-1, tf.reduce_prod(js)] for js in param_shapes]

    flattened_jacobian = [tf.reshape(j, f) for j, f in zip(jacobian, flat_shape)]
    jacobian = tf.concat(flattened_jacobian, axis=-1)
    return jacobian

def main(ninput, n_filters_list, n_jacobian_calculations):


    cross_check_parameters = {}

    # Create an input vector:
    input_vector = numpy.random.random([ninput,1])
    cross_check_parameters['input_sum'] = numpy.sum(input_vector)
    cross_check_parameters['input_std'] = numpy.std(input_vector)

    # Make sure to start with 1 input and finish with 1 output
    n_filters_list.insert(0,1)
    n_filters_list.append(1)

    # Use these as the layer weights:
    layer_weights = [ numpy.random.random([n_filters_list[i],n_filters_list[i+1]]) for i in range(len(n_filters_list)-1)]


    # Create the model, and input Tensor:
    input_vector = tf.convert_to_tensor(input_vector, dtype=tf.float32)
    M = model(layer_weights)

    # Forward pass of the model
    output = M(input_vector)

    # Capture the number of parameters:
    cross_check_parameters['n_params'] = M.count_parameters()

    # Capture the network output:
    cross_check_parameters['output_sum'] = numpy.sum(output.numpy())
    cross_check_parameters['output_std'] = numpy.std(output.numpy())

    start = time.time()
    cross_check_parameters['jacobian_times'] = []
    for i in range(n_jacobian_calculations):
        this_start = time.time()
        jacobian = compute_jacobians(input_vector, M)
        this_end = time.time()
        cross_check_parameters['jacobian_times'].append((this_end - this_start))


    end = time.time()
    # Store some jacobian information:
    cross_check_parameters['n_filters_list'] = n_filters_list
    cross_check_parameters['jacobian_sum']  = numpy.sum(jacobian.numpy())
    cross_check_parameters['jacobian_std']  = numpy.std(jacobian.numpy())
    cross_check_parameters['jacobian_prod'] = numpy.prod(jacobian.numpy())
    cross_check_parameters['jacobian_time'] = (end - start)
    cross_check_parameters['jacobian_n_calls'] = n_jacobian_calculations

    return cross_check_parameters

if __name__ == '__main__':
    ninput = 24
    network_list = [
        [32, 32, 16],
        [128, 128],
        [512, 512, 512],
        [16, 16, 16, 16, 16, 16],
        [2048],
    ]
    for network in network_list:
        ccp = main(ninput, network, 5)
        print(ccp)
