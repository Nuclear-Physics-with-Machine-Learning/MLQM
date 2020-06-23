import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import numpy
import tensorflow as tf

numpy.random.seed(0)

class model(tf.keras.models.Model):

    def __init__(self, n_hidden_params, layer_1_weights, layer_2_weights):
        tf.keras.models.Model.__init__(self)


        init1 = tf.constant_initializer(layer_1_weights)
        self.layer1 = tf.keras.layers.Dense(n_hidden_params, kernel_initializer=init1, use_bias=False)
        init2 = tf.constant_initializer(layer_2_weights)
        self.layer2 = tf.keras.layers.Dense(1, kernel_initializer=init2, use_bias=False)
    


    def call(self, x):

        x = self.layer1(x)
        return self.layer2(x)

    def count_parameters(self):
        return numpy.sum([ numpy.prod(p.shape) for p in self.trainable_variables ] )


def main(ninput, n_hidden_params, n_jacobian_calculations):

    cross_check_parameters = {}

    # Create an input vector:
    input_vector = numpy.random.random([ninput,1])
    cross_check_parameters['input_sum'] = numpy.sum(input_vector)
    cross_check_parameters['input_std'] = numpy.std(input_vector)


    # Use these as the layer weights:
    layer_1_weights = numpy.random.random([1,n_hidden_params])
    layer_2_weights = numpy.random.random([n_hidden_params,1])


    # Create the model, and input Tensor:
    input_vector = tf.convert_to_tensor(input_vector, dtype=tf.float32)
    M = model(n_hidden_params, layer_1_weights, layer_2_weights)

    # Forward pass of the model
    output = M(input_vector)

    # Capture the number of parameters:
    cross_check_parameters['n_params'] = M.count_parameters()

    # Capture the network output:
    cross_check_parameters['output_sum'] = numpy.sum(output.numpy())
    cross_check_parameters['output_std'] = numpy.std(output.numpy())

    start = time.time()
    for i in range(n_jacobian_calculations):
        with tf.GradientTape() as tape:
            output = M(input_vector)

        jacobian = tape.jacobian(output, M.trainable_variables, parallel_iterations=None)

        param_shapes = [ j.shape[1:] for j in jacobian ]
        flat_shape = [[-1, tf.reduce_prod(js)] for js in param_shapes]

        flattened_jacobian = [tf.reshape(j, f) for j, f in zip(jacobian, flat_shape)]
        jacobian = tf.concat(flattened_jacobian, axis=-1)

    end = time.time()

    # Store some jacobian information:
    cross_check_parameters['jacobian_sum']  = numpy.sum(jacobian.numpy())
    cross_check_parameters['jacobian_std']  = numpy.std(jacobian.numpy())
    cross_check_parameters['jacobian_prod'] = numpy.prod(jacobian.numpy())
    cross_check_parameters['jacobian_time'] = (end - start) / n_jacobian_calculations
    cross_check_parameters['jacobian_n_calls'] = n_jacobian_calculations

    return cross_check_parameters

if __name__ == '__main__':
    ccp = main(512, 128, 5)
    print(ccp)
