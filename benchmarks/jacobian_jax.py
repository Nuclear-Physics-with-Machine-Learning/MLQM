import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import numpy
import tensorflow as tf

import jax.numpy as np
from jax import grad, jit, vmap, jacfwd

import time

import numpy
numpy.random.seed(0)


def model(x, layer1, layer2):

    x = np.dot(x, layer1)
    return np.dot(x, layer2)



def main(ninput, n_hidden_params, n_jacobian_calculations):


    cross_check_parameters = {}
    
    # Create an input vector:
    input_vector = numpy.random.random([ninput, 1])


    cross_check_parameters['input_sum'] = numpy.sum(input_vector)
    cross_check_parameters['input_std'] = numpy.std(input_vector)

    layer_1_weights = numpy.random.random([1,n_hidden_params])
    layer_2_weights = numpy.random.random([n_hidden_params,1])


    # Create the model:
    M = jit(lambda x : model(x, layer_1_weights, layer_2_weights))

    # Forward pass:
    output = M(input_vector)
    
    # Capture the number of parameters:
    nparameters = numpy.sum([ numpy.prod(p.shape) for p in [layer_1_weights, layer_2_weights] ]) 
    cross_check_parameters['n_params'] = nparameters

    # Capture the network output:
    cross_check_parameters['output_sum'] = numpy.sum(output)
    cross_check_parameters['output_std'] = numpy.std(output)

    start = time.time()

    for i in range(n_jacobian_calculations):

        l1 = lambda weights : model(input_vector, *weights)
        jacobian = jacfwd(l1)([layer_1_weights, layer_2_weights])
     
        param_shapes = [ j.shape[1:] for j in jacobian ]
        flat_shape = [[-1, numpy.prod(js)] for js in param_shapes]

        flattened_jacobian = [numpy.reshape(j, f) for j, f in zip(jacobian, flat_shape)]
        jacobian = numpy.concatenate(flattened_jacobian, axis=-1)

 
    end = time.time()
    cross_check_parameters['jacobian_sum']  = numpy.sum(jacobian)
    cross_check_parameters['jacobian_std']  = numpy.std(jacobian)
    cross_check_parameters['jacobian_prod'] = numpy.prod(jacobian)
    cross_check_parameters['jacobian_time'] = (end - start) / i
    cross_check_parameters['jacobian_n_calls'] = n_jacobian_calculations

    return cross_check_parameters

if __name__ == '__main__':
    ccp = main(512, 128, 5)
    print(ccp)


