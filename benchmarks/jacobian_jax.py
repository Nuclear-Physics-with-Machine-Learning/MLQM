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

@jit
def model(x, layers):

    for i in range(len(layers)):
        x = np.dot(x, layers[i])
    return x

@jit
def compute_jacobian(input_vector, layer_weights):
        l1 = lambda weights : model(input_vector, weights)
        jacobian = jacfwd(l1)(layer_weights)
     
        jacobian = flatten_jacobian(jacobian)

        return jacobian

@jit
def flatten_jacobian(jacobian):
        param_shapes = [ j.shape[1:] for j in jacobian ]
        flat_shape = [[-1, numpy.prod(js)] for js in param_shapes]

        flattened_jacobian = [np.reshape(j, f) for j, f in zip(jacobian, flat_shape)]
        jacobian = np.concatenate(flattened_jacobian, axis=-1)
        return jacobian

def main(n_filters_list, n_jacobian_calculations):

    ninput = n_filters_list[0]

    cross_check_parameters = {}
    
    # Create an input vector:
    input_vector = numpy.random.random([ninput, 1])

    n_filters_list.insert(0,1)
    n_filters_list.append(1)

    cross_check_parameters['input_sum'] = numpy.sum(input_vector)
    cross_check_parameters['input_std'] = numpy.std(input_vector)

    layer_weights = [ numpy.random.random([n_filters_list[i],n_filters_list[i+1]]) for i in range(len(n_filters_list)-1)]



    # Create the model:
    M = jit(lambda x : model(x, layer_weights))

    # Forward pass:
    output = M(input_vector)
    
    # Capture the number of parameters:
    nparameters = numpy.sum([ numpy.prod(p.shape) for p in layer_weights ]) 
    cross_check_parameters['n_params'] = nparameters

    # Capture the network output:
    cross_check_parameters['output_sum'] = numpy.sum(output)
    cross_check_parameters['output_std'] = numpy.std(output)

    start = time.time()

    for i in range(n_jacobian_calculations):

        jacobian = compute_jacobian(input_vector, layer_weights)

 
    end = time.time()
    cross_check_parameters['jacobian_sum']  = numpy.sum(jacobian)
    cross_check_parameters['jacobian_std']  = numpy.std(jacobian)
    cross_check_parameters['jacobian_prod'] = numpy.prod(jacobian)
    cross_check_parameters['jacobian_time'] = (end - start) / i
    cross_check_parameters['jacobian_n_calls'] = n_jacobian_calculations

    return cross_check_parameters

if __name__ == '__main__':
    ccp = main([32, 32, 16], 5)
    print(ccp)


