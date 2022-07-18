
import jax.numpy as numpy
import jax.random as random

from jax import vmap, jit

import flax.linen as nn
from typing import Tuple, Sequence




class IndividualModule(nn.Module):
    n_outputs: Tuple[int, ...]

    def setup(self):
        self.layers = [ nn.Dense(n_out) for n_out in self.n_outputs]
  
    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = nn.sigmoid(layer(out))
            
        return out

class AggregateModule(nn.Module):
    n_outputs: Tuple[int, ...]

    def setup(self):
        self.layers = [ nn.Dense(n_out) for n_out in self.n_outputs]


    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = nn.sigmoid(layer(out))
            
        return out.reshape(())


class DeepSetsCorrelator(nn.Module):
    individual_layers:  Tuple[int, ...]
    aggregate_layers:   Tuple[int, ...]
    confinement:        float

    def setup(self):
        self.individual_network = IndividualModule(self.individual_layers)
        self.aggregate_network  = AggregateModule(self.aggregate_layers)
        
    def __call__(self, x):

        inputs = x
        # # First, do we mean subtract?
        # if self.mean_subtract:
        #     mean = x.mean(axis=0)
        #     inputs = x - mean
        # else:
        #     inputs = x
        
        # Apply the individual network, sum over particles:
        individual_response = self.individual_network(inputs).sum(axis=0)
        aggregate_response = self.aggregate_network(individual_response)
        
        # We also need apply a confinement:
        boundary = - self.confinement * (inputs**2).sum(axis=(0,1))
        
        # Return it flattened to a single value for a single walker:
        return numpy.exp(aggregate_response + boundary).reshape(())
        # return numpy.exp(aggregate_response + boundary)


def initialize_correlator(walkers, key, config):

    # TODO: build the layers from the config

    i_layers = [config.n_filters_per_layer for l in range(config.n_layers) ]
    a_layers = [config.n_filters_per_layer for l in range(config.n_layers) ]

    a_layers[-1] = 1


    correlator = DeepSetsCorrelator(
        individual_layers = tuple(i_layers),
        aggregate_layers  = tuple(a_layers),
        confinement       = config.confinement,
    )

    correlator_variables = correlator.init(key, walkers[0])
    # correlator.individual_network.init(key, walkers[0])

    # Call the correlator:
    c_out = correlator.apply(correlator_variables, walkers[0])

    correlator.apply = jit(vmap(jit(correlator.apply), in_axes=[None, 0]))

    return correlator, correlator_variables


    

