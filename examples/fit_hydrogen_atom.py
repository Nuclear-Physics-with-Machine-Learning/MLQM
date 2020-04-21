import torch
import numpy

import sys
sys.path.insert(0, "/Users/corey.adams/ML_QM/")

from mlqm.samplers      import CartesianSampler
from mlqm.hamiltonians  import Hydrogen
from mlqm.models        import NeuralWavefunction


def train():

    dimension = 3

    # First, create an instance of the sampler and the hamiltonian:
    delta   = 0.2
    mins    = -5.
    maxes   = 5.
    sampler = CartesianSampler(dimension, delta, mins, maxes)

    hamiltonian =  Hydrogen(mu=1.0, e=1.0)

    # Create an instance of the wave function:
    wavefunction = NeuralWavefunction(n=dimension)

    inputs = sampler.sample()

    wavefunction.update_normalization(inputs, delta**dimension)

    energy = hamiltonian.energy(wavefunction, inputs, delta**dimension)
    print("Initial Energy is ", energy)

    # Create an optimizer for the initial wave function:
    optimizer = torch.optim.Adam(wavefunction.parameters(), lr=0.001)

    # Now, iterate until the energy stops decreasing:


    for i in range(1000):

        # Reset the gradient on the inputs:
        inputs.grad = None

        # Compute the energy:
        energy = hamiltonian.energy(wavefunction, inputs, delta**dimension)

        # We back-prop'd through the wave function once already in the energy computation.  Clear the gradients:
        wavefunction.zero_grad()
        
        (energy + 100).backward()

        # print([p.grad for p in wavefunction.parameters()])

        optimizer.step()
        # if i % 100 == 0:
        print(f"step = {i}, energy = {energy.data:.2f}")
        
        # Lastly, update the normaliztion 
        wavefunction.update_normalization(inputs, delta**dimension)

    print(f"First wavefunction energy is {energy.data}")


if __name__ == "__main__":
    train()
