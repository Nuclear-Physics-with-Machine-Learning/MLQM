import torch
import numpy

import sys
sys.path.insert(0, "/Users/corey.adams/ML_QM/")

from mlqm.samplers      import CartesianSampler
from mlqm.hamiltonians  import HarmonicOscillator
from mlqm.models        import NeuralWavefunction


def train():

    dimension = 2

    # First, create an instance of the sampler and the hamiltonian:
    delta   = 0.5
    mins    = -20.
    maxes   = 20.
    sampler = CartesianSampler(dimension, delta, mins, maxes)

    hamiltonian =  HarmonicOscillator(n=dimension, M=1.0, omega = 1.0)

    # Create an instance of the wave function:
    wavefunction = NeuralWavefunction(n=dimension)

    inputs = sampler.sample()

    wavefunction.update_normalization(inputs, delta**dimension)

    energy = hamiltonian.energy(wavefunction, inputs, delta**dimension)
    print("Initial Energy is ", energy)

    # Create an optimizer for the initial wave function:
    optimizer = torch.optim.Adam(wavefunction.parameters(), lr=0.001)

    # Now, iterate until the energy stops decreasing:


    for i in range(3000):

        # Reset the gradient on the inputs:
        inputs.grad = None

        # Compute the energy:
        energy = hamiltonian.energy(wavefunction, inputs, delta**dimension)

        # We back-prop'd through the wave function once already in the energy computation.  Clear the gradients:
        wavefunction.zero_grad()
        
        energy.backward()

        # print([p.grad for p in wavefunction.parameters()])

        optimizer.step()
        if i % 100 == 0:
            print(f"step = {i}, energy = {energy.data:.2f}")
        
        # Lastly, update the normaliztion 
        wavefunction.update_normalization(inputs, delta**dimension)

    print(f"First wavefunction energy is {energy.data}")

    wavefunction_list   = [ wavefunction ]
    energy_list         = [ energy.data ]
    wavefunction_values = [ wavefunction(inputs).detach()]

    # Now, go ahead and compute more wavefunctions:

    for n in range(dimension):

        wavefunction = NeuralWavefunction(n=dimension)
        optimizer = torch.optim.Adam(wavefunction.parameters(), lr=0.001)

        for i in range(3000):

            # Reset the gradient on the inputs:
            inputs.grad = None

            # Compute the energy:
            energy = hamiltonian.energy(wavefunction, inputs, delta**dimension)

            # We back-prop'd through the wave function once already in the energy computation.  Clear the gradients:
            wavefunction.zero_grad()
        
            # Compute the orthogonality:
            orthogonality = None
            this_values = wavefunction(inputs)
            for w in wavefunction_values:
                ortho = torch.sum(this_values * w * delta**dimension)**2
                if orthogonality is None:
                    orthogonality = ortho
                else:
                    orthogonality += ortho
            # print([p.grad for p in wavefunction.parameters()])

            (energy + 5*orthogonality).backward()


            optimizer.step()
            if i % 100 == 0:
                print(f"step = {i}, energy = {energy.data:.2f}, orthogonality = {orthogonality.data:.2f}")
            
            # Lastly, update the normaliztion 
            wavefunction.update_normalization(inputs, delta**dimension)

        energy_list.append(energy.data)
        wavefunction_list.append(wavefunction)
        wavefunction_values.append(wavefunction(inputs).detach())




if __name__ == "__main__":
    train()
