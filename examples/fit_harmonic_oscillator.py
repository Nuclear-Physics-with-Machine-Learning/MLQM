import torch
import numpy 

torch.manual_seed(0)

import sys
sys.path.insert(0, "/Users/corey.adams/ML_QM/")

from mlqm.samplers      import MetropolisSampler
from mlqm.hamiltonians  import HarmonicOscillator
from mlqm.models        import NeuralWavefunction


def train():

    dimension       = 1  # Dimensionality of the physical space
    nvoid           = 50 # Number of times to kick the walkers before computing other parameters again
    n_prop          = 10 # Number of times to compute the observables, per model update
    n_model_updates = 100 # Number of times the model gets updated
    kick            = 0.5 # Gaussian sigma for the kick
    acceptance      = 0.0 # Initial value of acceptance


    # First, create an instance of the sampler and the hamiltonian:
    sampler = MetropolisSampler(
        ndim        = dimension, 
        nwalkers    = 10000, 
        initializer = torch.normal, 
        init_params = [0.0, 0.2])

    hamiltonian =  HarmonicOscillator(n=dimension, M=1.0, omega = 1.0)

    # Create an instance of the wave function:
    wavefunction = NeuralWavefunction(n=dimension)

    inputs = sampler.sample()


    wavefunction.update_normalization(inputs)

    energy, energy_by_parts = hamiltonian.energy(wavefunction, inputs )
    # print("Initial Energy is ", energy)

    # Create an optimizer for the initial wave function:
    optimizer = torch.optim.Adam(wavefunction.parameters(), lr=0.001)


    # Iterate over the model
    for i_update in range(n_model_updates):

        # Clear the accumulated gradients
        optimizer.zero_grad()

        # energy = 0.0
        # energy_by_parts = 0.0
        grad = None
        # Loop n_prop times to compute energy
        for i_prop in range(n_prop):

            # First, walk the walkers without computing any observables
            # Kick the sampler:
            for i_void in range(nvoid):
                acceptance = sampler.kick(
                    wavefunction = wavefunction, 
                    kicker=torch.normal, 
                    kicker_params=[0.0,kick])

            # Now, compute the observables:
            inputs = sampler.sample()

            # Reset the gradient on the inputs:
            inputs.grad = None

            # Compute the energy:
            energy, energy_by_parts = hamiltonian.energy(wavefunction, inputs)

            # We back-prop'd through the wave function once already in the energy computation.  Clear the gradients:
            wavefunction.zero_grad()
        
            summed_energy = torch.mean(energy)
            summed_energy_by_parts = torch.mean(energy_by_parts)
            summed_energy.backward()


            if grad is None:
                grad = [p.grad for p in wavefunction.parameters()]
            else:
                grad = [g + p.grad for g, p in zip(grad, wavefunction.parameters())]
# #        i = 0 

        for p in wavefunction.parameters():
#            i = i + 1
 #           print("p=",i,p.data)
 #           print("p grad=",i,p.grad)
            p.data = p.data - p.grad * 0.005


        # optimizer.step()
        if i_update % 10 == 0:
            print(f"step = {i_update}, energy = {summed_energy.data:.2f}")
            print(f"  step = {i_update}, energy_by_parts = {summed_energy_by_parts.data:.2f}")
            print(f"  step = {i_update}, acceptance = {acceptance:.2f}")
        
        # Update the normaliztion 
        wavefunction.update_normalization(inputs)


    exit()

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
