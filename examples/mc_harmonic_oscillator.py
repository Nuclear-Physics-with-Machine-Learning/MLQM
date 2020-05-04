import torch

import numpy 

import sys
sys.path.insert(0, "/Users/lovato/Dropbox/AI-for-QM/")

#from mlqm.samplers      import CartesianSampler
from mlqm.hamiltonians  import HarmonicOscillator_mc
from mlqm.models        import NeuralWavefunction

sig = 1.
dx = 1.
neq = 10
nav = 10
nprop = 10
nvoid = 10
nwalk = 2000
ndim = 1
seed = 17
mass = 1.
omega = 1.

# Initialize neural wave function
wavefunction = NeuralWavefunction(ndim)

# Initialize Hamiltonian 
hamiltonian =  HarmonicOscillator_mc(ndim, mass, omega)

# Initializa Seed
torch.manual_seed(seed)

# Propagation
def energy_metropolis(neq, nav, nprop, nvoid, hamiltonian, wavefunction):
    nblock = neq + nav
    nstep = nprop * nvoid
# Sample initial configurations uniformy between -sig and sig
    x_o = torch.normal(0., sig, size=[nwalk,ndim])
    tot_energy = 0
    tot_error = 0
    tot_acceptance = 0
    for i in range (nblock):
        block_energy = 0
        block_acceptance = 0
        block_weight = 0
        if (i == neq) :
           tot_energy = 0
           tot_acceptance = 0
           tot_error = 0
        for j in range (nstep):
            wpsi_o = wavefunction(x_o)

# Gaussian transition probability 
            x_n = x_o + torch.normal(0., dx, size=[nwalk,ndim])
            wpsi_n = wavefunction(x_n)

# Accepance probability |psi_n|**2 / |psi_o|**2
            prob = torch.FloatTensor.abs(wpsi_n)**2 / torch.FloatTensor.abs(wpsi_o)**2
            accept = torch.ge(prob, torch.rand(size=[nwalk]) )
            x_o = torch.where(accept.view([nwalk,1]), x_n, x_o)
            acceptance = torch.sum(torch.where(accept.view([nwalk,1]), torch.ones(size=[nwalk,1]), torch.zeros(size=[nwalk,1]))) / nwalk

# Compute energy and accumulate estimators within a given block
            if ( j % nvoid == 0):
                x_o.requires_grad_(True)
                energy = hamiltonian.energy(wavefunction, x_o) / nwalk
#                x_o.grad = None
                x_o.requires_grad_(False)
                weight = 1.
                block_energy += energy
                block_weight += weight
                block_acceptance += acceptance

# Accumulate block averages
        tot_energy += block_energy/block_weight
        tot_acceptance += block_acceptance/block_weight
        tot_error += (block_energy/block_weight)**2

# Compute final averages
    energy = tot_energy / nav
    acceptance = tot_acceptance / nav
    error = tot_error / nav
    error = torch.sqrt((error - energy**2)/(nav-1))
    return energy, error, acceptance

energy, error, acceptance = energy_metropolis(neq, nav, nprop, nvoid, hamiltonian, wavefunction)

print("initial_energy", energy, error)
print("initial_acceptance", acceptance)

for i in range(100):
#        optimizer.zero_grad()

        # Compute the energy:
        energy, error, acceptance = energy_metropolis(neq, nav, nprop, nvoid, hamiltonian, wavefunction)
        
        # We back-prop'd through the wave function once already in the energy computation.  Clear the gradients:
        wavefunction.zero_grad()
        
        energy.backward()

        print("energy_grad", energy.requires_grad)

        for p in wavefunction.parameters():
#            i = i + 1
 #           print("p=",i,p.data)
 #           print("p grad=",i,p.grad)
            p.data = p.data - p.grad * 0.01

      #  print([p.grad for p in wavefunction.parameters()])

      #  optimizer.step()
        if i % 1 == 0:
            print(f"step = {i}, energy = {energy.data:.3f}, err = {error.data:.3f}")

