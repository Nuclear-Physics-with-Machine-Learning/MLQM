# Python built ins:
import sys, os
import time
import logging

# Frameworks:
import numpy 
import torch

# Add the local folder to the import path:
top_folder = os.path.dirname(os.path.abspath(__file__))
top_folder = os.path.dirname(top_folder)
sys.path.insert(0,top_folder)

#from mlqm.samplers      import CartesianSampler
from mlqm.hamiltonians  import HarmonicOscillator_mc
from mlqm.models        import NeuralWavefunction
from mlqm.hamiltonians  import NuclearPotential
from mlqm.samplers      import Estimator
from mlqm.optimization  import Optimizer



sig = 0.2
dx = 0.2
neq = 10
nav = 10
nprop = 10
nvoid = 50
nwalk = 1
ndim = 3
npart = 4
seed = 17
mass = 1.
omega = 1.
delta = 0.002
eps = 0.0002
model_save_path = f"./helium{npart}.model"


# Initialize Seed
torch.manual_seed(seed)

# Initialize neural wave function and compute the number of parameters
wavefunction = NeuralWavefunction(ndim, npart)
wavefunction.count_parameters()

# Initialize Potential
potential = NuclearPotential(nwalk)

# Initialize Hamiltonian 
hamiltonian =  HarmonicOscillator_mc(mass, omega, nwalk, ndim, npart)

#Initialize Optimizer
opt=Optimizer(delta,eps,wavefunction.npt)

#This step loads the model:
wavefunction.load_state_dict(torch.load(model_save_path))

# Now you can use the wavefunction as before, with the parameters reloaded.
