import torch
import time
import numpy 

import sys
#sys.path.insert(0, "/Users/rocconoe/Desktop/AI-for-QM/")
sys.path.insert(0, "/Users/lovato/Dropbox/AI-for-QM/")

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
nwalk = 400
ndim = 3
npart = 3
seed = 17
mass = 1.
omega = 1.
delta = 0.002
eps = 0.0001

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

# Propagation
def energy_metropolis(neq, nav, nprop, nvoid, hamiltonian, wavefunction):
    nblock = neq + nav
    nstep = nprop * nvoid
    block_estimator = Estimator(info=None)
    block_estimator.reset()
    total_estimator = Estimator(info=None)
    total_estimator.reset()
# Sample initial configurations uniformy between -sig and sig
    x_o = torch.normal(0., sig, size=[nwalk, npart, ndim])
    for i in range (nblock):
        block_estimator.reset()
        if (i == neq) :
           total_estimator.reset()
        for j in range (nstep):
            with torch.no_grad(): 
                log_wpsi_o = wavefunction(x_o)
# Gaussian transition probability 
                x_n = x_o + torch.normal(0., dx, size=[nwalk, npart, ndim])
                log_wpsi_n = wavefunction(x_n)
# Accepance probability |psi_n|**2 / |psi_o|**2
                prob = 2 * ( log_wpsi_n - log_wpsi_o )
                accept = torch.ge(prob, torch.log(torch.rand(size=[nwalk])) )
                x_o = torch.where(accept.view([nwalk,1,1]), x_n, x_o)
                acceptance = torch.mean(accept.float())
# Compute energy and accumulate estimators within a given block
            if ( (j+1) % nvoid == 0 and i >= neq ):
                energy, energy_jf = hamiltonian.energy(wavefunction, potential, x_o)
                energy = energy / nwalk
                energy_jf = energy_jf / nwalk
                energy.detach_()
                energy_jf.detach_()

# Compute < O^i >, < H O^i >,  and < O^i O^j > 
                log_wpsi = wavefunction(x_o)
                jac = torch.zeros(size=[nwalk,wavefunction.npt])
                for n in range(nwalk):
                    log_wpsi_n = log_wpsi[n]
                    wavefunction.zero_grad()
                    params = wavefunction.parameters()
                    dpsi_dp = torch.autograd.grad(log_wpsi_n, params, retain_graph=True)
                    dpsi_i_n, indeces_flat = wavefunction.flatten_params(dpsi_dp)
                    jac[n,:] = torch.t(dpsi_i_n)
#                log_wpsi_n.detach_()
                dpsi_i = torch.sum(jac, dim=0) / nwalk
                dpsi_i = dpsi_i.view(-1,1)
                dpsi_i_EL = torch.matmul(energy, jac).view(-1,1)
                dpsi_ij = torch.mm(torch.t(jac), jac) / nwalk

#                print("dpsi_i", dpsi_i)
#                print("dpsi_ij", dpsi_ij)
#                exit()
                
                
                block_estimator.accumulate(torch.sum(energy),torch.sum(energy_jf),acceptance,1.,dpsi_i,dpsi_i_EL,dpsi_ij,1.)

# Accumulate block averages
        if ( i >= neq ):
            total_estimator.accumulate(block_estimator.energy,block_estimator.energy_jf,block_estimator.acceptance,0,block_estimator.dpsi_i,
                block_estimator.dpsi_i_EL,block_estimator.dpsi_ij,block_estimator.weight)

    error, error_jf = total_estimator.finalize(nav)
    energy = total_estimator.energy
    energy_jf = total_estimator.energy_jf
    acceptance = total_estimator.acceptance
    dpsi_i = total_estimator.dpsi_i
    dpsi_i_EL = total_estimator.dpsi_i_EL
    dpsi_ij = total_estimator.dpsi_ij

    with torch.no_grad(): 
        dp_i = opt.sr(energy,dpsi_i,dpsi_i_EL,dpsi_ij)
        gradient = wavefunction.recover_flattened(dp_i, indeces_flat, wavefunction)
        delta_p = [ g for g in gradient]

    return energy, error, energy_jf, error_jf, acceptance, delta_p

t0 = time.time()
energy, error, energy_jf, error_jf, acceptance, delta_p = energy_metropolis(neq, nav, nprop, nvoid, hamiltonian, wavefunction)
t1 = time.time()
print("initial_energy", energy, error)
print("initial_jf_energy", energy_jf, error_jf)
print("initial_acceptance", acceptance)
print("elapsed time", t1 - t0)


for i in range(400):

        # Compute the energy:
        energy, error, energy_jf, error_jf, acceptance, delta_p = energy_metropolis(neq, nav, nprop, nvoid, hamiltonian, wavefunction)
        
        for (p, dp) in zip (wavefunction.parameters(),delta_p):
            p.data = p.data + dp 
            
        if i % 1 == 0:
            print(f"step = {i}, energy = {energy.data:.3f}, err = {error.data:.3f}")
            print(f"step = {i}, energy_jf = {energy_jf.data:.3f}, err = {error_jf.data:.3f}")
            print(f"acc = {acceptance.data:.3f}")


