import torch
import time
import numpy 

import sys
sys.path.insert(0, "/Users/lovato/Dropbox/AI-for-QM/")

#from mlqm.samplers      import CartesianSampler
from mlqm.hamiltonians  import HarmonicOscillator_mc
from mlqm.models        import NeuralWavefunction

sig = 0.2
dx = .5
neq = 10
nav = 10
nprop = 10
nvoid = 20
nwalk = 200
ndim = 3
seed = 17
mass = 1.
omega = 1.

# Initialize Seed
torch.manual_seed(seed)

# Initialize neural wave function and compute the number of parameters
wavefunction = NeuralWavefunction(ndim)
wavefunction.count_parameters()

# Initialize Hamiltonian 
hamiltonian =  HarmonicOscillator_mc(ndim, mass, omega, nwalk, ndim)

# Propagation
def energy_metropolis(neq, nav, nprop, nvoid, hamiltonian, wavefunction):
    nblock = neq + nav
    nstep = nprop * nvoid
# Sample initial configurations uniformy between -sig and sig
    x_o = torch.normal(0., sig, size=[nwalk,ndim])
    tot_energy = 0
    tot_energy_jf = 0
    tot_energy2 = 0
    tot_energy2_jf = 0
    tot_acceptance = 0
    tot_dpsi_i = 0
    tot_dpsi_i_EL = 0
    tot_dpsi_ij = 0
    for i in range (nblock):
        block_energy = 0
        block_energy2 = 0
        block_energy_jf = 0
        block_energy2_jf = 0
        block_acceptance = 0
        block_weight = 0
        block_dpsi_i = 0
        block_dpsi_i_EL = 0
        block_dpsi_ij = 0
        if (i == neq) :
           tot_energy = 0
           tot_energy2 = 0
           tot_energy_jf = 0
           tot_energy2_jf = 0
           tot_acceptance = 0
           tot_dpsi_i = 0
           tot_dpsi_i_EL = 0
           tot_dpsi_ij = 0
        for j in range (nstep):
            with torch.no_grad(): 
                wpsi_o = wavefunction(x_o)

# Gaussian transition probability 
                x_n = x_o + torch.normal(0., dx, size=[nwalk,ndim])
                wpsi_n = wavefunction(x_n)

# Accepance probability |psi_n|**2 / |psi_o|**2
                prob = torch.FloatTensor.abs(wpsi_n)**2 / torch.FloatTensor.abs(wpsi_o)**2
                accept = torch.ge(prob, torch.rand(size=[nwalk]) )
#                print("wpsi_o=", wpsi_o)
#                print("wpsi_n=", wpsi_n)
#                print("prob=", prob)
#                print("accept=", accept)
                x_o = torch.where(accept.view([nwalk,1]), x_n, x_o)
                acceptance = torch.sum(torch.where(accept.view([nwalk,1]), torch.ones(size=[nwalk,1]), torch.zeros(size=[nwalk,1]))) / nwalk

# Compute energy and accumulate estimators within a given block
            if ( (j+1) % nvoid == 0):
                x_o.requires_grad_(True)
                energy, energy_jf = hamiltonian.energy(wavefunction, x_o)
#                x_o.grad.data.zero_()
                energy = energy / nwalk
                energy_jf = energy_jf / nwalk
                x_o.requires_grad_(False)
                energy.detach_()
                energy_jf.detach_()

# Compute < O^i >, < H O^i >,  and < O^i O^j > 
                wpsi = wavefunction(x_o)
                jac = torch.zeros(size=[nwalk,wavefunction.npt])
                for n in range(nwalk):
                    log_wpsi_n = torch.log(wpsi[n])
                    wavefunction.zero_grad()
                    params = wavefunction.parameters()
                    dpsi_dp = torch.autograd.grad(log_wpsi_n, params, retain_graph=True)
                    dpsi_i_n, indeces_flat = wavefunction.flatten_params(dpsi_dp)
                    jac[n,:] = torch.t(dpsi_i_n)
                log_wpsi_n.detach_()
                dpsi_i = torch.sum(jac, dim=0) / nwalk
                dpsi_i = dpsi_i.view(-1,1)
                dpsi_i_EL = torch.matmul(energy, jac).view(-1,1)
                dpsi_ij = torch.mm(torch.t(jac), jac) / nwalk

                block_energy += torch.sum(energy)
                block_energy2 += torch.sum(energy)**2
                block_energy_jf += torch.sum(energy_jf)
                block_energy2_jf += torch.sum(energy_jf)**2
                block_dpsi_i += dpsi_i
                block_dpsi_i_EL += dpsi_i_EL
                block_dpsi_ij += dpsi_ij
                block_weight += 1
                block_acceptance += acceptance

# Accumulate block averages
        tot_energy += block_energy / block_weight
        tot_energy2 += block_energy2  / block_weight
        tot_energy_jf += block_energy_jf / block_weight
        tot_energy2_jf += block_energy2_jf  / block_weight
        tot_dpsi_i += block_dpsi_i / block_weight
        tot_dpsi_i_EL += block_dpsi_i_EL / block_weight
        tot_dpsi_ij += block_dpsi_ij / block_weight
        tot_acceptance += block_acceptance / block_weight

    energy = tot_energy / nav
    energy2 = tot_energy2 / nav
    energy_jf = tot_energy_jf / nav
    energy2_jf = tot_energy2_jf / nav
    acceptance = tot_acceptance / nav
    error = torch.sqrt((energy2 - energy**2)/(nav-1))
    error_jf = torch.sqrt((energy2_jf - energy_jf**2)/(nav-1))


#    print("tot_grad_1", tot_grad_1)
#    print("tot_dpsi_i_EL", tot_dpsi_i_EL)

#    print("tot_grad_2", tot_grad_2)
#    print("tot_dpsi_i", tot_dpsi_i)

    dpsi_i = tot_dpsi_i / nav
    dpsi_i_EL = tot_dpsi_i_EL / nav
    dpsi_ij = tot_dpsi_ij / nav

    with torch.no_grad(): 
        delta = 0.04
        eps = 0.001
        f_i = delta * ( dpsi_i * energy - dpsi_i_EL )
        S_ij = torch.zeros_like(dpsi_ij)
        for i in range (wavefunction.npt):
            for j in range (wavefunction.npt):
                S_ij[i,j] = dpsi_ij[i,j] - dpsi_i[i] * dpsi_i[j]
            S_ij[i,i] = S_ij[i,i] + eps
        U_ij = torch.cholesky(S_ij, upper=True, out=None)
        dp_i = torch.cholesky_solve(f_i, U_ij, upper=True, out=None) 
#        test = torch.mm(S_ij, dp_i)
        dp_0 = 1. - delta * energy - torch.sum(dpsi_i*dp_i) 
        dp_i = dp_i / dp_0
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

#print("initial_gradient", gradient)

for i in range(80):
#        optimizer.zero_grad()

        # Compute the energy:
        energy, error, energy_jf, error_jf, acceptance, delta_p = energy_metropolis(neq, nav, nprop, nvoid, hamiltonian, wavefunction)
        
        lr = 1.
        print("lr=", lr)
        for (p, dp) in zip (wavefunction.parameters(),delta_p):
            p.data = p.data + dp * lr
            
        if i % 1 == 0:
            print(f"step = {i}, energy = {energy.data:.3f}, err = {error.data:.3f}")
            print(f"step = {i}, energy_jf = {energy_jf.data:.3f}, err = {error_jf.data:.3f}")
            print(f"acc = {acceptance.data:.3f}")


