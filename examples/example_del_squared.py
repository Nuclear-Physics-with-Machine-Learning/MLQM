import torch
import time
import numpy 

import sys
sys.path.insert(0, "/Users/corey.adams/ML_QM/")

from mlqm.samplers import CartesianSampler

def forward_model(inputs):
    # Polynomial function:

    return 5 + 7*torch.sum(inputs, dim=-1) - inputs[:,1]**2 + 9 *torch.sum(inputs**3, dim=-1)

def grad_forward(inputs):
    return 7 + 3*9 * inputs**2

def del_forward(inputs):
    return 3*9*2 * torch.sum(inputs, dim=-1) - 2

sampler = CartesianSampler(n=2, delta=0.5, mins=-1, maxes=1)
inputs = sampler.sample()

print(inputs)
            
grad_accum = torch.ones(len(inputs))

w_of_x = forward_model(inputs)

dw_dx = torch.autograd.grad(
    outputs=w_of_x, 
    inputs=inputs, 
    grad_outputs = grad_accum,
    retain_graph=True, 
    create_graph=True)[0]

print(dw_dx)
print(grad_forward(inputs))

grad_accum = torch.ones(inputs.shape)

# Compute the second derivative:
d2w_dx2 = torch.autograd.grad(
    outputs=dw_dx, 
    inputs=inputs, 
    grad_outputs = grad_accum,
    retain_graph=True, 
    create_graph=True)[0]

print(torch.sum(d2w_dx2, dim=-1))
print(del_forward(inputs))