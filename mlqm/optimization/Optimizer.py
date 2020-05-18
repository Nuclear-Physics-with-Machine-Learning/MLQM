import torch
import numpy

class Optimizer(object):


    def __init__(self,delta,eps,npt):
        self.eps=eps
        self.delta=delta
        self.npt=npt
       
    def sr(self,energy,dpsi_i,dpsi_i_EL,dpsi_ij):

        f_i= self.delta * ( dpsi_i * energy - dpsi_i_EL )
        S_ij = torch.zeros_like(dpsi_ij)
        for i in range (self.npt):
            for j in range (self.npt):
                S_ij[i,j] = dpsi_ij[i,j] - dpsi_i[i] * dpsi_i[j]
            S_ij[i,i] = S_ij[i,i] + self.eps

        U_ij = torch.cholesky(S_ij, upper=True, out=None)
        dp_i = torch.cholesky_solve(f_i, U_ij, upper=True, out=None) 
#        test = torch.mm(S_ij, dp_i)
        dp_0 = 1. - self.delta * energy - torch.sum(dpsi_i*dp_i) 
        dp_i = dp_i / dp_0
        return dp_i




    
