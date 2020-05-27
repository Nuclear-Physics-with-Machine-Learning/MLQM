import torch
import numpy

class Optimizer(object):

    def __init__(self,delta,eps,npt):
        self.eps=eps
        self.delta=delta
        self.npt=npt

    def par_dist(self, dp_i, S_ij):
        dist = 0
        for i in range (self.npt):
            for j in range (self.npt):
                dist += S_ij[i,j]*dp_i[i]*dp_i[j]
        return dist
       
    def sr(self,energy,dpsi_i,dpsi_i_EL,dpsi_ij):
        f_i= (self.delta * ( dpsi_i * energy - dpsi_i_EL )).double()
        S_ij = torch.zeros_like(dpsi_ij)
        for i in range (self.npt):
            for j in range (self.npt):
                S_ij[i,j] = dpsi_ij[i,j] - dpsi_i[i] * dpsi_i[j]

#        print("dpsi_i", dpsi_i)
#        print("dpsi_ij", dpsi_ij)
#        print("S_ij=", S_ij)
#        print("dpsi_i_EL", dpsi_i_EL)
#        print("energy", energy)
        
        for i in range (1000):
            S_ij_d = torch.clone(torch.detach(S_ij)).double()
            S_ij_d += 2**i * self.eps * torch.eye(self.npt)
#            print("S_ij_d", S_ij_d)
            U_ij = torch.cholesky(S_ij_d, upper=True, out=None)
            dp_i = torch.cholesky_solve(f_i, U_ij, upper=True, out=None) 
#            print("f_i", dp_i)
            dp_0 = 1. - self.delta * energy - torch.sum(dpsi_i*dp_i) 
            dp_i = dp_i / dp_0
#            print("dp_i", dp_i)
            dist = self.par_dist(dp_i, S_ij)
            torch.set_printoptions(precision=8)
            print("dist param", dist.data[0])
            dp_i = dp_i.float()
            if (dist < 0.0005):
                break
        return dp_i







    
