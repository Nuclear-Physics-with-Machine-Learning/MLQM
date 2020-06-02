import torch
import numpy

class Optimizer(object):

    def __init__(self,delta,eps,npt):
        self.eps=eps
        self.delta=delta
        self.npt=npt

    def par_dist(self, dp_i, S_ij):
        # dist = 0
        # for i in range (self.npt):
        #     for j in range (self.npt):
        #         dist += S_ij[i,j]*dp_i[i]*dp_i[j]

        # This replaces the double for loop.  Don't do that in python.
        D_ij = S_ij * (dp_i * dp_i.T)
        dist = torch.sum(D_ij_fast.flatten())


        return dist

    def sr(self,energy,dpsi_i,dpsi_i_EL,dpsi_ij):
        f_i= (self.delta * ( dpsi_i * energy - dpsi_i_EL )).double()

        # This also replaces the double for loop ... don't do that :)
        S_ij = dpsi_ij - dpsi_i * dpsi_i.T

        #

#        print("dpsi_i", dpsi_i)
#        print("dpsi_ij", dpsi_ij)
#        print("S_ij=", S_ij)
#        print("dpsi_i_EL", dpsi_i_EL)
#        print("energy", energy)

        for i in range (1000):
            S_ij_d = torch.clone(torch.detach(S_ij)).double()
            S_ij_d += 10**i * self.eps * torch.eye(self.npt)
            det_test = torch.det(S_ij_d)
            torch.set_printoptions(precision=8)
            try:
               U_ij = torch.cholesky(S_ij_d, upper=True, out=None)
               positive_definite = True
            except RuntimeError:
               positive_definite = False
               print("Warning, Cholesky did not find a positive definite matrix")
            if (positive_definite):
               dp_i = torch.cholesky_solve(f_i, U_ij, upper=True, out=None)
#            print("f_i", dp_i)
               dp_0 = 1. - self.delta * energy - torch.sum(dpsi_i*dp_i)
               dp_i = dp_i / dp_0
#            print("dp_i", dp_i)
               dist = self.par_dist(dp_i, S_ij)
               torch.set_printoptions(precision=8)
               # Originally this accessed the [0] element but that's not necessary now
               print("dist param", dist.data)
               dp_i = dp_i.float()
               if (dist < 0.0005):
                  break
        return dp_i
