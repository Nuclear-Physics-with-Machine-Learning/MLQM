import torch
import numpy

#from mlqm import H_BAR
#from mlqm import M


class NuclearPotential(object):

    def __init__(self,nwalk):
        self.nwalk=nwalk
        object.__init__(self)

    def pionless(self, rr):
        pot=torch.zeros(self.nwalk,6)
        vkr=4.0
        v0r=-487.6128
        v0s=-17.5515
        #ar3b=677.79890
        x= vkr*rr
        vr=torch.exp(-x**2/4.0)
        pot[:,0]=vr*v0r*rr
        pot[:,2]=vr*v0s*rr
        return pot


