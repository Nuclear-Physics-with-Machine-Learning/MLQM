import torch
import numpy as np

#from mlqm import H_BAR
#from mlqm import M


class NuclearPotential(object):

    def __init__(self,nwalk):
        self.nwalk=nwalk
        object.__init__(self)

    def pionless_2b(self, rr):
        pot_2b=torch.zeros(self.nwalk,6)
        vkr = 4.0
        v0r = -487.6128
        v0s = -17.5515
        x = vkr*rr
        vr = torch.exp(-x**2/4.0)
        pot_2b[:,0] = vr*v0r
        pot_2b[:,2] = vr*v0s
        return pot_2b

    def pionless_3b(self, rr):
        pot_3b = torch.zeros(self.nwalk)
        vkr = 4.0
        ar3b = np.sqrt(677.79890)
        x = vkr*rr
        vr = torch.exp(-x**2/4.0)
        pot_3b = vr*ar3b
        return pot_3b



