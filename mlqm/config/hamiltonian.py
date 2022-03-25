from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class SpinConf:
    """
    This class describes a spin configuration.
    Set the total spin, as well as the total z projection.

    Units are set as integer multiples of hbar/2 (the spin of a single particle)
    """
    # Total spin of 2 means two particles, etc.
    total_spin:   int = 1

    # Z projection of 1 means 1 particle is up, etc.
    z_projection: int = 1

    def __post_init__(self):
        """
        Check that the z projection is physical
        """
        if abs(self.z_projection) > self.total_spin:
            raise AttributeError("Z projection must be less than or equal to total spin")

class Potential(Enum):
    NuclearPotential  = 0
    HarmonicOscilator = 1
    AtomicPotential   = 2

@dataclass
class Hamiltonian:
    mass:     float = 1.0
    form: Potential = MISSING
    spin:  SpinConf = SpinConf(2,2)

@dataclass
class NuclearHamiltonian(Hamiltonian):
    """
    This class describes a nuclear hamiltonian.
    Total particles is set elsewhere, so only need to specify how many protons
    """
    model:      str = 'o' # pick from ['a', 'b', 'c', 'd', 'o']
    mass:     float =  938.95
    form: Potential = Potential.NuclearPotential
    isospin: SpinConf = SpinConf(2,1)

@dataclass
class AtomicHamiltonian(Hamiltonian):

    mass: float     =  1.
    form: Potential = Potential.AtomicPotential
    z:    float     =  1.

@dataclass
class HarmonicOscillatorHamiltonian(Hamiltonian):
    mass:  float     = 1.
    omega: float     = 1.
    form:  Potential = Potential.HarmonicOscilator


cs = ConfigStore.instance()
cs.store(group="hamiltonian", name="nuclear",  node=NuclearHamiltonian)
cs.store(group="hamiltonian", name="atomic",   node=AtomicHamiltonian)
cs.store(group="hamiltonian", name="harmonic", node=HarmonicOscillatorHamiltonian)
