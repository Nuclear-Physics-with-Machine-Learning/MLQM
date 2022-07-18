from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class Potential(Enum):
    NuclearPotential  = 0
    HarmonicOscilator = 1
    AtomicPotential   = 2

@dataclass
class Hamiltonian:
    mass:     float = 1.0
    form: Potential = MISSING

@dataclass
class NuclearHamiltonian(Hamiltonian):
    """
    This class describes a nuclear hamiltonian.
    Total particles is set elsewhere, so only need to specify how many protons
    """
    model:      str = 'o' # pick from ['a', 'b', 'c', 'd', 'o']
    mass:     float =  938.95
    form: Potential = Potential.NuclearPotential
@dataclass
class AtomicHamiltonian(Hamiltonian):

    mass: float     =  1.
    form: Potential = Potential.AtomicPotential

@dataclass
class HarmonicOscillatorHamiltonian(Hamiltonian):
    mass:  float     = 1.
    omega: float     = 1.
    form:  Potential = Potential.HarmonicOscilator


cs = ConfigStore.instance()
cs.store(group="hamiltonian", name="nuclear",  node=NuclearHamiltonian)
cs.store(group="hamiltonian", name="atomic",   node=AtomicHamiltonian)
cs.store(group="hamiltonian", name="harmonic", node=HarmonicOscillatorHamiltonian)
