from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class Wavefunction:
    pass

@dataclass 
class DeepSetsWavefunction(Wavefunction):
    n_filters_per_layer: int = 16
    n_layers:            int = 2
    bias:               bool = True
    activation:          str = "tanh"
    residual:           bool = False
    mean_subtract:      bool = True
    confinement:       float = 0.1


cs = ConfigStore.instance()
cs.store(group="wavefunction", name="deepsets",   node=DeepSetsWavefunction)
