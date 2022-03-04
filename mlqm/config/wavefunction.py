from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class DeepSetsCfg():
    n_filters_per_layer: int = 16
    n_layers:            int = 2
    bias:               bool = True
    activation:          str = "tanh"
    residual:           bool = False
    confinement:       float = .5

@dataclass
class SpatialCfg():
    n_layers: int = 2
    n_filters_per_layer: int = 16
    bias:               bool = True
    activation:          str = "tanh"
    residual:           bool = False
    confinement:       float = .5

@dataclass
class ManyBodyCfg():
    mean_subtract:        bool = True
    deep_sets_cfg: DeepSetsCfg = DeepSetsCfg()
    spatial_cfg:    SpatialCfg = SpatialCfg()


cs = ConfigStore.instance()
cs.store(group="wavefunction", name="many_body", node=ManyBodyCfg)
