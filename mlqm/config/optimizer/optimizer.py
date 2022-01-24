from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

class Optimizer(Enum):
    flat             = 0
    adaptive_delta   = 1
    adaptive_epsilon = 2

@dataclass
class AdaptiveDelta:
    form:  Optimizer = Optimizer.adaptive_delta
    epsilon:   float = 0.0001
    delta_max: float = 0.1
    delta_min: float = 0.00001
    # form: AdaptiveDelta

@dataclass
class AdaptiveEpsilon:
    form:    Optimizer = Optimizer.adaptive_epsilon
    delta:       float = 0.01
    epsilon_max: float = 0.1
    epsilon_min: float = 1e-8

@dataclass
class Flat:
    form: Optimizer = Optimizer.flat
    delta:    float = 0.001
    epsilon:  float = 0.0001


cs = ConfigStore.instance()
cs.store(group="optimizer", name="adaptivedelta",   node=AdaptiveDelta)
cs.store(group="optimizer", name="adaptiveepsilon", node=AdaptiveEpsilon)
cs.store(group="optimizer", name="flat",            node=Flat)
