from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import List, Any
from omegaconf import MISSING

from .wavefunction import ManyBodyCfg
from .hamiltonian  import Hamiltonian
from .optimizer    import Optimizer

"""
iterations: 5
nparticles: 2
dimension: 3
run_id: ???
model_name: ${hamiltonian.form}_${nparticles}part_${dimension}D.model

"""


@dataclass
class Sampler:
    n_thermalize:              int = 1000
    n_void_steps:              int = 200
    n_observable_measurements: int = 5
    n_walkers_per_observation: int = 50
    n_concurrent_obs_per_rank: int = 5
    use_spin:                 bool = True
    use_isospin:              bool = True

cs = ConfigStore.instance()

cs.store(group="sampler", name="sampler", node=Sampler)

cs.store(
    name="disable_hydra_logging",
    group="hydra/job_logging",
    node={"version": 1, "disable_existing_loggers": False, "root": {"handlers": []}},
)


defaults = [
    {"hamiltonian"  : "nuclear"},
    {"optimizer"    : "adaptivedelta"},
    {"wavefunction" : "many_body"},
    {"sampler"      : "sampler"},
]

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    hamiltonian:   Hamiltonian = MISSING
    optimizer:       Optimizer = MISSING
    wavefunction:  ManyBodyCfg = MISSING
    sampler:           Sampler = MISSING

    run_id:     str = MISSING
    nparticles: int = 2
    dimension:  int = 3
    iterations: int = 200
    save_path:  str = "output/${hamiltonian.form}/${nparticles}particles/${dimension}D/${optimizer}.${run_id}/"
    model_name: str = "${hamiltonian.form}_${nparticles}part_${dimension}D.model"

cs.store(name="base_config", node=Config)
