from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import List, Any
from omegaconf import MISSING

from .wavefunction import ManyBodyCfg
from .hamiltonian  import Hamiltonian
from .optimizer    import Optimizer




@dataclass
class Sampler:
    n_thermalize:              int = 1000
    n_void_steps:              int = 200
    n_observable_measurements: int = 10
    n_walkers_per_observation: int = 1000
    n_concurrent_obs_per_rank: int = 10
    n_particles:               int = 4
    n_dim:                     int = 3
    use_isospin:              bool = True
    use_spin:                 bool = True
    n_spin_up:                 int = 1
    n_protons:                 int = 1

    def __post_init__(self):
        """
        Check that the z projection is physical
        """
        if abs(self.n_spin_up) > self.n_particles:
            raise AttributeError("N spin up particles must be less than or equal to total particles")
        if abs(self.n_protons) > self.n_particles:
            raise AttributeError("N protons particles must be less than or equal to total particles")

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
    iterations: int = 200
    save_path:  str = "output/${hamiltonian.form}/${sampler.n_particles}particles/${sampler.n_dim}D/${optimizer}.${run_id}/"
    model_name: str = "${hamiltonian.form}_${sampler.n_particles}part_${sampler.n_dim}D.model"

cs.store(name="base_config", node=Config)
