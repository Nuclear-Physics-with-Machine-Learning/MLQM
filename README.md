[![DOI](https://zenodo.org/badge/255445949.svg)](https://zenodo.org/badge/latestdoi/255445949)



# MLQM

MLQM stands for "Machine Learning Quantum Montecarlo".  This repository contains tools to perform variational monte carlo for nuclear physics, though there are some additional Hamiltonians implemented as development tools and cross checks: the harmonic oscillator and the Hydrogen atom.

## Requirements

The requirements to run this code are:
- python > 3.6
- tensorflow > 2.X (not TF1 compatible)
- hydra-core > 1.0 (for configuration)
- horovod (for scaling and multi-node running)


There is no installation step, it's expect that once you have the requirements install you can begin running immediately.


## Configuration and Running

The main executable is `bin/stochastic_reconfiguration.py`.  You can execute it with:
```bash

python bin/stochastic_reconfiguration.py run_id=MyTestRun 

```

Most parameters have reasonable defaults, which you can change in configuration files (in `config/`) or override on the command line:

```bash

python bin/stochastic_reconfiguration.py run_id=deuteron nparticles=2 iterations=500 optimizer=AdaptiveDelta [... other argument overrides] 

```

## Computational Performance

This software is compatible with both CPUs and GPUs through Tensorflow.  It has good weak and strong scaling performance:

![Scaling performance for 4He on A100 GPUs (ThetaGPU@ALCF)](https://github.com/coreyjadams/AI-for-QM/blob/master/images/Scaling_Performance.png)

The software also has good scaling performance with increasing number of nucleons:

![Nucleon scaling performance on A100 GPUs (ThetaGPU@ALCF)](https://github.com/coreyjadams/AI-for-QM/blob/master/images/NucleonScaling.png)



## Reference

If you use this software, please reference our publication [on arxiv](https://arxiv.org/abs/2007.14282)
