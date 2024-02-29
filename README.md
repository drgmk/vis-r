vis-r
-----

A method for rapid radial profile modelling of interferometric visibilities.

Install the most recent commit on an M2 Mac with `conda`:
```
CONDA_SUBDIR=osx-arm64 conda create -n vis-r -c conda-forge python emcee matplotlib jupyter scipy multiprocess arviz cmdstanpy corner
conda activate vis-r
pip install https://github.com/drgmk/vis-r/archive/refs/heads/main.zip
```