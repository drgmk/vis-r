vis-r
-----

A method for rapid radial profile modelling of interferometric visibilities.

Install the most recent commit on an M2 Mac with `conda`. This will create the shell command `vis-r`:
```shell
CONDA_SUBDIR=osx-arm64 conda create -n vis-r -c conda-forge python emcee matplotlib jupyter scipy multiprocess arviz cmdstanpy corner
conda activate vis-r
pip install https://github.com/drgmk/vis-r/archive/refs/heads/main.zip
```

To install and env for the `stan` version `vis-r-stan`, the install is similar and listed at the top of `vis_r_stan.py`