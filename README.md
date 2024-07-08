vis-r
-----

A method for rapid radial profile modelling of interferometric visibilities.

Install the most recent commit on an M2 Mac with `conda`, which will create the shell command `vis-r`, which runs the `python/emcee` version of the code:
```shell
CONDA_SUBDIR=osx-arm64 conda create -n vis-r -c conda-forge python emcee matplotlib scipy multiprocess arviz corner
conda activate vis-r
pip install https://github.com/drgmk/vis-r/archive/refs/heads/main.zip
```

The run the fitting on some visibilities like so
```shell
vis-r -v data/HD109573.12m*npy -g 0.012 -0.035 26.6 76.5 -p 0.013 1.07 0.06 0.01
```

To create an env for the `stan` version `vis-r-stan`, the install is similar and listed at the top of `vis_r_stan.py`