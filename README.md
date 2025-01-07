vis-r
-----

A method for rapid radial profile modelling of interferometric visibilities.
The article outlines the method, and this repo contains some code,
which may do what you want. If it doesn't, 
roll your own, perhaps starting with `examples/hello_world.py`.

A minimal version is outlined in a hello world script [here](examples/hello_world.py).
If you clone this repo, install a new env and the `vis-r` package, and `cd`
into the examples directory it should run with`python hello_world.py`

The `vis-r` executable in the install below will model multiple sets of 
visibilities exported from an ms file. There are `stan` and `python/emcee` 
versions that are generally the same, but with minor differences.
It assumes a single disk plane, but can include multiple radial components 
with independent vertical scales, though these must all be of the same type 
(e.g. Gaussian, power). Each set of visibilities can have independent offset
parameters in the `emcee` version.  Compact central star, point, and Gaussian
sources can also be included. All of this is fairly easy to change for your
use case, e.g. a model with mutiple disk components
could also have independent orientations for each.

### Install

Install the most recent commit on an M2 Mac with `conda`,
which will create the shell command `vis-r` (`frank` wants scipy<1.12):
```shell
CONDA_SUBDIR=osx-arm64 conda create -n vis-r -c conda-forge python emcee matplotlib scipy=1.11.4 multiprocess arviz corner pytest numpy cmdstanpy ipython arviz jupyter
conda activate vis-r
pip install https://github.com/drgmk/vis-r/archive/refs/heads/main.zip
```

If you are likely to fiddle with the code, e.g. to add new radial profile functions,
it is better to clone the repo into a folder, `cd` into it, and install `vis-r`
with `pip install -e .`. Calls to the installed `vis-r` executable will use the code
in this folder.

### Fitting

Run the fitting on some visibilities like so
```shell
vis-r -v data/HD109573.12m*npy -g 0.012 -0.035 26.6 76.5 -p 0.013 1.07 0.06 0.01
```

or for the `stan` version, just add the `--stan` flag:

```shell
vis-r -v data/HD109573.12m*npy --stan -g 0.012 -0.035 26.6 76.5 -p 0.013 1.07 0.06 0.01
```
 The options for each version are mostly the same, but there are some differences.
 The `stan` version first estimates the parameter scales with the Pathfinder algorithm,
 and only the `emcee` version has independent astrometry fitting for 
 each visibility dataset `--astrom` implemented. Both default to the same number of 
 warmup/sampling steps, but `stan` will yield many more independent samples
 for a given number of steps.
 
The suggested method of fitting is to initially use rather hard u,v averaging
with the `--sz` parameter, setting it to something similar to the disk size,
or even smaller. Runs should complete quickly, give reasonable parameter 
estimates, and an idea of whether the model fits the data well via subtraction
and creation of dirty images. Given some initial success, more robust posterior
estimates can be obtained with larger `--sz`, and either longer `emcee` runs 
or sampling with the `stan` implementation.