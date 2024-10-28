vis-r
-----

A method for rapid radial profile modelling of interferometric visibilities.
The article outlines the method, and this repo contains some code,
which may do what you want. If it doesn't, 
roll your own or start with `examples/hello_world.py`.

A minimal version is outlined in a hello world script [here](examples/hello_world.py).
If you clone this repo, install a new env and the `vis-r` package, and go into the examples directory it should run with
`python hello_world.py`

The `vis-r` executable in the install below will model multiple sets of visibilities exported from an ms file.
It assumes a single disk plane, but can include multiple radial components with independent vertical scales,
though these must all be of the same type (e.g. Gaussian, power).
Each set of visibilities can have independent offset parameters.
Compact central star, point, and Gaussian sources can also be included.
All of this is fairly easy to change for your use case.

#### Install


Install the most recent commit on an M2 Mac with `conda`,
which will create the shell command `vis-r`,
which runs the `python/emcee` version of the code:
```shell
CONDA_SUBDIR=osx-arm64 conda create -n vis-r -c conda-forge python emcee matplotlib scipy multiprocess arviz corner pytest
conda activate vis-r
pip install https://github.com/drgmk/vis-r/archive/refs/heads/main.zip
```

(If you are likely to fiddle with the code, it is better to clone the repo into a folder,
cd into it, and instead install vis-r with `pip install -e .`. Calls to the installed
`vis-r` executable will use the code in this folder.)

The run the fitting on some visibilities like so
```shell
vis-r -v data/HD109573.12m*npy -g 0.012 -0.035 26.6 76.5 -p 0.013 1.07 0.06 0.01
```
