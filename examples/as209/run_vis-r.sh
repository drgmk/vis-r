# calls to run fitting

# this model is somewhat independent of Guzman+18, but is inspired by it.
# this command has been run many times by hand to build up the model
# and to get parameters to converge sensibly, this is the last one
# that was run (where the previous run is being extended and the input
# parameters are ignored)
vis-r -v ~/tmp/dsharp/AS209/visibilities/AS209_continuum.8ch.npy -o ./ -g 0.0015 -0.003 86 34.9 -p 0.0064 0.0025 0.025 0.005 -p 0.02 0.122 0.022 0.005 -p 0.075 0.23 0.08 0.005 -p 0.04 0.37 0.045 0.002 -p 0.035 0.61 0.026 0.001 -p 0.005 0.75 0.06 0.01 -p 0.04 1. 0.025 0.002 -p 0.035 1.15 0.08 0.001 --sz 2 --steps 2000 --walker-factor 2 --restore --rew
vis-r -v ~/tmp/dsharp/AS209/visibilities/AS209_continuum.8ch.npy -o as209_flat -g 0.0015 -0.003 86 34.9 -p 0.0064 0.0025 0.025 -p 0.02 0.122 0.022 -p 0.075 0.23 0.08 -p 0.04 0.37 0.045 -p 0.035 0.61 0.026 -p 0.005 0.75 0.06 -p 0.04 1. 0.025 -p 0.035 1.15 0.08 --sz 2 --steps 2000 --walker-factor 2 --rew --z-lim 0

# with gaussian as ~unresolved inner disk 
vis-r -v ~/tmp/dsharp/AS209/visibilities/AS209_continuum.8ch.npy -o as209_flat -g 0.0015 -0.003 86 34.9 --inner 0.006 0.028 -p 0.02 0.122 0.022 -p 0.075 0.23 0.08 -p 0.04 0.37 0.045 -p 0.035 0.61 0.026 -p 0.005 0.75 0.06 -p 0.04 1. 0.025 -p 0.035 1.15 0.08 --sz 2 --steps 2000 --walker-factor 2 --rew --z-lim 0

# the same, but starting from Guzman+18 parameters, and a Gaussian at the center
# this has problems converging
vis-r -v ~/tmp/dsharp/AS209/visibilities/AS209_continuum.8ch.npy -o as209_g18init -g 0.0017 -0.0031 85.7 34.9 --inner 0.006 0.028 -p 0.026 0.125 0.0261 -p 0.04 0.224 0.0412 -p 0.07 0.342 0.0611 -p 0.033 0.612 0.0258 -p 0.007 0.758 0.0822 -p 0.043 0.995 0.0346 -p 0.033 1.149 0.0812 --sz 2 --steps 1000 --walker-factor 2 --rew --z-lim 0

# try the MPOL solution from
# https://github.com/MPoL-dev/examples/blob/main/AS209-pyro-inference/pyro.ipynb
vis-r -v ~/tmp/dsharp/AS209/visibilities/AS209_continuum.8ch.npy -o as209_mpol -g 0.00182 -0.0031 85.8 33.5 --inner 0.006 0.028 -p 0.026 0.121 0.019 -p 0.04 0.230 0.099 -p 0.07 0.374 0.0417 -p 0.033 0.610 0.025 -p 0.007 0.787 0.066 -p 0.043 0.985 0.028 -p 0.033 1.095 0.11 --sz 2 --steps 1000 --walker-factor 2 --rew --z-lim 0


## residual images
import alma.casa
alma.casa.residual('/Users/grant/tmp/dsharp/AS209/visibilities/AS209_continuum.split.8ch.ms/',
	vis_model='as209_flat/AS209_continuum.8ch-vismod.npy',ms_new='as209_flat/residuals/residuals.ms')

from casatasks import tclean, exportfits
tclean(vis='as209_flat/residuals/residuals.ms/',
	imagename='as209_flat/residuals/residuals',
	cell='0.05arcsec', imsize=[1024,1024],
	interactive=False, niter=0,
	robust=1, weighting='briggs')

exportfits(imagename='as209_flat/residuals/residuals.image/',
	fitsimage='as209_flat/residuals/residuals.fits')