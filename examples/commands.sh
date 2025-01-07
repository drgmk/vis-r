#!/bin/zsh

# examples, run these from the examples directory

# single Gaussian fit for emcee and stan
vis-r -v ../data/hr4796.selfcal.npy -o vis-r-emc_1gauss -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --walker-factor 4.5 --rew --sz 4 --save-chains
vis-r -v ../data/hr4796.selfcal.npy -o vis-r-stn_1gauss -g 0.03 0.03 0.464 1.335 -p 0.015 1.07 0.06 0.04 --rew --sz 4 --stan --log --save-chains

# timing, run stan once first to compile, imaging timing done by hand
/usr/bin/time -p -o vis-r-emc_1gauss/1as.txt vis-r -v ../data/hr4796.selfcal.npy -o vis-r-emc_1gauss_1as -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --walker-factor 4.5 --rew --sz 1 --save-chains
/usr/bin/time -p -o vis-r-emc_1gauss/2as.txt vis-r -v ../data/hr4796.selfcal.npy -o vis-r-emc_1gauss_2as -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --walker-factor 4.5 --rew --sz 2 --save-chains
/usr/bin/time -p -o vis-r-emc_1gauss/3as.txt vis-r -v ../data/hr4796.selfcal.npy -o vis-r-emc_1gauss_3as -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --walker-factor 4.5 --rew --sz 3 --save-chains
/usr/bin/time -p -o vis-r-emc_1gauss/4as.txt vis-r -v ../data/hr4796.selfcal.npy -o vis-r-emc_1gauss_4as -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --walker-factor 4.5 --rew --sz 4 --save-chains
/usr/bin/time -p -o vis-r-emc_1gauss/5as.txt vis-r -v ../data/hr4796.selfcal.npy -o tmp -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --walker-factor 4.5 --rew --sz 5
/usr/bin/time -p -o vis-r-emc_1gauss/6as.txt vis-r -v ../data/hr4796.selfcal.npy -o tmp -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --walker-factor 4.5 --rew --sz 6
/usr/bin/time -p -o vis-r-emc_1gauss/7as.txt vis-r -v ../data/hr4796.selfcal.npy -o tmp -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --walker-factor 4.5 --rew --sz 7
/usr/bin/time -p -o vis-r-emc_1gauss/8as.txt vis-r -v ../data/hr4796.selfcal.npy -o tmp -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --walker-factor 4.5 --rew --sz 8
/usr/bin/time -p -o vis-r-emc_1gauss/0as.txt vis-r -v ../data/hr4796.selfcal.npy -o tmp -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --walker-factor 4.5 --rew --sz 0

vis-r -v ../data/hr4796.selfcal.npy -o tmp -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --walker-factor 4.5 --rew --sz 8 --stan
/usr/bin/time -p -o vis-r-stn_1gauss/1as.txt vis-r -v ../data/hr4796.selfcal.npy -o tmp -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --rew --sz 1 --stan
/usr/bin/time -p -o vis-r-stn_1gauss/2as.txt vis-r -v ../data/hr4796.selfcal.npy -o tmp -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --rew --sz 2 --stan
/usr/bin/time -p -o vis-r-stn_1gauss/3as.txt vis-r -v ../data/hr4796.selfcal.npy -o tmp -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --rew --sz 3 --stan
/usr/bin/time -p -o vis-r-stn_1gauss/4as.txt vis-r -v ../data/hr4796.selfcal.npy -o tmp -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --rew --sz 4 --stan
/usr/bin/time -p -o vis-r-stn_1gauss/5as.txt vis-r -v ../data/hr4796.selfcal.npy -o tmp -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --rew --sz 5 --stan
/usr/bin/time -p -o vis-r-stn_1gauss/6as.txt vis-r -v ../data/hr4796.selfcal.npy -o tmp -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --rew --sz 6 --stan
/usr/bin/time -p -o vis-r-stn_1gauss/7as.txt vis-r -v ../data/hr4796.selfcal.npy -o tmp -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --rew --sz 7 --stan
/usr/bin/time -p -o vis-r-stn_1gauss/8as.txt vis-r -v ../data/hr4796.selfcal.npy -o tmp -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --rew --sz 8 --stan
# /usr/bin/time -p -o vis-r-stn_1gauss/0as.txt vis-r -v ../data/hr4796.selfcal.npy -o tmp -g 0.03 0.03 26 76 -p 0.015 1.07 0.06 0.04 --rew --sz 0 --stan

rm -r tmp

cd ../

# check that things are OK, run in vis-r directory
# pytest -v

cd examples

# in python/casa, make residual images
import alma.casa
from casatasks import tclean,exportfits
folder = 'examples/vis-r-emc_1gauss'
alma.casa.residual('/Users/grant/astro/data/alma/hr4796_c3/hr4796-avg.ms', 
    f'/Users/grant/astro/projects/vis-r/{folder}/hr4796.selfcal-vismod.npy')

tclean(vis=f'/Users/grant/astro/projects/vis-r/{folder}/hr4796.selfcal-vismod-residual.ms/',
    imagename=f'/Users/grant/astro/projects/vis-r/{folder}/residual',
    niter=0,interactive=False,imsize=[512,512],cell='0.05arcsec')

exportfits(f'/Users/grant/astro/projects/vis-r/{folder}/residual.image/',
    f'/Users/grant/astro/projects/vis-r/{folder}/residual.fits')
