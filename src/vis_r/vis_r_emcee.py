"""Script for fitting with emcee.

For reasons I don't understand, but possibly related to global variables, this script runs
about 5x faster as a script compared to as a function via an entry point. It's still set up
to be run as a command line executable, but is set up by vis_r_emcee_main.py which has the
entry point.
"""

import os
import argparse
import numpy as np
from scipy.special import jn_zeros
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import multiprocess as mp  # for running as script
import frank
import emcee
import corner

from vis_r import functions


def pprint(cols):
    """Print some columns nicely.
    https://stackoverflow.com/questions/9989334/create-nice-column-output-in-python
    """
    rows = []
    for i in range(len(cols[0])):
        rows.append([str(c[i]) for c in cols])

    widths = [max(map(len, col)) for col in zip(*rows)]
    for row in rows:
        print("  ".join((val.ljust(width) for val, width in zip(row, widths))))


# setup
parser = argparse.ArgumentParser(description='vis-r with emcee')
parser.add_argument('-v', dest='visfiles', metavar=('vis1.npy', 'vis2.npy'), nargs='+', required=True,
                    help='Numpy save files (u, v, re, im, w, wav, file)')
parser.add_argument('-t', dest='type', metavar='gauss', default='gauss',
                    help='Model type (power[6], gauss[4])')
parser.add_argument('-g', dest='g', type=float, nargs=4, required=True,
                    metavar=('dra', 'ddec', 'pa', 'inc'),
                    help='Geometry parameters')
parser.add_argument('-p', dest='p', type=float, action='append', required=True, nargs='+',
                    metavar='norm r ... zh',
                    help='Radial component model parameters')
parser.add_argument('--out-rel', dest='outrel',  type=str, default='../models',
                    help='Path to output relative to first data file')
parser.add_argument('-o', dest='outdir',  type=str, default=None,
                    help='Folder for output')
parser.add_argument('--sz', dest='sz', metavar='8.84', type=float, default=8.84,
                    help='Radius (arcsec) for uv binning')
parser.add_argument('--star', dest='star', metavar='flux',
                    type=float, nargs=1, help='Point source at disk center')
parser.add_argument('--bg', dest='bg', metavar=('dra', 'ddec', 'f', 'r', 'pa', 'inc'), action='append',
                    type=float, nargs=6, help='Resolved background sources')
parser.add_argument('--pt', dest='pt', metavar=('dra', 'ddec', 'f'), action='append',
                    type=float, nargs=3, help='Unresolved background sources')
parser.add_argument('--rmax', dest='rmax', metavar='rmax', type=float, default=None,
                    help='Rmax for Hankel transform')
parser.add_argument('--z-prior', dest='zprior', metavar='0.2', type=float, default=0.2,
                    help='1sigma upper limit on z/r')
parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help="Plot initial model")
parser.add_argument('--rew', dest='reweight', action='store_true', default=False,
                    help="Reweight visibilities")
parser.add_argument('--min', dest='minimize', action='store_true', default=False,
                    help="Attempt initial minimisation")
parser.add_argument('--walker-factor', dest='walk', metavar='4', type=int, default=4,
                    help='walkers = factor * parameters for emcee')
parser.add_argument('--steps', dest='steps', metavar='1400', type=int, default=1400,
                    help='Number of total steps for emcee')
parser.add_argument('--keep', dest='keep', metavar='400', type=int, default=400,
                    help='Number of steps to keep for emcee')
parser.add_argument('--restore', dest='restore', action='store_true', default=False,
                    help="Restore walkers from prior run")
parser.add_argument('--threads', dest='threads', metavar='Ncpu', type=int, default=None,
                    help='Number of threads for emcee')

args = parser.parse_args()

# set up initial parameters, start with geometry
inits = np.append(args.g, args.p)
for i in [0, 1]:
    if inits[i] == 0:
        inits[i] = 0.001
params = ['$\\Delta \\alpha$', '$\\Delta \\delta$', '$\\phi$', '$i$']

# pick a radial profile
nr = len(args.p)
if args.type == 'power':
    params_ = ['$F$', '$r$', '$a_{in}$', '$a_{out}$', '$\\gamma$']
    r_prof = functions.r_prof_power
elif args.type == 'gauss':
    params_ = ['$F$', '$r$', '$\\sigma_r$']
    r_prof = functions.r_prof_gauss
elif args.type == 'gauss2':
    params_ = ['$F$', '$r$', '$\\sigma_{in}$', '$\\sigma_{out}$']
    r_prof = functions.r_prof_gauss2
elif args.type == 'erf_power':
    params_ = ['$F$', '$r$', '$\\sigma_{in}$', '$a_{out}$']
    r_prof = functions.r_prof_erf_power
elif args.type == 'erf2_power':
    params_ = ['$F$', '$r_{in}$', '$a_{out}$', '$\\sigma_{in}$', '$r_{in}$', '$\\sigma_{out}$']
    r_prof = functions.r_prof_erf2_power
else:
    exit(f'Radial model {args.type} not known.')

params_ += ['$\\sigma_z$']
nrp = len(params_)
if nr > 1:
    for i in range(nr):
        for p in params_:
            params += [f'{p}[{i}]']
else:
    params += params_

i = len(inits)
if args.star:
    istar = i
    i += 1
    inits = np.append(inits, args.star[0])
    params += ['$F_\\star$']

if args.bg:
    ibg = i
    nbg = len(args.bg)
    i += 6*nbg
    inits = np.append(inits, args.bg)
    params_ = ['$\\alpha_{bg}$', '$\\delta_{bg}$', '$F_{bg}$', '$r_{bg}$', '$\\phi_{bg}$', '$i_{bg}$']
    if nbg > 1:
        for i in range(nbg):
            for p in params_:
                params += [f'{p}[{i}]']
    else:
        params += params_

if args.pt:
    ipt = i
    npt = len(args.pt)
    i += 3*npt
    inits = np.append(inits, args.pt)
    params_ = ['$\\alpha_{pt}$', '$\\delta_{pt}$', '$F_{pt}$']
    if npt > 1:
        for i in range(npt):
            for p in params_:
                params += [f'{p}[{i}]']
    else:
        params += params_

n_param = len(params)

params_text = []
for p in params:
    p = p.split('[')[0]
    p = p.replace('Delta ', 'd_')
    p = p.replace('\\', '')
    p = p.replace('{', '')
    p = p.replace('}', '')
    p = p.replace('$', '')
    params_text.append(p)

# set up priors, limits for now
all_limits = {'F': [0, np.inf],
              'r': [0, np.inf],
              'phi': [-180, 180],
              'i': [0, 90],
              'a_in': [0, 50],
              'a_out': [-50, 0],
              'gamma': [0, 20],
              'sigma_r': [0, np.inf],
              'sigma_z': [0, args.zprior]
              }

all_limits['r_in'] = all_limits['r_out'] = all_limits['r_bg'] = all_limits['r']
all_limits['F_star'] = all_limits['F_bg'] = all_limits['F_pt'] = all_limits['F']
all_limits['sigma_in'] = all_limits['sigma_out'] = all_limits['sigma_r']
all_limits['phi_bg'] = all_limits['phi']
all_limits['i_bg'] = all_limits['i']

limits = np.zeros((n_param, 2))
for i, p in enumerate(params_text):
    if p in all_limits.keys():
        limits[i, :] = all_limits[p]
    else:
        limits[i, :] = [-np.inf, np.inf]

p0 = inits
ndim = len(p0)

# set up output directory
if args.outdir:
    outdir = args.outdir.rstrip()
else:
    relpath = os.path.dirname(args.visfiles[0])
    outdir = f'{relpath}/{args.outrel}/vis-r_{nr}{args.type}'
    if args.star:
        outdir += '_star'
    if args.bg:
        outdir += f'_{nbg}bg'
    if args.pt:
        outdir += f'_{npt}pt'

if not os.path.exists(outdir):
    os.mkdir(outdir)

# load data
print(f'Loading data')
u = v = re = im = w = np.array([])
for i, f in enumerate(args.visfiles):
    u_, v_, re_, im_, w_ = functions.read_vis(f)
    print(f' {f} with nvis: {len(u_)}')

    reweight_factor = 2 * len(w_) / np.sum((re_**2.0 + im_**2.0) * w_)
    print(f' reweighting factor would be {reweight_factor}')
    if args.reweight:
        print(' applying reweighting')
        w_ *= reweight_factor

    u = np.append(u, u_)
    v = np.append(v, v_)
    w = np.append(w, w_)
    re = np.append(re, re_)
    im = np.append(im, im_)

if args.sz > 0:
    nu = len(u)
    print('')
    u, v, re, im, w = functions.bin_uv(u, v, re, im, w, size_arcsec=args.sz)
    print(f" original nvis: {nu}, fitting nvis: {len(u)}")

# set up the DHT
arcsec = np.pi/180/3600
arcsec2pi = arcsec*2*np.pi

uvmax = np.max(np.sqrt(u**2 + v**2))
uvmin = np.min(np.sqrt(u**2 + v**2))

if args.rmax:
    r_max = args.rmax
else:
    # r_max = jn_zeros(0, 1)[0] / (2*np.pi*uvmin*np.cos(np.deg2rad(p0[3]))) / arcsec
    r_max = jn_zeros(0, 1)[0] / (2*np.pi*uvmin) / arcsec

# set based on probable primary beam HWHM
if r_max < 10:
    r_max = 10

# this is lazy, should do bisection
nhpt = 1
while True:
    q_tmp = jn_zeros(0, nhpt)[-1]
    if q_tmp > uvmax * 2*np.pi*r_max*arcsec:
        break
    nhpt += 1

h = frank.hankel.DiscreteHankelTransform(r_max*arcsec, nhpt)
Rnk, Qnk = h.get_collocation_points(r_max*arcsec, nhpt)

print(f'\nR_out: {r_max:.1f}, N: {nhpt}')
pprint(([' min/max q_k: ', ' min/max u,v:'],
        [f'{Qnk[0]:.0f}', f'{uvmin:.0f}'],
        [f'{Qnk[-1]:.0f}', f'{uvmax:.0f}']))


def lnprob(p, model=False):

    if np.any(p < limits[:, 0]) or np.any(p > limits[:, 1]):
        return -np.inf

    # u,v rotation
    urot, ruv = functions.uv_trans(u, v, np.deg2rad(p[2]), np.deg2rad(p[3]))

    # radial arrays, sb is not really sb
    vis = np.zeros(len(ruv))
    sb = np.zeros(len(Rnk))

    # radial profile, loop over components
    # (should do matrices as in stan implementation)
    rp = p[4:4+nrp*nr].reshape((nr, -1))
    rz_part = np.sin(np.deg2rad(p[3])) * urot * arcsec2pi
    for i in range(nr):
        f = 1/2.35e-11*r_prof(Rnk/arcsec, rp[i, 1:])
        fth = h.transform(f)
        # normalise on shortest baseline
        fth = fth * rp[i, 0] / fth[0]
        sb += rp[i, 0] * f

        # interpolate, interp sets values for ruv<Qnk[0] to Qnk[0]
        # which is the desired behaviour
        vis_ = np.interp(ruv, Qnk, fth)
        # frank has a method for this too but it is about 10x slower
        # vis_ = h.interpolate(fth, ruv, space='Fourier')

        # vertical structure
        rz = rp[i, -1] * rp[i, 1] * rz_part
        vis += vis_ * np.exp(-0.5*np.square(rz))

    # star (before shift, i.e. assuming disk is star-centered)
    if args.star:
        vis += p[istar]

    # phase shift
    rot = (u*p[0] + v*p[1])*arcsec2pi
    vis = vis * np.exp(1j*rot)

    # point background source, all at once
    if args.pt:
        ptp = p[ipt:ipt+npt*3].reshape((npt, -1))
        rot = ptp[:, 0][:, np.newaxis] * u + ptp[:, 1][:, np.newaxis] * v  # [npt x nvis]
        vis += np.inner(ptp[:, 2][:, np.newaxis].T, np.exp(1j*rot*arcsec2pi).T).squeeze()  # sum over npt

    # resolved background source, one at a time
    if args.bg:
        bgp = p[ibg:ibg+nbg*6].reshape((nbg, -1))
        for i in range(nbg):
            urot, ruv = functions.uv_trans(u, v, np.deg2rad(bgp[i, 4]), np.deg2rad(bgp[i, 5]))
            vis_ = bgp[i, 2] * np.exp(-0.5*np.square(bgp[i, 3])*ruv*arcsec2pi)
            rot = bgp[i, 0] * u + bgp[i, 1] * v
            vis += vis_ * np.exp(1j*rot*arcsec2pi)

    # return model
    if model:
        return rot, ruv, vis, sb

    # chi^2
    chi2 = -0.5 * np.sum(((re-vis.real)**2.0 + (im-vis.imag)**2.0) * w)

    # priors (using limits above instead)
    # chi2 += -0.5 * np.sum(np.square(rp[:, -1]/args.zprior))

    if not np.isfinite(chi2):
        print(f'non-finite chi2 with parameters\n{p}')
        return -np.inf

    return chi2


# mcmc setup
nwalkers = args.walk*ndim
savefile = f'{outdir}/vismod.h5'
backend = emcee.backends.HDFBackend(savefile)

if os.path.exists(savefile) and args.restore:
    print(f'\nIgnoring input param values, restoring from previous run')
    p0 = np.median(backend.get_chain(flat=True), axis=0)
    pos = backend.get_last_sample().coords
    # pos = np.load(f'{outdir}/chains.npy')[:, -1, :]
else:
    pos = [p0 + p0*0.01*np.random.randn(ndim) for i in range(nwalkers)]
    backend.reset(nwalkers, ndim)

print('\nFitting parameters (name, initial value, lo/hi limits)')
print(  '------------------------------------------------------')
pprint((range(len(p0)), params, p0, limits[:, 0], limits[:, 1]))
print('')

test = lnprob(p0)
print(f'\nInitial ln(prob) {test}\n')
if not np.isfinite(test):
    exit('Initial probability not finite')

# attempt initial minimisation
if args.minimize:
    nlnprob = lambda x: -lnprob(x)
    fit = minimize(nlnprob, p0, method='Nelder-Mead',
                   options={'maxiter': 10000})
    print(f"Initial minimisation: {fit['message']}")
    p0 = fit['x']
    print(f' minimised ln(p) {lnprob(p0)}')

# plot the initial model
if args.test:
    rot, ruv, vis, sb = lnprob(p0, model=True)
    srt = np.argsort(ruv)
    vis *= np.exp(-1j*rot)
    ruv_bin, vis_bin = functions.bin_ruv(ruv, re+1j*im, bins=nhpt)

    # visibilities
    # plt.scatter(ruv, re, s=0.2, label='data', alpha=0.5)
    plt.scatter(ruv_bin, vis_bin.real, s=1, label='binned', alpha=0.5)
    plt.semilogx(ruv[srt], vis.real[srt], 'r', label='model')
    plt.xlabel('baseline / $\\lambda$')
    plt.ylabel('real(visibility) / Jy')

    plt.legend()
    plt.show()
    exit()

# run mcmc fitting
nsteps = args.steps
if args.threads:
    nthreads = args.threads
else:
    nthreads = os.cpu_count()

print(f'Running emcee with {nwalkers} walkers on {nthreads} threads for {nsteps} steps')

# we are using emcee v3
with mp.Pool(nthreads) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=backend)
    pos, prob, state = sampler.run_mcmc(pos, nsteps, progress=True)

# see what the chains look like, skip a burn in period
print('Plotting chains')
burn = backend.iteration - args.keep
fig, ax = plt.subplots(ndim+1, 2, figsize=(10, int(0.75*ndim)), sharex='col', sharey=False)

# get data (from h5 save file) once, much quicker
probdata = sampler.lnprobability
plotdata = sampler.chain[:, :burn, :]
for j in range(nwalkers):
    # ax[-1, 0].plot(sampler.lnprobability[j, :burn:every])
    ax[-1, 0].plot(probdata[j, :burn])
    ax[-1, 0].set_ylabel('ln(prob)', rotation=0, va='center')
    for i in range(ndim):
        ax[i, 0].plot(plotdata[j, :, i])
        ax[i, 0].set_ylabel(params[i], rotation=0, va='center')

# plot all post-burn, as there won't be too many
plotdata = sampler.chain[:, burn:, :]
for j in range(nwalkers):
    # ax[-1, 1].plot(sampler.lnprobability[j, burn:])
    ax[-1, 1].plot(probdata[j, burn:])
    for i in range(ndim):
        ax[i, 1].plot(plotdata[j, :, i])

ax[-1, 0].set_xlabel('burn in')
ax[-1, 1].set_xlabel('sampling')
fig.subplots_adjust(hspace=0.1, top=0.99, right=0.98, bottom=0.05)
fig.align_ylabels(ax[:, 0])
fig.savefig(f'{outdir}/chains.png', dpi=150)

# save chains
np.save(f'{outdir}/chains.npy', sampler.chain)

# save model and visibilities
print('Saving')
p = np.median(sampler.chain[:, burn:, :].reshape((-1, ndim)), axis=0)
p25 = np.percentile(sampler.chain[:, burn:, :].reshape((-1, ndim)), 2.5, axis=0)
p97 = np.percentile(sampler.chain[:, burn:, :].reshape((-1, ndim)), 97.5, axis=0)
np.save(f'{outdir}/best_params.npy', np.vstack((params, p, p25, p97)))
for f in args.visfiles:
    print(f' model for {os.path.basename(f)}')
    u, v, re, im, w = functions.read_vis(f)
    _, _, vis, _ = lnprob(p, model=True)
    f_ = os.path.splitext(os.path.basename(f))
    f_save = f_[0] + '-vismod' + f_[1]
    np.save(f'{outdir}/{f_save}', vis)

# make a corner plot (may run out of memory for many parameters so do last)
print('Corner plot')
fig = corner.corner(plotdata.reshape((-1, ndim)), labels=params, show_titles=True)
fig.savefig(f'{outdir}/corner.pdf')
