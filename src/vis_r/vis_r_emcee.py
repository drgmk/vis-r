"""Script for fitting with emcee.

For reasons I don't understand, but possibly related to global variables, this script runs
about 5x faster as a script compared to as a function. It's still set up to be run as a
command line executable, but is set up by vis_r_emcee_main.py which has the entry point.
"""

import os
import argparse
import numpy as np
from scipy.special import jn_zeros
import matplotlib.pyplot as plt
import multiprocess as mp
import frank
import emcee

from vis_r import functions

# setup
parser = argparse.ArgumentParser(description='vis-r with emcee')
parser.add_argument('-v', dest='visfiles', metavar=('vis1.npy', 'vis2.npy'), nargs='+', required=True,
                    help='Numpy save files (u, v, re, im, w, wav, file)')
parser.add_argument('-t', dest='type', metavar='power', default='gauss',
                    help='Model type (power[6], gauss[4])')
parser.add_argument('-g', dest='g', type=float, nargs=4, required=True,
                    metavar=('dra', 'ddec', 'pa', 'inc'),
                    help='Geometry parameters')
parser.add_argument('-p', dest='p', type=float, action='append', required=True, nargs='+',
                    metavar='norm r ... zh',
                    help='Radial component model parameters')
parser.add_argument('-o', dest='outdir', metavar='./', type=str, default='./',
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
# parser.add_argument('--inc-lim', dest='inc_lim', action='store_true', default=False,
#                     help="Limit range of inclinations")
# parser.add_argument('--pa-lim', dest='pa_lim', action='store_true', default=False,
#                     help="limit range of position angles")
# parser.add_argument('--z-lim', dest='zlim', metavar='zlim', type=float, default=None,
#                     help='1sigma upper limit on z/r')
parser.add_argument('--rew', dest='reweight', action='store_true', default=False,
                    help="Reweight visibilities")
parser.add_argument('--no-save', dest='save', action='store_false', default=True,
                    help="Don't save model")
parser.add_argument('--save-chains', dest='save_chains', action='store_true', default=False,
                    help="Export model chains as numpy")

args = parser.parse_args()

outdir = args.outdir.rstrip()
if not os.path.exists(outdir):
    os.mkdir(outdir)

visfiles = args.visfiles

# set up initial parameters
inits = np.append(args.g, args.p)
nr = len(args.p)
params = ['dra', 'ddec', 'PA', 'inc']

if args.type == 'power':
    params_ = ['norm', 'r', 'ai', 'ao', 'gam', 'dz']
    r_prof = functions.r_prof_power
elif args.type == 'gauss':
    params_ = ['norm', 'r', 'dr', 'dz']
    r_prof = functions.r_prof_gauss

nrp = len(params_)
for i in range(nr):
    for p in params_:
        params += [f'{p}_{i}']

i = len(inits)
if args.star:
    istar = i
    i += 1
    inits = np.append(inits, args.star[0])
    params += ['fstar']

if args.bg:
    ibg = i
    nbg = len(args.bg)
    i += 6*nbg
    inits = np.append(inits, args.bg)
    params += ['bgx', 'bgy', 'bgn', 'bgr', 'bgpa', 'bgi']

if args.pt:
    ipt = i
    npt = len(args.pt)
    i += 3*npt
    inits = np.append(inits, args.pt)
    params += ['ptx', 'pty', 'ptn']

p0 = inits
for par, p in zip(params, p0):
    print(f'{par}\t {p}')

# load data
u = v = re = im = w = np.array([])
for i, f in enumerate(visfiles):
    u_, v_, re_, im_, w_ = functions.read_vis(f)
    print(f'loading: {f} with nvis: {len(u_)}')

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
    u, v, re, im, w = functions.bin_uv(u, v, re, im, w, size_arcsec=args.sz)

print(f" original nvis: {len(u)}, fitting nvis: {len(u)}")

# set up the DHT
arcsec = np.pi/180/3600
twopi = 2*np.pi
arcsec2pi = arcsec*twopi

uvmax = np.max(np.sqrt(u**2 + v**2))
uvmin = np.min(np.sqrt(u**2 + v**2))

fac = 1.5  # safety factor
if args.rmax:
    r_max = args.rmax
else:
    r_max = jn_zeros(0, 1)[0] / (2*np.pi*uvmin*np.cos(np.deg2rad(p0[3]))) / arcsec * fac

nhpt = 1
while True:
    q_tmp = jn_zeros(0, nhpt)[-1]
    if q_tmp > uvmax * 2*np.pi*r_max*arcsec:
        break
    nhpt += 1

h = frank.hankel.DiscreteHankelTransform(r_max*arcsec, nhpt)
Rnk, Qnk = h.get_collocation_points(r_max*arcsec, nhpt)

print(f'R_out: {r_max}, N: {nhpt}')
print(f'min/max q_k: {Qnk[0]}, {Qnk[-1]}')
print(f'min/max u,v: {uvmin}, {uvmax}')


def lnprob(p, model=False):

    # u,v rotation
    urot, ruv = functions.uv_trans(u, v, np.deg2rad(p[2]), np.deg2rad(p[3]))
    vis = np.zeros(len(ruv))

    # radial profile, loop over components
    # (should do matrices as in stan implementation)
    rp = p[4:4+nrp*nr].reshape((nr, -1))
    if np.min(rp) < 0:
        return -np.inf
    rz_part = np.sin(np.deg2rad(p[3])) * urot * arcsec2pi
    for i in range(nr):
        f = 1/2.35e-11*r_prof(Rnk/arcsec, rp[i, 1:])
        fth = h.transform(f)
        # normalise on shortest baseline
        fth = fth * rp[i, 0] / fth[0]

        # interpolate, frank has a method for this too
        # but it is about 10x slower
        # vis = h.interpolate(fth, ruv, space='Fourier')
        vis_ = np.interp(ruv, Qnk, fth)

        # vertical structure
        rz = rp[i, -1] * rp[i, 1] * rz_part
        vis += vis_ * np.exp(-0.5*np.square(rz))

    # star (before shift, i.e. assuming disk is star-centered)
    if args.star:
        if p[istar] < 0:
            return -np.inf
        vis += p[istar]

    # phase shift
    rot = (u*p[0] + v*p[1])*arcsec2pi
    vis = vis * np.exp(1j*rot)

    # point background source
    if args.pt:
        ptp = p[ipt:ipt+npt*3].reshape((npt, -1))
        if np.min(ptp[:, 2]) < 0:
            return -np.inf
        rot = ptp[:, 0][np.newaxis, :] * u + ptp[:, 1][np.newaxis, :] * v  # [npt x nvis]
        vis += np.inner(ptp[:, 2][np.newaxis, :].T, np.exp(1j*rot*arcsec2pi).T).squeeze()  # sum over npt

    # resolved background source
    if args.bg:
        bgp = p[ibg:ibg+nbg*6].reshape((nbg, -1))
        if np.min(bgp[:, 2:]) < 0 or np.max(bgp[:, 4:]) > 180:
            return -np.inf
        for i in range(nbg):
            urot, ruv = functions.uv_trans(u, v, np.deg2rad(bgp[i, 4]), np.deg2rad(bgp[i, 5]))
            vis_ = bgp[i, 2] * np.exp(-0.5*np.square(bgp[i, 3])*ruv*arcsec2pi)
            rot = bgp[i, 0] * u + bgp[i, 1] * v
            vis += vis_ * np.exp(1j*rot*arcsec2pi)

    # chi^2
    chi2 = np.sum(((re-vis.real)**2.0 + (im-vis.imag)**2.0) * w)
    if model:
        return ruv, fth, vis

    # if not np.isfinite(chi2):
    #     return -np.inf

    return -0.5 * chi2


print(f'initial log(p) {lnprob(p0)}')

# set up and run mcmc fitting
ndim = len(p0)
nwalkers = 2*ndim
nsteps = 1000
nthreads = 6

# we are using emcee v3
with mp.Pool(nthreads) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
    pos = [p0 + p0*0.01*np.random.randn(ndim) for i in range(nwalkers)]
    pos, prob, state = sampler.run_mcmc(pos, nsteps, progress=True)

# see what the chains look like, skip a burn in period
burn = nsteps - 200
fig, ax = plt.subplots(ndim+1, 2, figsize=(9.5, 5), sharex='col', sharey=False)

for j in range(nwalkers):
    ax[-1, 0].plot(sampler.lnprobability[j, :burn])
    for i in range(ndim):
        ax[i, 0].plot(sampler.chain[j, :burn, i])
        ax[i, 0].set_ylabel(params[i])

for j in range(nwalkers):
    ax[-1, 1].plot(sampler.lnprobability[j, burn:])
    for i in range(ndim):
        ax[i, 1].plot(sampler.chain[j, burn:, i])
        # ax[i, 1].set_ylabel(params[i])

ax[-1, 0].set_xlabel('burn in')
ax[-1, 1].set_xlabel('sampling')
fig.savefig('example-chains.png')

p = np.median(sampler.chain[:, burn:, :].reshape((-1, ndim)), axis=0)
ruv, fth, vis = lnprob(p, model=True)
fig, ax = plt.subplots()
ax.scatter(ruv/1e6, re, s=0.1)
ax.scatter(ruv/1e6, vis.real, s=0.1, color='yellow')
ax.set_xlabel('baseline / M$\\lambda$')
ax.set_ylabel('flux / Jy')
fig.tight_layout()
fig.savefig('example-vis.png')

# save model visibilities
if args.save:
    for f in visfiles:
        print(f'saving model for {os.path.basename(f)}')
        u, v, re, im, w = functions.read_vis(f)
        _, _, vis = lnprob(p, model=True)
        f_ = os.path.splitext(os.path.basename(f))
        f_save = f_[0] + '-vismod' + f_[1]
        np.save(f"{outdir}/{f_save}", vis)
