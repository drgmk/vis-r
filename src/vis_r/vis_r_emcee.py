"""Script for fitting with emcee.

For reasons I don't understand, but possibly related to global variables, this script runs
about 5x faster as a script compared to as a function via an entry point. It's still set up
to be run as a command line executable, but is set up by vis_r_emcee_main.py which has the
entry point.
"""

import os
import sys
import argparse
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import multiprocess as mp  # for running as script
import emcee
import corner
import arviz as az

from vis_r import functions
from vis_r import vis_r_stan


# pass args on and run the stan version
if '--stan' in sys.argv:
    vis_r_stan.vis_r_stan_radial(sys.argv)
    exit()

# setup
parser = argparse.ArgumentParser(description='vis-r with emcee')
parser = functions.add_default_parser(parser)
# emcee specific args
parser.add_argument('--astrom', dest='astrom', action='store_true', default=False,
                    help="Fit offsets for each input file")
parser.add_argument('--surf-dens', dest='surf_dens', action='store_true', default=False,
                    help="Include 1/sqrt(r) temp. effect")
parser.add_argument('--inner', dest='inner', metavar=('flux', 'sigma'),
                    type=float, nargs=2, help='Partially resolved (Gaussian) inner source')
parser.add_argument('--minimise', dest='minimize', action='store_true', default=False,
                    help="Attempt initial minimisation")
parser.add_argument('--walker-factor', dest='walk', metavar='4', type=float, default=4,
                    help='walkers = factor * parameters for emcee')
parser.add_argument('--restore', dest='restore', action='store_true', default=False,
                    help="Restore walkers from prior run")
parser.add_argument('--prune', dest='prune', action='store_true', default=False,
                    help="Prune restored walkers to median")
args = parser.parse_args()

if args.prune and not args.restore:
    exit('Only set prune if restoring.')

# set up initial parameters, start with geometry
if args.astrom:
    n_off = len(args.visfiles)
    if n_off == 1:
        print(' Turning astrom off, fitting only one file')
        args.astrom = False
else:
    n_off = 1

n_off2 = 2*n_off

# expand if only one set of offsets given
if len(args.g) != n_off2+2:
    geom = np.tile(args.g[:-2], n_off)
    geom = np.append(geom, args.g[-2:])
else:
    geom = args.g

params = []
for i in range(n_off):
    params.append(f'$\\Delta \\alpha[{i}]$')
    params.append(f'$\\Delta \\delta[{i}]$')

params += ['$\\phi$', '$i$']

# don't allow exactly zero offset
for i in range(n_off2):
    if geom[i] == 0:
        geom[i] = 0.001

# pick a radial profile
inits = np.append(geom, args.p)
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
    params_ = ['$F$', '$r_{in}$', '$a_{disk}$', '$\\sigma_{in}$', '$r_{out}$', '$\\sigma_{out}$']
    r_prof = functions.r_prof_erf2_power
elif args.type == 'erf2_power_ggap':
    params_ = ['$F$', '$r_{in}$', '$a_{disk}$', '$\\sigma_{in}$', '$r_{out}$', '$\\sigma_{out}$',
               '$d_{gap}$', '$r_{gap}$', '$\\sigma_{gap}$']
    r_prof = functions.r_prof_erf2_power_ggap
elif args.type == 'erf2_power_ggap2':
    params_ = ['$F$', '$r_{in}$', '$a_{disk}$', '$\\sigma_{in}$', '$r_{out}$', '$\\sigma_{out}$',
               '$d_{gap1}$', '$r_{gap1}$', '$\\sigma_{gap1}$', '$d_{gap2}$', '$r_{gap2}$', '$\\sigma_{gap2}$']
    r_prof = functions.r_prof_erf2_power_ggap2
else:
    exit(f'Radial model {args.type} not known.')

if args.zlim > 0:
    params_ += ['$\\sigma_z$']

nrp = len(params_)
if nr > 1:
    for i in range(nr):
        for p in params_:
            params += [f'{p}[{i}]']
else:
    params += params_

i = len(inits)
istar = iin = ibg = ipt = 0
if args.star:
    istar = i
    i += 1
    inits = np.append(inits, args.star[0])
    params += ['$F_\\star$']

if args.inner:
    iin = i
    i += 1
    inits = np.append(inits, args.inner)
    params += ['$F_{in}$', '$\\sigma_{in}$']

if args.bg:
    ibg = i
    nbg = len(args.bg)
    i += 6*nbg
    inits = np.append(inits, args.bg)
    params_ = ['$\\alpha_{bg}$', '$\\delta_{bg}$', '$F_{bg}$', '$\\sigma_{bg}$', '$\\phi_{bg}$', '$i_{bg}$']
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

if len(inits) < n_param:
    exit(f'Fewer parameters given than required, did you forget z?\n{inits}\n{params}')

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
              'r': [0.001, np.inf],
              'phi': [-180, 180],
              'i': [0, 90],
              'a_in': [0, 30],
              'a_out': [-30, 0],
              'gamma': [0, 20],
              'sigma_r': [0.001, np.inf],
              'sigma_z': [0, np.inf],  # limit set by prior in lnprob
              'd': [0, 1]
              }

# adjust if modelling surface density
if args.surf_dens:
    all_limits['a_in'] = [0.5, 50]
    all_limits['a_out'] = [-50, 0.5]

all_limits['r_in'] = all_limits['r_out'] = all_limits['r']
all_limits['r_gap'] = all_limits['r_gap1'] = all_limits['r_gap2'] = all_limits['r']
all_limits['F_star'] = all_limits['F_bg'] = all_limits['F_pt'] = all_limits['F_in'] = all_limits['F']
all_limits['sigma_in'] = all_limits['sigma_out'] = all_limits['sigma_bg'] = all_limits['sigma_r']
all_limits['sigma_gap'] = all_limits['sigma_gap1'] = all_limits['sigma_gap2'] = all_limits['sigma_r']
all_limits['phi_bg'] = all_limits['phi']
all_limits['i_bg'] = all_limits['i']
all_limits['d_gap'] = all_limits['d_gap1'] = all_limits['d_gap2'] = all_limits['d']

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
    outdir = f'{relpath}/{args.outrel.rstrip()}/vis-r-emc_{nr}{args.type}'
    if args.star:
        outdir += '_star'
    if args.bg:
        outdir += f'_{nbg}bg'
    if args.pt:
        outdir += f'_{npt}pt'
    if args.surf_dens:
        outdir += '_surfdens'

if not os.path.exists(outdir):
    try:
        os.mkdir(outdir)
    except FileNotFoundError:
        print(f'Output dir could not be created, check path to\n{outdir}\nexists')
        exit()

# load data
u, v, re, im, w, n_uv = functions.load_data(args, astrom=args.astrom)

# setup DHT
arcsec = np.pi/180/3600
arcsec2pi = arcsec*2*np.pi
Rnk, Qzero, Ykm = functions.setup_dht(args.sz, u, v)

def lnprob(p, model=False):
    """Return ln probability of model.

    Parameters
    ----------
    p : list
        List of parameters
    model : bool, optional
        Return rot, ruv, vis, sb for plotting/testing instead of ln(prob)

    Uses Hankel transform for radial profile, and analytic models for
    other components. Equations for those can be found for example in
    table 10.2 of 2017isra.book.....T
    """

    if np.any(p < limits[:, 0]) or np.any(p > limits[:, 1]):
        return -np.inf

    # u,v rotation
    urot, ruv = functions.uv_trans(u, v, np.deg2rad(p[n_off2]), np.deg2rad(p[n_off2+1]))

    rp = p[n_off2+2:n_off2+2+nrp*nr].reshape((nr, -1))
    rz_part = np.sin(np.deg2rad(p[n_off2+1])) * urot * arcsec2pi

    # radial profile, loop over components
    vis = np.zeros(len(ruv), dtype=complex)
    sb = np.zeros(len(Rnk))  # sb is not really sb
    for i in range(nr):
        f = 1/2.35e-11*r_prof(Rnk/arcsec, rp[i, 1:])
        if args.surf_dens:
            f /= np.sqrt(Rnk/arcsec)
        fth = np.dot(Ykm, f)  # same as fth = h.transform(f, q=Qzero)
        # normalise on shortest (zero) baseline
        fth = fth * rp[i, 0] / fth[0]
        sb += rp[i, 0] * f

        # interpolate
        vis_ = np.interp(ruv, Qzero, fth)

        # vertical structure
        if args.zlim > 0:
            rz = rp[i, -1] * rp[i, 1] * rz_part
            vis += vis_ * np.exp(-0.5*np.square(rz))
        else:
            vis += vis_

    # via matrices, this is actually quite a bit slower
    # perhaps because we are already burning all CPU
    # f = np.zeros((nhpt, nr))
    # for i in range(nr):
    #     f[:, i] = r_prof(Rnk/arcsec, rp[i, 1:])
    #
    # fth = np.dot(Ykm, f)  # fth is N x nr
    # fth = fth * rp[:, 0] / fth[0]
    # sb = np.sum(rp[:, 0] * f, axis=1)
    # interp = interp1d(Qzero, fth.T, bounds_error=False, fill_value=fth[0])
    # vis_ = interp(ruv)  # nr x N
    # rz = np.outer(rp[:, -1] * rp[:, 1], rz_part)  # nr x N
    # vis = np.sum(vis_ * np.exp(-0.5*np.square(rz)), axis=0)  # N

    # partially resolved inner disk
    if args.inner:
        pin = p[iin:iin+2]
        vis += pin[0] * np.exp(-0.5*np.square(pin[1]*ruv*arcsec2pi))

    # star (before shift, i.e. assuming disk is star-centered)
    if args.star:
        vis += p[istar]

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
            vis_ = bgp[i, 2] * np.exp(-0.5*np.square(bgp[i, 3]*ruv*arcsec2pi))
            rot = bgp[i, 0] * u + bgp[i, 1] * v
            vis += vis_ * np.exp(1j*rot*arcsec2pi)

    # phase shift, for astrom every vis gets its
    # own shift, but same applied to all otherwise
    if args.astrom:
        shift_u = np.repeat(p[0:n_off2:2], n_uv)
        shift_v = np.repeat(p[1:n_off2:2], n_uv)
        rot = (u*shift_u + v*shift_v)*arcsec2pi
    else:
        rot = (u*p[0] + v*p[1])*arcsec2pi

    vis = vis * np.exp(1j*rot)

    # return model
    if model:
        return rot, ruv, vis, sb

    # chi^2
    chi2 = -0.5 * np.sum(((re-vis.real)**2.0 + (im-vis.imag)**2.0) * w)

    # priors (z prior for consistency with stan version)
    chi2 += -0.5 * np.sum(np.square(rp[:, -1]/args.zlim))

    if not np.isfinite(chi2):
        print(f'non-finite chi2 with parameters\n{p}')
        return -np.inf

    return chi2


# mcmc setup
nwalkers = int(args.walk*ndim)
savefile = f'{outdir}/vismod.h5'
backend = emcee.backends.HDFBackend(savefile)

if os.path.exists(savefile) and args.restore:
    print(f'\nIgnoring input param values, restoring from previous run')
    pos = backend.get_last_sample().coords
    p0 = np.median(pos, axis=0)
    if args.prune:
        print(' pruning parameters to median of last step + randomness')
        pos = [p0 + p0*0.01*np.random.randn(ndim) for i in range(nwalkers)]
else:

    # attempt initial minimisation
    if args.minimize:
        nlnprob = lambda x: -lnprob(x)
        fit = minimize(nlnprob, p0, method='Nelder-Mead',
                       options={'maxiter': 10000})
        print(f"Initial minimisation: {fit['message']}")
        p0 = fit['x']
        print(f' minimised ln(p) {lnprob(p0)}')

    pos = [p0 + p0*0.01*np.random.randn(ndim) for i in range(nwalkers)]
    backend.reset(nwalkers, ndim)

print('\nFitting parameters (name, initial value, lo/hi limits)')
print(f' model is: {args.type}')
print(  '------------------------------------------------------')
functions.pprint((range(len(p0)), params, p0, limits[:, 0], limits[:, 1]))

test = lnprob(p0)
print(f'\nInitial ln(prob) {test}\n')
if not np.isfinite(test):
    exit('Initial probability not finite')

# run mcmc fitting
nsteps = args.steps
if args.threads:
    nthreads = args.threads
else:
    nthreads = os.cpu_count()

print(f'Running emcee with {nwalkers} walkers on {nthreads} threads for {nsteps} steps')

# do the MCMC
with mp.Pool(nthreads) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=backend)
    pos, prob, state = sampler.run_mcmc(pos, nsteps, progress=True)

burn = backend.iteration - args.keep
fit = az.from_emcee(sampler, var_names=params)
fit = fit.sel(draw=slice(burn, None))
df = az.summary(fit)
df = df.drop(['mcse_mean', 'hdi_3%', 'hdi_97%', 'mcse_sd', 'ess_tail'], axis=1)
print(df)

print(f'Output in {outdir}')

# see what the chains look like, skip a burn in period
fig, ax = plt.subplots(ndim+1, 2, figsize=(10, int(0.75*ndim)), sharex='col', sharey=False)

# get data (from h5 save file) once, much quicker
# than doing it for every walker
probdata = sampler.lnprobability
plotdata = sampler.chain[:, :burn, :]
for j in range(nwalkers):
    ax[-1, 0].plot(probdata[j, :burn])
    ax[-1, 0].set_ylabel('ln(prob)', rotation=0, va='center')
    for i in range(ndim):
        ax[i, 0].plot(plotdata[j, :, i])
        ax[i, 0].set_ylabel(params[i], rotation=0, va='center')

# plot all post-burn, as there won't be too many
plotdata = sampler.chain[:, burn:, :]
for j in range(nwalkers):
    ax[-1, 1].plot(probdata[j, burn:])
    for i in range(ndim):
        ax[i, 1].plot(plotdata[j, :, i])

ax[-1, 0].set_xlabel('burn in')
ax[-1, 1].set_xlabel('sampling')
fig.subplots_adjust(hspace=0.1, top=0.99, right=0.98, bottom=0.05)
fig.align_ylabels(ax[:, 0])
fig.savefig(f'{outdir}/chains.png', dpi=150)

# save chains as numpy
if args.save_chains:
    np.save(f'{outdir}/chains.npy', sampler.chain)

p = np.median(sampler.chain[:, burn:, :].reshape((-1, ndim)), axis=0)
s = np.std(sampler.chain[:, burn:, :].reshape((-1, ndim)), axis=0)

if args.input_model:
    print('Ignoring fitting and using initial parameters for save')
    p = p0
    s = np.zeros_like(p)

np.save(f'{outdir}/best_params.npy', np.vstack((params, p, s)))

# save model and visibilities
if args.save_model:

    # change some things since we will run lnprob for a single dataset
    # but it was possibly being run for multiple earlier
    limits = np.inf * np.vstack((-1*np.ones(n_param-n_off2+2),
                                 np.ones(n_param-n_off2+2))).T
    iin -= n_off2-2
    istar -= n_off2-2
    ipt -= n_off2-2
    ibg -= n_off2-2
    n_off_orig = n_off
    n_off = 1
    n_off2 = 2

    for i,f in enumerate(args.visfiles):
        print(f' saving model for {os.path.basename(f)}')
        u, v, re, im, w = functions.read_vis(f)
        n_uv = len(u)
        if args.astrom:
            p_ = np.append([p[2*i], p[2*i+1]], p[2*n_off_orig:])
        else:
            p_ = p
        _, _, vis, _ = lnprob(p_, model=True)
        f_ = os.path.splitext(os.path.basename(f))
        f_save = f_[0] + '-vismod' + f_[1]
        np.save(f'{outdir}/{f_save}', vis)

# make a corner plot (may run out of memory for many parameters so do last)
print('Corner plot')
fig = corner.corner(plotdata.reshape((-1, ndim)), labels=params, show_titles=True)
fig.savefig(f'{outdir}/corner.pdf')
