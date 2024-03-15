# minimal example script, this runs in less than 10s on an M2 Mac
import sys
import numpy as np
from scipy.special import jn_zeros
import matplotlib.pyplot as plt
import multiprocess as mp
import frank
import emcee
import vis_r

# initial parameters
params = ['dra', 'ddec', 'PA', 'inc', 'flux', 'r0', 'dr', 'dz']
if len(sys.argv) > 3:
    p0 = np.array(sys.argv[3:], dtype=float)
else:
    p0 = np.array([0.03, 0.03, 26.7, 76.7, 0.015, 1.08, 0.06, 0.04])

# get visibilities
if len(sys.argv) > 1:
    u, v, Re, Im, w = vis_r.read_vis(sys.argv[1])
else:
    uv_file = '../data/hr4796.selfcal.npy'
    u, v, Re, Im, w, wavelength, ms_file = np.load(uv_file, allow_pickle=True)

# estimate re-weighting factor
# so that chi^2 for null model would be 1, and d.o.f = 2*len(w)
reweight_factor = 2*len(w) / np.sum((Re**2.0 + Im**2.0) * w)
print('reweighting factor is {}'.format(reweight_factor))
w *= reweight_factor

# bin the data
if len(sys.argv) > 2:
    sz = float(sys.argv[2])
else:
    sz = 4

u, v, Re, Im, w = vis_r.bin_uv(u, v, Re, Im, w, size_arcsec=sz)

# set up the DHT
arcsec = np.pi/180/3600
arcsec2pi = arcsec*2*np.pi

uvmax = np.max(np.sqrt(u**2 + v**2))
uvmin = np.min(np.sqrt(u**2 + v**2))

fac = 1.5  # safety factor
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

# radial profile
def r_prof(r, par):
    return np.exp(-0.5*np.square((r - par[5])/par[6]))


def lnprob(p, test=False):

    # u,v rotation
    urot, ruv = vis_r.uv_trans(u, v, np.deg2rad(p[2]), np.deg2rad(p[3]))

    # radial profile
    f = 1/2.35e-11*r_prof(Rnk/arcsec, p)
    fth = h.transform(f)

    # normalise on shortest baseline
    fth = fth * p[4]/fth[0]

    # interpolate, frank has a method for this too
    # but it is about 10x slower
    # vis = h.interpolate(fth, ruv, space='Fourier')
    vis = np.interp(ruv, Qnk, fth)

    # vertical structure
    rz = p[7]*p[5]*np.sin(np.deg2rad(p[3])) * urot * arcsec2pi
    vis = vis * np.exp(-0.5*np.square(rz))

    # phase shift
    rot = (u*p[0] + v*p[1])*arcsec2pi
    vis = vis * np.exp(1j*rot)

    # chi^2
    chi2 = np.sum(((Re-vis.real)**2.0 + (Im-vis.imag)**2.0) * w)
    if test:
        return ruv, fth, vis

    return -0.5 * chi2


# set up and run mcmc fitting
ndim = len(p0)
nwalkers = 18
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
ruv, fth, vis = lnprob(p, test=True)
fig, ax = plt.subplots()
ax.scatter(ruv/1e6, Re, s=0.1)
ax.scatter(ruv/1e6, vis.real, s=0.1, color='yellow')
ax.set_xlabel('baseline / M$\\lambda$')
ax.set_ylabel('flux / Jy')
fig.tight_layout()
fig.savefig('example-vis.png')
