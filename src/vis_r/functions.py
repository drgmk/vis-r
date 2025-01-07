import hashlib
import os
import numpy as np
from scipy.stats import binned_statistic, binned_statistic_2d
from scipy.special import erfc
from scipy.special import jn_zeros
import frank


def pprint(cols):
    """Print some columns nicely.
    https://stackoverflow.com/questions/9989334/create-nice-column-output-in-python
    """

    po = np.get_printoptions()
    np.set_printoptions(precision=3)

    rows = []
    for i in range(len(cols[0])):
        # rows.append([str(c[i]) for c in cols])  # no dps for floats
        tmp = []
        for c in cols:
            if isinstance(c[i], (float, np.floating)):
                tmp.append(f'{c[i]:.3g}')
            else:
                tmp.append(str(c[i]))

        rows.append(tmp)

    widths = [max(map(len, col)) for col in zip(*rows)]
    for row in rows:
        print("  ".join((val.ljust(width) for val, width in zip(row, widths))))

    np.set_printoptions(**po)


def update_stanfile(code, file):
    """Update stan file only if changed."""
    hasher = hashlib.md5()
    hasher.update(code.encode())
    newhash = hasher.hexdigest()

    hasher = hashlib.md5()
    oldhash = ''
    if os.path.exists(file):
        with open(file) as f:
            hasher.update(f.read().encode())
            oldhash = hasher.hexdigest()

    if newhash != oldhash:
        with open(file, 'w') as f:
            f.write(code)


def add_default_parser(parser):
    """Add parser arguments common to emcee/stan."""
    parser.add_argument('--stan', dest='stan', action='store_true', default=True,
                        help="Run stan version instead of emcee")
    parser.add_argument('-v', dest='visfiles', metavar=('vis1.npy', 'vis2.npy'), nargs='+', required=True,
                        help='Visibility files (u, v, re, im, w, wav, file)')
    parser.add_argument('-t', dest='type', metavar='gauss', default='gauss',
                        help='Model type (power[6], gauss[4])')
    parser.add_argument('-g', dest='g', type=float, nargs=4, required=True,
                        metavar=('dra', 'ddec', 'pa', 'inc'),
                        help='Geometry parameters')
    parser.add_argument('-p', dest='p', type=float, action='append', required=True, nargs='+',
                        metavar='norm r ... zh',
                        help='Radial component model parameters')
    parser.add_argument('--z-lim', dest='zlim', metavar='zlim', type=float, default=0.2,
                        help='1sigma upper prior on z/r_0')
    parser.add_argument('--star', dest='star', metavar='flux',
                        type=float, nargs=1, help='Point source at disk center')
    parser.add_argument('--bg', dest='bg', metavar=('dra', 'ddec', 'f', 'r', 'pa', 'inc'), action='append',
                        type=float, nargs=6, help='Resolved background sources')
    parser.add_argument('--pt', dest='pt', metavar=('dra', 'ddec', 'f'), action='append',
                        type=float, nargs=3, help='Unresolved background sources')
    parser.add_argument('--out-rel', dest='outrel',  type=str, default='.',
                        metavar='./',
                        help='Path to output relative to first data file')
    parser.add_argument('-o', dest='outdir',  type=str, default=None,
                        help='Path to output (override --out-rel)')
    parser.add_argument('--sz', dest='sz', metavar='8.84', type=float, default=8.84,
                        help='Radius (arcsec) for uv binning')
    parser.add_argument('--rew', dest='reweight', action='store_true', default=False,
                        help="Reweight visibilities for chi^2(no model)=1")
    parser.add_argument('--threads', dest='threads', metavar='6', type=int, default=6,
                        help='Number of threads to run on')
    parser.add_argument('--steps', dest='steps', metavar='1400', type=int, default=1400,
                        help='Burn in/warmup steps for emcee/stan')
    parser.add_argument('--keep', dest='keep', metavar='400', type=int, default=400,
                        help='Posterior sampling steps for emcee/stan')
    parser.add_argument('--no-model', dest='save_model', action='store_false', default=True,
                        help="Don't save model")
    parser.add_argument('--save-chains', dest='save_chains', action='store_true', default=False,
                        help="Export model chains as numpy")
    parser.add_argument('--input-model', dest='input_model', action='store_true', default=False,
                        help="Use input parameters for model")
    return parser


def read_vis(f):
    """Read visibilities in a few niche formats."""
    if '.npy' in f:
        # my format, u/v already in wavelengths
        tmp = np.load(f, allow_pickle=True)
        if len(tmp) > 5:
            u, v, re, im, w, wavelength_, ms_file_ = tmp
        else:
            u, v, re, im, w = tmp
    elif '.txt' in f:
        # 3 comment lines, line 2 is average wave in m
        # then u[m], v[m], Re, Im, w (5 lines)
        # or u[lambda], v[lambda], Re, Im, w, wavelength (6 lines)
        tmp = np.loadtxt(f, comments='#')
        if tmp.shape[1] == 5:
            fh = open(f)
            lines = fh.readlines()
            wavelength_ = float(lines[1].strip().split(' ')[-1])
            u, v, re, im, w = tmp.T
            u /= wavelength_
            v /= wavelength_
        elif tmp.shape[1] == 6:
            u, v, re, im, w, wavelength_ = tmp.T
            # u /= wavelength_
            # v /= wavelength_
        else:
            exit('text file has neither 5 nor 6 columns')
    else:
        exit('file type not txt or npy')

    return u, v, re, im, w


def _get_duv(R, wav, D):
    """Equivalent to get_duv below"""
    return 1/0.6 * D/wav * np.sqrt(1/R**2 - 1)


def get_duv(r=0.99, size_arcsec=None):
    """Return u,v cell size for binning.

    Parameters
    ----------
    r: float, optional
        Flux loss for point source at size_arcsec
    size_arcsec: float, optional
        How far from phase center pt. src. will lose (1-r)% flux.
    """
    if size_arcsec is None:
        size_arcsec = 8.84
    return 1/(size_arcsec/3600*np.pi/180) * np.sqrt(1/r**2 - 1)


def bin_uv(u_, v_, re_, im_, w_, size_arcsec=None, verb=True):
    """Return binned visibilities.

    Parameters
    ----------
    u_, v_, re_, im_, w_: arrays of floar
        u,v baselines, real and imaginary visibilities, and weights.
    size_arcsec: float, optional
        Passed to get_duv to get bin size.
    """

    if size_arcsec == 0:
        print('no binning')
        return u_, v_, re_, im_, w_

    uneg = u_ < 0
    u_ = np.abs(u_)
    v_[uneg] = -v_[uneg]
    im_[uneg] *= -1

    binsz = get_duv(size_arcsec=size_arcsec)
    if verb:
        print(f'u,v bin: {binsz:.0f}')

    bins = [int(np.max(np.abs(v_))/binsz), int(np.max(np.abs(u_))/binsz)*2]

    u,  _, _, _ = binned_statistic_2d(u_, v_, u_*w_, statistic='sum', bins=bins)
    v,  _, _, _ = binned_statistic_2d(u_, v_, v_*w_, statistic='sum', bins=bins)
    re, _, _, _ = binned_statistic_2d(u_, v_, re_*w_, statistic='sum', bins=bins)
    im, _, _, _ = binned_statistic_2d(u_, v_, im_*w_, statistic='sum', bins=bins)
    w,  _, _, _ = binned_statistic_2d(u_, v_, w_, statistic='sum', bins=bins)

    # keep non-empty cells
    ok = w != 0
    u = (u[ok] / w[ok]).flatten()
    v = (v[ok] / w[ok]).flatten()
    re = (re[ok] / w[ok]).flatten()
    im = (im[ok] / w[ok]).flatten()
    w = w[ok].flatten()

    return u, v, re, im, w


def bin_ruv(ruv, vis, bins=100):
    v, edges, _ = binned_statistic(ruv, vis, statistic='mean', bins=bins)
    ok = np.isfinite(v)
    return ((edges[:-1] + edges[1:])/2)[ok], v[ok]


def load_data(args, astrom=False):
    # load data
    print(f'\nLoading data')
    u = v = re = im = w = np.array([])
    n_uv = []
    sum_uv = 0
    for i, f in enumerate(args.visfiles):
        u_, v_, re_, im_, w_ = read_vis(f)
        sum_uv += len(u_)
        print(f' {f} with nvis: {len(u_)}')

        reweight_factor = 2 * len(w_) / np.sum((re_**2.0 + im_**2.0) * w_)
        print(f' reweighting factor would be {reweight_factor}')
        if args.reweight:
            print(' applying reweighting')
            w_ *= reweight_factor

        if args.sz > 0 and astrom:
            nu = len(u_)
            u_, v_, re_, im_, w_ = bin_uv(u_, v_, re_, im_, w_, size_arcsec=args.sz)
            print(f" original nvis: {nu}, binned nvis: {len(u_)}")
            n_uv.append(len(u_))

        u = np.append(u, u_)
        v = np.append(v, v_)
        w = np.append(w, w_)
        re = np.append(re, re_)
        im = np.append(im, im_)

    if args.sz > 0 and not astrom:
        nu = len(u)
        print('')
        u, v, re, im, w = bin_uv(u, v, re, im, w, size_arcsec=args.sz)
        print(f" original nvis: {nu}, binned nvis: {len(u)}")
        n_uv = [len(u)]
    else:
        print(f"\nTotal nvis ({len(args.visfiles)} files): {sum_uv}, fitting nvis: {len(u)}")

    return u, v, re, im, w, n_uv


def setup_dht(sz, u, v, nhpt=300):
    # set up the DHT
    arcsec = np.pi/180/3600

    uvmax = np.max(np.sqrt(u**2 + v**2))
    uvmin = np.min(np.sqrt(u**2 + v**2))
    if sz > 0:
        uvmin_ = get_duv(size_arcsec=sz)
    else:
        uvmin_ = uvmin

    # set up q array
    nq = np.ceil((uvmax+uvmin_/2)/uvmin_)
    r_out = jn_zeros(0, nq+1)[-1] / (2*np.pi*uvmax) / arcsec
    Qzero = np.arange(nq) * uvmin_ + uvmin_/2
    Qzero = np.append(0, Qzero)

    # set up transform, pre-computing new matrix for transform
    h = frank.hankel.DiscreteHankelTransform(r_out*arcsec, nhpt)
    Rnk, Qnk = h.get_collocation_points(r_out*arcsec, nhpt)
    Ykm = h.coefficients(q=Qzero)

    print(f'\nNq: {nq+1}, Nhpt: {nhpt}')
    pprint(([' min/max q_k: ', ' min/max u,v:'],
            [f'{Qzero[1]:.0f}', f'{uvmin:.0f}'],
            [f'{Qzero[-1]:.0f}', f'{uvmax:.0f}']))

    return Rnk, Qzero, Ykm


def uv_trans(u, v, PA, inc, return_uv=False):
    """Return transformed and rotated u,v

    Parameters
    ----------
    u, v: arrays of float
        u,v baselines.
    PA: float
        Position angle in radians.
    inc: float
        Inclination in radians.
    return_uv: bool, optional
        Return rotated u,v.
    """
    cos_PA = np.cos(PA)
    sin_PA = np.sin(PA)
    urot = u * cos_PA - v * sin_PA
    vrot = u * sin_PA + v * cos_PA
    if return_uv:
        return urot, vrot
    ruv = np.hypot(urot*np.cos(inc), vrot)
    return urot, ruv


def r_prof_gauss(r, par):
    """Gaussian, par: r0, sigma_r"""
    return np.exp(-0.5*np.square((r - par[0])/par[1]))


def r_prof_power(r, par):
    """Double power-law, par: r0, a_i, a_o, gamma"""
    return 1/((r/par[0])**(-par[3]*par[1]) + (r/par[0])**(-par[3]*par[2]))**(1/par[3])


def r_prof_erf_power(r, par):
    """Inner erf and outer power, par: r0, sigma_i, a"""
    return erf_in(r, par[0], par[1]) * (r/par[0])**(par[2])


def r_prof_erf2_power(r, par):
    """Inner and outer erf with power, par: r_i, a, sigma_i, r_o, sigma_o"""
    return erf_in(r, par[0], par[2]) * erf_out(r, par[3], par[4]) * (r/par[0])**(par[1])


def r_prof_gauss2(r, par):
    """Gaussian, par: r0, sigma_i, sigma_o"""
    return np.piecewise(r, [r < par[0], r >= par[0]],
                        [lambda r_: r_prof_gauss(r_, [par[0], par[1]]),
                        lambda r_: r_prof_gauss(r_, [par[0], par[2]])])


def r_prof_erf2_power_ggap(r, par):
    """erf2_power with 1 Gaussian gap, par: r_i, a, sigma_i, r_o, sigma_o, d_gap, r_gap, sigma_gap"""
    disk = r_prof_erf2_power(r, par[:5])
    gap = r_prof_erf2_power(par[5], par[:5]) * par[6] * r_prof_gauss(r, par[6:])
    return disk - gap


def r_prof_erf2_power_ggap2(r, par):
    """erf2_power with 2 Gaussian gaps, par: r_i, a, sigma_i, r_o, sigma_o,
                                             d_gap1, r_gap1, sigma_gap1,
                                             d_gap2, r_gap2, sigma_gap2"""
    disk = r_prof_erf2_power(r, par[:5])
    gap1 = r_prof_erf2_power(par[6], par[:5]) * par[5] * r_prof_gauss(r, par[6:8])
    gap2 = r_prof_erf2_power(par[9], par[:5]) * par[8] * r_prof_gauss(r, par[9:])
    return disk - gap1 - gap2


def erf_in(r, r0, sigi):
    """Scaled error function, inner edge."""
    return 0.5 * erfc((r0-r)/(np.sqrt(2)*sigi*r0))


def erf_out(r, r0, sigo):
    """Scaled error function, outer edge."""
    return 0.5 * erfc((r-r0)/(np.sqrt(2)*sigo*r0))
