import numpy as np
from scipy.stats import binned_statistic_2d


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


def bin_uv(u_, v_, re_, im_, w_, size_arcsec=None):
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
    print(f'uv bin: {binsz:.2f}')

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

    print(f' original nvis:{len(u_)}')
    print(f' binned nvis:{len(u)}')

    return u, v, re, im, w


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
    return np.exp(-0.5*np.square((r - par[0])/par[1]))

def r_prof_power(r, par):
    return 1/((r/par[0])**(-par[3]*par[1]) + (r/par[0])**(-par[3]*par[2]))**(1/par[3])