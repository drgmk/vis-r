import shutil
import os
import argparse
import pickle
import numpy as np
from scipy.special import jn_zeros
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import arviz as az
import corner
import frank

from . import functions
from . import vis_r_stan_code

"""
Create M2 (arm-64) conda env with:

>CONDA_SUBDIR=osx-arm64 conda create -n stan python=3.11 ipython numpy scipy matplotlib
>conda activate stan
>conda env config vars set CONDA_SUBDIR=osx-arm64
>conda deactivate
>conda activate stan
>conda install -c conda-forge cmdstanpy corner arviz
>pip install https://github.com/drgmk/vis-r/archive/refs/heads/main.zip

To update stan to the latest version, which will likely be
more recent than in `conda` and will allow use of pathfinder
to find initial parameters and estimate posteriors:
```python
import cmdstanpy
cmdstanpy.install_cmdstan()
```
Then uncomment the code below to point `cmdstanpy` to this version:
"""
# import cmdstanpy
# import glob
# try:
#     vs = glob.glob(os.path.expanduser('~') + '/.cmdstan/*')
#     vs.sort()
#     cmdstanpy.set_cmdstan_path(vs[-1])
# except:
#     pass


def vis_r_stan_radial():

    # setup
    parser = argparse.ArgumentParser(description='stan implementation of vis-r')
    parser.add_argument('-v', dest='visfiles', metavar=('vis1.npy', 'vis2.npy'), nargs='+', required=True,
                        help='Numpy save files (u, v, re, im, w, wav, file)')
    parser.add_argument('-t', dest='type', metavar='power', default='power',
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
    parser.add_argument('--sc', dest='sc', metavar='1', type=float, default=1,
                        help='Scale parameters for std ~ 1')
    parser.add_argument('--norm-mul', dest='norm_mul', metavar='10', type=float, default=10,
                        help='Scaling for norm')
    parser.add_argument('--r-mul', dest='r_mul', metavar='1', type=float, default=1,
                        help='Scaling for radius')
    parser.add_argument('--star', dest='star', metavar='flux',
                        type=float, nargs=1, help='Point source at disk center')
    parser.add_argument('--bg', dest='bg', metavar=('dra', 'ddec', 'f', 'r', 'pa', 'inc'), action='append',
                        type=float, nargs=6, help='Resolved background sources')
    parser.add_argument('--pt', dest='pt', metavar=('dra', 'ddec', 'f'), action='append',
                        type=float, nargs=3, help='Unresolved background sources')
    parser.add_argument('-m', dest='metric', metavar='metric.pkl', type=str,
                        help='Pickled metric')
    parser.add_argument('--rmax', dest='rmax', metavar='rmax', type=float, default=None,
                        help='Rmax for Hankel transform')
    parser.add_argument('--inc-lim', dest='inc_lim', action='store_true', default=False,
                        help="Limit range of inclinations")
    parser.add_argument('--pa-lim', dest='pa_lim', action='store_true', default=False,
                        help="limit range of position angles")
    parser.add_argument('--z-lim', dest='zlim', metavar='zlim', type=float, default=None,
                        help='1sigma upper limit on z/r')
    parser.add_argument('--rew', dest='reweight', action='store_true', default=False,
                        help="Reweight visibilities")
    parser.add_argument('--no-save', dest='save', action='store_false', default=True,
                        help="Don't save model")
    parser.add_argument('--save-chains', dest='save_chains', action='store_true', default=False,
                        help="Export model chains as numpy")
    parser.add_argument('--pf', dest='pf', action='store_true', default=False,
                        help="Run pathfinder for initial posteriors")

    args = parser.parse_args()

    outdir = args.outdir.rstrip()
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    visfiles = args.visfiles

    # set up initial parameters
    inits = {'dra': args.g[0],
             'ddec': args.g[1],
             'pa': args.g[2],
             'inc': args.g[3]}

    p = np.array(args.p)
    inits['norm'] = p[:, 0]
    inits['r'] = p[:, 1]
    inits['zh'] = p[:, -1]
    if args.type == 'power':
        assert p.shape[1] == 6
        inits['ai'] = p[:, 2]
        inits['ao'] = p[:, 3]
        inits['gam'] = p[:, 4]
    elif args.type == 'erf_power':
        assert p.shape[1] == 5
        inits['sigi'] = p[:, 2]
        inits['ao'] = p[:, 3]
    elif args.type == 'erf_power':
        assert p.shape[1] == 7
        inits['ri'] = inits.pop('r')
        inits['ai'] = p[:, 2]
        inits['sigi'] = p[:, 3]
        inits['ro'] = p[:, 4]
        inits['ao'] = p[:, 5]
    elif args.type == 'gauss':
        assert p.shape[1] == 4
        inits['dr'] = p[:, 2]
    elif args.type == 'gauss2':
        assert p.shape[1] == 5
        inits['dri'] = p[:, 2]
        inits['dro'] = p[:, 3]

    if args.star:
        inits['star'] = args.star[0]

    if args.bg:
        bg = np.array(args.bg)
        inits['bgx'] = bg[:, 0]
        inits['bgy'] = bg[:, 1]
        inits['bgn'] = bg[:, 2]
        inits['bgr'] = bg[:, 3]
        inits['bgpa'] = bg[:, 4]
        inits['bgi'] = bg[:, 5]

    if args.pt:
        pt = np.array(args.pt)
        inits['ptx'] = pt[:, 0]
        inits['pty'] = pt[:, 1]
        inits['ptn'] = pt[:, 2]

    # scale parameters to have approx unit standard deviation
    sc = args.sc
    mul = {}
    for p in list(inits.keys()):
        mul[p] = sc
        if p in ['norm', 'bgn', 'ptn', 'star']:
            mul[p] *= args.norm_mul
        if p in ['bgpa', 'bgi']:
            mul[p] /= args.norm_mul
        if p in ['ai', 'ao', 'gam']:
            mul[p] /= args.norm_mul*10
        if p in ['zh'] and args.zlim:
            mul[p] = 1.0

    # use the std from a previous run if desired
    if args.metric:
        with open(args.metric, 'rb') as f:
            par = pickle.load(f)
            mul = pickle.load(f)
            std = pickle.load(f)
            # metric = pickle.load(f)

        for p in par:
            if '_' not in p:
                p_ = p.split('[')[0]
                if '[2]' not in p and '[3]' not in p:
                    mul[p_] = mul[p_] / std[p]

        print(f'scaling from previous std: {mul}')

    # set up parameter multipliers
    data = {'nr': len(inits['norm'])}
    for k in inits.keys():
        inits[k] *= mul[k]
        data[f'{k}_0'] = inits[k]
        data[f'{k}_mul'] = mul[k]

    data['nbg'] = 0
    if args.bg:
        data['nbg'] = len(bg)

    data['npt'] = 0
    if args.pt:
        data['npt'] = len(pt)

    # load data
    u_ = v_ = re_ = im_ = w_ = np.array([])
    for i, f in enumerate(visfiles):
        u, v, re, im, w = functions.read_vis(f)
        print(f'loading: {f} with nvis: {len(u)}')

        reweight_factor = 2 * len(w) / np.sum((re**2.0 + im**2.0) * w)
        print(f' reweighting factor would be {reweight_factor}')
        if args.reweight:
            print(' applying reweighting')
            w *= reweight_factor

        u_ = np.append(u_, u)
        v_ = np.append(v_, v)
        w_ = np.append(w_, w)
        re_ = np.append(re_, re)
        im_ = np.append(im_, im)

    if args.sz > 0:
        data['u'], data['v'], data['re'],  data['im'], data['w'] = \
            functions.bin_uv(u_, v_, re_, im_, w_, size_arcsec=args.sz)
    else:
        data['u'] = u_
        data['v'] = v_
        data['re'] = re_
        data['im'] = im_
        data['w'] = w_

    data['nvis'] = len(data['u'])
    data['sigma'] = 1/np.sqrt(data['w'])

    print(f" original nvis: {len(u_)}, fitting nvis: {data['nvis']}")

    arcsec = np.pi/180/3600
    uvmax = np.max(np.sqrt(data['u']**2 + data['v']**2))
    uvmin = np.min(np.sqrt(data['u']**2 + data['v']**2))
    # estimate lowest frequency given inclination, and include a safety factor
    fac = 1.5  # safety factor
    # r_max = jn_zeros(0, 1)[0] / (2*np.pi*uvmin*np.cos(inits['inc']/mul['inc'])) / arcsec * fac
    r_max = jn_zeros(0, 1)[0] / (2*np.pi*uvmin) / arcsec * fac

    nhpt = 1
    while True:
        q_tmp = jn_zeros(0, nhpt)[-1]
        if q_tmp > uvmax * 2*np.pi*r_max*arcsec:
            break
        nhpt += 1

    h = frank.hankel.DiscreteHankelTransform(r_max*arcsec, nhpt)
    Rnk, Qnk = h.get_collocation_points(r_max*arcsec, nhpt)

    data['nhpt'] = nhpt
    data['Ykm'] = h._Ykm
    data['Rnk'] = Rnk/arcsec
    data['Qnk'] = Qnk
    data['hnorm'] = 1/2.35e-11 * (2 * np.pi * (r_max*arcsec)**2) / h._j_nN

    print(f'Hankel points: {nhpt}')
    if 2*Qnk[0] > uvmin:
        print(f' WARNING: minimum Q not much smaller than minimum u,v')
        print(f'          potential problem for highly inclined disks')
    print(f' R_max: {r_max}')
    print(f' Q_min: {Qnk[0]}, uv_min: {uvmin}')
    print(f' Q_max: {Qnk[-1]}, uv_max: {uvmax}')

    # get stan code and compile
    code = vis_r_stan_code.get_code(args.type, gq=False,
                                    star=args.star is not None,
                                    bg=data['nbg'] > 0, pt=data['npt'] > 0,
                                    inc_lim=args.inc_lim, pa_lim=args.pa_lim,
                                    z_prior=args.zlim)

    stanfile = f'/tmp/alma{str(np.random.randint(100_000))}.stan'
    with open(stanfile, 'w') as f:
        f.write(code)

    model = CmdStanModel(stan_file=stanfile,
                         cpp_options={'STAN_THREADS': 'TRUE'})

    # print(model.exe_info())

    # initial run with pathfinder to estimate parameters and metric
    metric = 'dense'
    if args.pf:
        pf = model.pathfinder(data=data, inits=inits,
                              # show_console=True
                              )

        cn = pf.column_names
        ok = ['_' in c and '__' not in c for c in cn]
        fig = corner.corner(pf.draws()[:, ok],
                            titles=np.array(pf.column_names)[ok], show_titles=True)
        fig.savefig(f'{outdir}/corner_pf.pdf')

        ok = ['_' not in c for c in cn]
        metric = {'inv_metric': np.cov(pf.draws()[:, ok].T)}

        for k in inits.keys():
            med = np.median(pf.stan_variable(f'{k}_'), axis=0)
            std = np.std(pf.stan_variable(f'{k}_'), axis=0)
            data[f'{k}_mul'] = 1 / np.mean(std)
            inits[k] = med * data[f'{k}_mul']
            data[f'{k}_0'] = inits[k]
            # print(k, data[f'{k}_mul'], inits[k], (inits[k]/data[f'{k}_mul']))

    fit = model.sample(data=data, chains=6,
                       metric=metric,
                       iter_warmup=1000, iter_sampling=300,
                       inits=inits,
                       save_warmup=True,
                       show_console=False,
                       refresh=50)

    # shutil.copy(fit.metadata.cmdstan_config['profile_file'], outdir)
    with open(f'{outdir}/metric.pkl', 'wb') as f:
        pickle.dump(fit.column_names, f)
        pickle.dump(mul, f)
        pickle.dump(fit.summary()['StdDev'], f)
        pickle.dump(fit.metric, f)

    df = fit.summary(percentiles=(5, 95))
    print(df[df.index.str.contains('_') is False])
    # print(df.filter(regex='[a-z]_', axis=0))
    # print(fit.diagnose())

    xr = fit.draws_xr()
    for k in inits.keys():
        xr = xr.drop_vars(k)

    if args.save_chains:
        fit.save_csvfiles(outdir)
        np.save(f'{outdir}/chains.npy', xr.to_dataarray().as_numpy().squeeze())

    _ = az.plot_trace(xr)
    fig = _.ravel()[0].figure
    fig.tight_layout()
    fig.savefig(f'{outdir}/trace.pdf')

    best = {}
    for k in fit.stan_variables().keys():
        if '_' in k and '__' not in k:
            best[k] = np.median(fit.stan_variable(k), axis=0)
    print(f'best: {best}')

    # comment if memory problems ("python killed")
    fig = corner.corner(xr, show_titles=True)
    fig.savefig(f'{outdir}/corner.pdf')

    # save model visibilities
    if args.save:

        # save radial profiles
        code = vis_r_stan_code.get_code(args.type, gq='prof',
                                        star=args.star is not None,
                                        bg=data['nbg'] > 0, pt=data['npt'] > 0,
                                        inc_lim=args.inc_lim, pa_lim=args.pa_lim,
                                        z_prior=args.zlim)
        with open(stanfile, 'w') as f:
            f.write(code)

        model = CmdStanModel(stan_file=stanfile,
                             cpp_options={'STAN_THREADS': 'TRUE'})

        gq = model.generate_quantities(data=data, previous_fit=fit)
        prof = gq.stan_variables()['f']
        np.save(f"{outdir}/profile_r.npy", Rnk/arcsec)
        np.save(f"{outdir}/profile_f.npy", prof)

        fig, ax = plt.subplots()
        for i in range(prof.shape[-1]):
            for n in range(100):
                ax.plot(Rnk/arcsec, prof[np.random.default_rng().integers(prof.shape[0]), :, i])
        ax.set_xlabel('radius / arcsec')
        ax.set_ylabel('flux / Jy/sq arcsec')
        fig.tight_layout()
        fig.savefig(f"{outdir}/profile.pdf")

        # save model visibilities
        code = vis_r_stan_code.get_code(args.type, gq='vis',
                                        star=args.star is not None,
                                        bg=data['nbg'] > 0, pt=data['npt'] > 0,
                                        inc_lim=args.inc_lim, pa_lim=args.pa_lim,
                                        z_prior=args.zlim)
        with open(stanfile, 'w') as f:
            f.write(code)

        model = CmdStanModel(stan_file=stanfile,
                             cpp_options={'STAN_THREADS': 'TRUE'})

        for k in inits.keys():
            inits[k] = best[f'{k}_'] * mul[k]
        fit1 = model.sample(data=data, inits=inits, fixed_param=True,
                            chains=1, iter_warmup=0, iter_sampling=1)

        for f in visfiles:
            print(f'saving model for {os.path.basename(f)}')
            u, v, re, im, w = functions.read_vis(f)

            data['nvis'] = len(u)
            data['u'] = u
            data['v'] = v
            data['re'] = re
            data['im'] = im
            data['sigma'] = 1/np.sqrt(w)
            gq = model.generate_quantities(data=data, previous_fit=fit1)
            vis_mod = gq.stan_variables()['vismod_re'] + 1j*gq.stan_variables()['vismod_im']
            f_ = os.path.splitext(os.path.basename(f))
            f_save = f_[0] + '-vismod' + f_[1]
            np.save(f"{outdir}/{f_save}", vis_mod)
