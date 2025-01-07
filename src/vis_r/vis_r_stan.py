# import shutil
import os
import argparse
import logging
import numpy as np
from scipy.special import jn_zeros
import cmdstanpy
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import arviz as az
import corner

from . import functions
from . import vis_r_stan_code

def vis_r_stan_radial(argv=None):

    # setup
    parser = argparse.ArgumentParser(description='vis-r with stan')
    parser = functions.add_default_parser(parser)
    # stan specific args
    parser.add_argument('--sc', dest='sc', metavar='10', type=float, default=10,
                        help='Scale parameters for std ~ 1')
    parser.add_argument('--norm-mul', dest='norm_mul', metavar='10', type=float, default=10,
                        help='Additional scaling for flux normalisation')
    parser.add_argument('--inc-lim', dest='inc_lim', action='store_true', default=False,
                        help="Limit range of inclinations")
    parser.add_argument('--pa-lim', dest='pa_lim', action='store_true', default=False,
                        help="limit range of position angles")
    parser.add_argument('--no-pf', dest='pf', action='store_false', default=True,
                        help="Don't run pathfinder for initial posteriors")
    parser.add_argument('--log', dest='log', action='store_true', default=False,
                        help="Save cmdstanpy logger output")

    if argv is not None:
        args = parser.parse_args(args=argv[1:])
    else:
        args = parser.parse_args()

    # print(f'\nUsing cmdstan {cmdstanpy.cmdstan_version()} with cmdstanpy {cmdstanpy.__version__} in\n'
    #       f'{cmdstanpy.cmdstan_path()}')

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
    elif args.type == 'erf2_power':
        assert p.shape[1] == 7
        inits['ri'] = inits.pop('r')
        inits['ai'] = p[:, 2]
        inits['sigi'] = p[:, 3]
        inits['ro'] = p[:, 4]
        inits['ao'] = p[:, 5]
    elif args.type == 'gauss' or args.type == 'gauss_bessel':
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
        if p in ['pa', 'inc', 'bgpa', 'bgi']:
            mul[p] /= args.norm_mul
        if p in ['ai', 'ao', 'gam']:
            mul[p] /= args.norm_mul*10
        if p in ['zh'] and args.zlim:
            mul[p] = 1.0

    # set up parameter multipliers
    data = {'nr': len(inits['norm'])}
    for k in inits.keys():
        inits[k] *= mul[k]
        data[f'{k}_0'] = inits[k]
        data[f'{k}_mul'] = mul[k]

    mul = {}

    data['nbg'] = 0
    if args.bg:
        data['nbg'] = len(bg)

    data['npt'] = 0
    if args.pt:
        data['npt'] = len(pt)

    # set up output directory
    if args.outdir:
        outdir = args.outdir.rstrip()
    else:
        relpath = os.path.dirname(args.visfiles[0])
        outdir = f'{relpath}/{args.outrel.rstrip()}/vis-r-stn_{data["nr"]}{args.type}'
        if args.star:
            outdir += '_star'
        if args.bg:
            outdir += f'_{nbg}bg'
        if args.pt:
            outdir += f'_{npt}pt'

    if not os.path.exists(outdir):
        try:
            os.mkdir(outdir)
        except FileNotFoundError:
            print(f'Output dir could not be created, check path to\n{outdir}\nexists')
            exit()

    standir = f'{outdir}/stan'
    if not os.path.exists(standir):
        os.mkdir(standir)

    # logging
    cmdstanpy_logger = logging.getLogger("cmdstanpy")
    if args.log:
        cmdstanpy_logger.handlers = []
        cmdstanpy_logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(f'{standir}/cmdstanpy.log')
        handler.setLevel(logging.DEBUG)
        cmdstanpy_logger.addHandler(handler)
    else:
        cmdstanpy_logger.disabled = True

    # load data
    data['u'], data['v'], data['re'], data['im'], data['w'], _ = functions.load_data(args)
    data['nvis'] = len(data['u'])
    data['sigma'] = 1/np.sqrt(data['w'])

    # setup DHT
    arcsec = np.pi/180/3600
    Rnk, Qzero, Ykm = functions.setup_dht(args.sz, data['u'], data['v'])

    data['nhpt'] = len(Rnk)
    data['nq'] = len(Qzero)
    data['Ykm'] = Ykm
    data['Rnk'] = Rnk/arcsec
    data['Qnk'] = Qzero
    data['hnorm'] = 1/2.35e-11  # * (2 * np.pi * (r_out*arcsec)**2) / h._j_nN

    # get stan code and compile
    code = vis_r_stan_code.get_code(args.type, gq=False,
                                    star=args.star is not None,
                                    bg=data['nbg'] > 0, pt=data['npt'] > 0,
                                    inc_lim=args.inc_lim, pa_lim=args.pa_lim,
                                    z_prior=args.zlim)

    stanfile = f'{standir}/vis-r.stan'
    functions.update_stanfile(code, stanfile)

    model = CmdStanModel(stan_file=stanfile,
                         cpp_options={'STAN_THREADS': 'TRUE'})

    # print(model.exe_info())

    # initial run with pathfinder to estimate parameter multipliers
    # so that metric is approximately unit diagonal (stan init)
    metric = 'dense'
    if args.pf and not args.input_model:
        print('\nEstimate posteriors/multipliers with pathfinder')
        pf = model.pathfinder(data=data, inits=inits,
                              # output_dir=f'{outdir}/pf'
                              # show_console=True,
                              )

        cn = pf.column_names
        ok = ['_' in c and '__' not in c for c in cn]
        fig = corner.corner(pf.draws()[:, ok],
                            titles=np.array(pf.column_names)[ok], show_titles=True)
        fig.savefig(f'{outdir}/corner_pf.pdf')

        # ok = ['_' not in c for c in cn]
        # if we wanted to set metric rather than scales
        # metric = {'inv_metric': np.cov(pf.draws()[:, ok].T)}
        # print(np.diag(np.cov(pf.draws()[:, ok].T)))

        # set multipliers for appropriate scale, it seems faster to
        # set scale ~0.1 rather than 1
        for k in inits.keys():
            med = np.median(pf.stan_variable(f'{k}_'), axis=0)
            std = np.std(pf.stan_variable(f'{k}_'), axis=0)
            data[f'{k}_mul'] = 0.1 / np.mean(std)  # comment if setting metric above
            inits[k] = med * data[f'{k}_mul']
            data[f'{k}_0'] = inits[k]

    print('\nFitting parameters (name, initial value, multiplier)')
    print(f' model is: {args.type}')
    print(  '------------------------------------------------------')
    functions.pprint((range(len(inits.keys())), list(inits.keys()),
                      [inits[k] / data[f'{k}_mul'] for k in inits.keys()],
                      [f"{data[f'{k}_mul']:.2f}" for k in inits.keys()]))
    print('')

    fit = model.sample(data=data, chains=args.threads,
                       metric=metric,
                       iter_warmup=args.steps-args.keep, iter_sampling=args.keep,
                       inits=inits,
                       save_warmup=False,
                       show_console=False,
                       refresh=50)

    # shutil.copy(fit.metadata.cmdstan_config['profile_file'], outdir)

    df = fit.summary(sig_figs=3, percentiles=[50])
    df.drop(['MCSE', 'N_Eff/s'], axis=1, inplace=True)
    df_print = df[df.index.str.contains('_') & np.invert(df.index.str.contains('__'))].copy()
    med = df_print['50%']
    df_print.drop(['50%'], inplace=True, axis=1)
    df_print['StdDev (sc)'] = 0.
    for k in df_print.index:
        df_print.loc[k, 'StdDev (sc)'] = df.loc[k.replace('_',''), 'StdDev']
    print(df_print)
    # print(fit.diagnose())

    print(f'\nOutput in {outdir}')

    xr = fit.draws_xr()
    for k in inits.keys():
        xr = xr.drop_vars(k)

    if args.save_chains:
        # fit.save_csvfiles(outdir)
        chains = np.array(xr.to_dataarray().as_numpy().squeeze())
        np.save(f'{outdir}/chains.npy', np.moveaxis(chains, 0, -1))

    _ = az.plot_trace(xr)
    fig = _.ravel()[0].figure
    fig.tight_layout()
    fig.savefig(f'{outdir}/chains.png')

    best = {}
    for k in fit.stan_variables().keys():
        if '_' in k and '__' not in k:
            best[k] = np.median(fit.stan_variable(k), axis=0)
    # print(f'best: {best}')

    if not args.input_model:
        np.save(f'{outdir}/best_params.npy', np.vstack(([s for s in df_print.index], med, df_print['StdDev'])))

    # save model visibilities
    if args.save_model:

    # how to generate and save radial profiles from the sampling
        if 0:
            code = vis_r_stan_code.get_code(args.type, gq='prof',
                                            star=args.star is not None,
                                            bg=data['nbg'] > 0, pt=data['npt'] > 0,
                                            inc_lim=args.inc_lim, pa_lim=args.pa_lim,
                                            z_prior=args.zlim)
            stanfile = f'{standir}/vis-r-prof.stan'
            functions.update_stanfile(code, stanfile)

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
        stanfile = f'{standir}/vis-r-gq.stan'
        functions.update_stanfile(code, stanfile)

        model = CmdStanModel(stan_file=stanfile,
                             cpp_options={'STAN_THREADS': 'TRUE'})

        if not args.input_model:
            for k in inits.keys():
                inits[k] = best[f'{k}_'] * data[f'{k}_mul']
        else:
            print(' ignoring fitting and using initial parameters for model')

        for f in visfiles:
            print(f' saving model for {os.path.basename(f)}')
            u, v, re, im, w = functions.read_vis(f)

            data['nvis'] = len(u)
            data['u'] = u
            data['v'] = v
            data['re'] = re
            data['im'] = im
            data['sigma'] = 1/np.sqrt(w)
            # set iter_warmup=1; seems to be a bug in cmdstanpy where adapt_engaged=False not recognised
            gq = model.sample(data=data, inits=inits, fixed_param=True, adapt_engaged=False,
                              chains=1, iter_warmup=1, iter_sampling=1,
                              show_console=False, show_progress=False)
            vis_mod = gq.stan_variables()['vismod_re'] + 1j*gq.stan_variables()['vismod_im']
            f_ = os.path.splitext(os.path.basename(f))
            f_save = f_[0] + '-vismod' + f_[1]
            np.save(f"{outdir}/{f_save}", vis_mod.squeeze())

    # make a corner plot (may run out of memory for many parameters so do last)
    fig = corner.corner(xr, show_titles=True)
    fig.savefig(f'{outdir}/corner.pdf')
