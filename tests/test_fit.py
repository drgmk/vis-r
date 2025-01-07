import os
import shutil
import numpy as np


def random_params():
    """Make some random disk params.

    These are a bit on the restricted side so that the fitting can
    average the visibilities fairly heavily so run faster, and so
    that the disk should be vertically resolved, so that the final
    parameter has a reasonable good chance of being constrained.
    """
    #    dra   ddec            pa                   inc
    g = [0.05, 0.05, np.random.uniform()*180, np.random.uniform()*30+50]
    #              flux                          radius
    p = [np.random.uniform()*0.05+0.1, np.random.uniform()*0.2+1,
         #           width                      height
         np.random.uniform()*0.05+0.1, np.random.uniform()*0.05+0.1]
    return g, p


def put_model(file, out):
    """Put model from a file into an empty one."""
    u, v, re, im, w = np.load('data/hr4796.empty.npy')
    re /= 10  # shrink the noise w.r.t. the input model
    im /= 10
    vis = np.load(file)
    re += vis.real
    im += vis.imag
    np.save(out, [u, v, re, im, w])


# def test_prior_fit():
#     """Fit a model to a previously fitted model and compare parameters."""
#
#     # Gaussian model
#     run_test('examples/vis-r_1gauss/hr4796.selfcal-vismod.npy',
#              ('vis-r -v tmp/vis.npy -o tmp '
#               '--steps 2000 --keep 1000 '
#               '-g 0.03 0.03 26 77 -p 0.015 1 0.1 0.05'))
#     # double-sided Gaussian, this is a bit sub-optimal since the
#     # sigma parameters aren't well constrained
#     # run_test('examples/vis-r_1gauss2/hr4796.selfcal-vismod.npy',
#     #          ('vis-r -v tmp/vis.npy -o tmp -t gauss2 '
#     #           '--steps 2000 --keep 1000 '
#     #           ' -g 0.03 0.03 26 76 -p 0.015 1 0.05 0.05 0.05'))


def test_random_model_fit():
    # generate random model, put into empty data, fit with emcee, compare
    # with what was injected

    for i in range(10):

        g, p = random_params()
        os.system('vis-r -v data/hr4796.selfcal.npy -o tmp '
                  '--steps 50 --keep 40 --input-model '
                  f'-g {" ".join(map(str, g))} -p {" ".join(map(str, p))}')
        put_model('tmp/hr4796.selfcal-vismod.npy', 'tmp/vis.npy')
        os.system('vis-r -v tmp/vis.npy -o tmp2 --sz 4 '
                   '--steps 3000 --keep 2000 '
                   f'-g {" ".join(map(str, g))} -p {" ".join(map(str, p))}')

        p1 = np.load('tmp/best_params.npy')[1].astype(float)
        p2 = np.load('tmp2/best_params.npy')[1].astype(float)
        assert np.allclose(p1, p2, rtol=1e-2)

    shutil.rmtree('tmp')
    shutil.rmtree('tmp2')


def test_emcee_stan_fits():
    # generate random model, put into empty data, fit with emcee and stan,
    # compare the results

    for i in range(10):

        # make random model
        g, p = random_params()
        os.system('vis-r -v data/hr4796.selfcal.npy -o tmp '
                  '--steps 50 --keep 40 --input-model '
                  f'-g {" ".join(map(str, g))} -p {" ".join(map(str, p))}')
        put_model('tmp/hr4796.selfcal-vismod.npy', 'tmp/vis.npy')

        os.system('vis-r -v tmp/vis.npy -o tmp --sz 4 '
                  '--steps 3000 --keep 2000 '
                  f'-g {" ".join(map(str, g))} -p {" ".join(map(str, p))}')
        os.system('vis-r -v tmp/vis.npy -o tmp2 --stan --sz 4 '
                  f'-g {" ".join(map(str, g))} -p {" ".join(map(str, p))}')

        p1 = np.load('tmp/best_params.npy')[1].astype(float)
        p2 = np.load('tmp2/best_params.npy')[1].astype(float)
        assert np.allclose(p1, p2, rtol=1e-2)

    shutil.rmtree('tmp')
    shutil.rmtree('tmp2')


def test_vis_models():
    # generate random model with emcee and stan, compare the visibilities

    for i in range(10):

        g, p = random_params()
        command = ('vis-r -v data/hr4796.selfcal.npy --sz 4 '
                   '--steps 50 --keep 40 --input-model '
                   f'-g {" ".join(map(str, g))} -p {" ".join(map(str, p))}')
        os.system(command + ' -o tmp')
        os.system(command + ' -o tmp2 --stan')

        v1 = np.load('tmp/hr4796.selfcal-vismod.npy')
        v2 = np.load('tmp2/hr4796.selfcal-vismod.npy')
        assert np.allclose(v1, v2)

    shutil.rmtree('tmp')
    shutil.rmtree('tmp2')
