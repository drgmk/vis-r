import os
import shutil
import numpy as np

def test_fit():
    """Fit a model to a model and compare parameters."""

    def run_test(file, command):

        u, v, re, im, w = np.load('data/hr4796.empty.npy')
        vis = np.load(file)
        re += vis.real
        im += vis.imag
        if not os.path.exists('tmp'):
            os.mkdir('tmp')

        np.save('tmp/vis.npy', [u, v, re, im, w])

        real_par = np.load(f'{os.path.dirname(file)}/best_params.npy')

        # run the fit
        os.system(command)
        model_par = np.load('tmp/best_params.npy')

        assert(np.allclose(real_par[1].astype(float),
                           model_par[1].astype(float),
                           rtol=1e-2, atol=1e-2))

        shutil.rmtree('tmp')

    # Gaussian model
    run_test('examples/vis-r_1gauss/hr4796.selfcal-vismod.npy',
             ('vis-r -v tmp/vis.npy -o tmp '
              '--steps 2000 --keep 1000 '
              '-g 0.03 0.03 26 77 -p 0.015 1 0.1 0.05'))
    run_test('examples/vis-r_1gauss2/hr4796.selfcal-vismod.npy',
             ('vis-r -v tmp/vis.npy -o tmp -t gauss2 '
              '--steps 2000 --keep 1000 '
              ' -g 0.03 0.03 26 76 -p 0.015 1 0.05 0.05 0.05'))
    # power law model, comparison not so good as not all parameters constrained
    # run_test('examples/vis-r_1power/hr4796.selfcal-vismod.npy',
    #          ('vis-r -v tmp/vis.npy -o tmp -t power '
    #           '--steps 2000 --keep 1000 '
    #           '-g 0.03 0.03 26 77 -p 0.015 1 25 -25 2 0.05'))
