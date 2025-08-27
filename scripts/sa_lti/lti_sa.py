import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spl
import torch

from dymad.models import DKBF, KBF
from dymad.numerics import complex_plot
from dymad.sako import estimate_pseudospectrum, resolvent_analysis, SpectralAnalysis
from dymad.training import LinearTrainer, NODETrainer
from dymad.utils import load_model, plot_trajectory, setup_logging, TrajectorySampler

B = 64
N = 21
t_grid = np.linspace(0, 10, N)
dt = t_grid[1] - t_grid[0]

# prf = 'lti_har'
# A = np.array([
#     [0.0, 1.0],
#     [-4.0, 0.0]])
# prf = 'lti_dmp'
# A = np.array([
#     [0.0, 1.0],
#     [-4.0, -1.0]])
prf = 'lti_dgn'
A = 0.5*np.array([
    [-1.0,-0.9],
    [0.0, -1.0]])

def f(t, x):
    return (x @ A.T)
g = lambda t, x: x

# True eigenvalues
w0 = np.linalg.eig(A)[0]
w0 = np.hstack(
    [w0,
     2*w0[0], 2*w0[1], w0[0]+w0[1],
     3*w0[0], 3*w0[1], 2*w0[0]+w0[1], w0[0]+2*w0[1],
     4*w0[0], 4*w0[1], 3*w0[0]+w0[1], 2*w0[0]+2*w0[1], w0[0]+3*w0[1]])
wa = np.exp(w0*dt)

mdl_kb = {
    "name" : 'sa_model',
    "encoder_layers" : 1,
    "decoder_layers" : 1,
    "koopman_dimension" : 16,
    "activation" : "none",
    "autoencoder_type" : "cat",
    "weight_init" : "xavier_uniform",
    "predictor_type" : "exp",
    }

mdl_kl = {
    "name" : 'sa_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0,
    "koopman_dimension" : 16,
    "activation" : "none",
    "weight_init" : "xavier_uniform",
    "predictor_type" : "exp",}
trn_kl = [
        {"type": "scaler", "mode": "-11"},
        {"type": "lift", "fobs": "poly", "Ks": [4, 4]}
    ]

trn_nd = {
    "n_epochs": 600,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "sweep_lengths": [2, 4],
    "sweep_epoch_step": 100,
    "ls_update": {
        "method": "truncated_log",
        "params": 2,
        "interval": 50,
        "times": 1}
        }
trn_dt = copy.deepcopy(trn_nd)
trn_dt["ls_update"]["method"] = "truncated"
ref = {
    "n_epochs": 1,
    "save_interval": 1,
    "load_checkpoint": False,}
trn_ln = {
    "ls_update": {
        "method": "full"}}
trn_ln.update(ref)
trn_tr = {
    "ls_update": {
        "method": "truncated",
        "params": 0.999}}
trn_tr.update(ref)
trn_sa = {
    "ls_update": {
        "method": "sako",
        "params": 9,
        "remove_one": True}}
trn_sa.update(ref)

config_path = 'sa_model.yaml'

cfgs = [
    ('kbf_nd',  KBF,  NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ('dkbf_nd', DKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_dt}),
    ('dkbf_ln', DKBF, LinearTrainer,   {"model": mdl_kl, "transform_x" : trn_kl, "training" : trn_ln}),
    ('dkbf_tr', DKBF, LinearTrainer,   {"model": mdl_kl, "transform_x" : trn_kl, "training" : trn_tr}),
    ('dkbf_sa', DKBF, LinearTrainer,   {"model": mdl_kl, "transform_x" : trn_kl, "training" : trn_sa}),
    ]

IDX = [0, 1, 2, 3, 4]
# IDX = [0]
# IDX = [2, 3, 4]
labels = [cfgs[i][0] for i in IDX]

ifdat = 1
iftrn = 1
ifprd = 1
ifint = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config='sa_data.yaml')
    ts, xs, ys = sampler.sample(t_grid, batch=B, save='./data/sa.npz')

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"sa_{mdl}"
        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifprd:
    sampler = TrajectorySampler(f, config='sa_data.yaml')
    ts, xs, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]

    res = [x_data]
    for i in IDX:
        mdl, MDL, _, _ = cfgs[i]
        _, prd_func = load_model(MDL, f'sa_{mdl}.pt')
        with torch.no_grad():
            pred = prd_func(x_data, t_data)
        res.append(pred)

    plot_trajectory(
        np.array(res), t_data, "SA",
        labels=['Truth'] + labels, ifclose=False)

if ifint:
    sact = SpectralAnalysis(KBF,  'sa_kbf_nd.pt',  dt=dt, reps=1e-10)
    sand = SpectralAnalysis(DKBF, 'sa_dkbf_nd.pt', dt=dt, reps=1e-10)
    saln = SpectralAnalysis(DKBF, 'sa_dkbf_ln.pt', dt=dt, reps=1e-10)
    satr = SpectralAnalysis(DKBF, 'sa_dkbf_tr.pt', dt=dt, reps=1e-10)
    sasa = SpectralAnalysis(DKBF, 'sa_dkbf_sa.pt', dt=dt, reps=1e-10)

    sas = [saln, satr, sasa, sand, sact]
    lbs = ['DT-LN', 'DT-TR', 'DT-SA', 'DT-ND', 'CT-ND']

    ifeig, ifeic, ifpsp, ifres = 1, 1, 1, 1
    ifspe, ifegf = 1, 1

    if ifeig:
        ## Eigenvalues
        MRK = 15
        f, ax = plt.subplots(ncols=5, sharey=True, figsize=(15,5))
        for _i in range(5):
            f, ax[_i], _ls = sas[_i].plot_eigs(fig=(f, ax[_i]))
            _l, = ax[_i].plot(wa.real, wa.imag, 'kx', markersize=MRK)
            ax[_i].set_title(f'{lbs[_i]}\nMax res: {sas[_i]._res[-1]:4.3e}')
            ax[_i].legend(_ls+[_l], [lbs[_i], "Filtered", "Truth"], loc=1)

        if ifpsp:
            # Pseudospectra
            xs = np.linspace(-1.3, 1.3, 51)
            gg = np.vstack([xs, xs])
            rng = np.array([0.1, 0.25])

            # Predicted
            pss, psk = [], []
            for _s in sas:
                grid, _pss = _s.estimate_ps(gg, mode='disc', method='standard', return_vec=False)
                # grid, _psk = _s.estimate_ps(gg, mode='disc', method='sako', return_vec=False)
                pss.append((grid, _pss))
                # psk.append((grid, _psk))
            # Exact
            psrf = estimate_pseudospectrum(
                grid, resolvent_analysis, return_vec=False,
                A=spl.expm(A*dt), B=None, ord=1)

            for _i in range(5):
                grid, _pss = pss[_i]
                # grid, _psk = psk[_i]
                f, ax[_i] = complex_plot(grid, 1/psrf, rng, fig=(f, ax[_i]), mode='line', lwid=0.5, lsty='dashed')
                f, ax[_i] = complex_plot(grid, 1/_pss, rng, fig=(f, ax[_i]), mode='line', lwid=2, lsty='dotted')
                # f, ax[_i] = complex_plot(grid, 1/_psk, rng, fig=(f, ax[_i]), mode='line', lwid=1)

    if ifeic:
        ## Eigenvalues
        MRK = 15
        f, ax = plt.subplots(ncols=5, sharey=True, figsize=(15,5))
        for _i in range(5):
            f, ax[_i], _ls = sas[_i].plot_eigs(fig=(f, ax[_i]), mode='cont')
            _l, = ax[_i].plot(w0.real, w0.imag, 'kx', markersize=MRK)
            ax[_i].set_title(f'{lbs[_i]}\nMax res: {sas[_i]._res[-1]:4.3e}')
            ax[_i].legend(_ls+[_l], [lbs[_i], "Filtered", "Truth"], loc=1)

        if ifpsp:
            # Pseudospectra
            zs = np.linspace(-4.0,0.5,51)
            ws = np.linspace(-4.5,4.5,51)
            gg = np.vstack([zs,ws])
            rng = np.array([0.25, 0.5])
            # Predicted
            pss, psk = [], []
            for _s in sas:
                grid, _pss = _s.estimate_ps(gg, mode='cont', method='standard', return_vec=False)
                # grid, _psk = _s.estimate_ps(gg, mode='cont', method='sako', return_vec=False)
                pss.append((grid, _pss))
                # psk.append((grid, _psk))
            # Exact
            psrf = estimate_pseudospectrum(
                grid, resolvent_analysis, return_vec=False,
                A=A, B=None, ord=1)

            for _i in range(5):
                grid, _pss = pss[_i]
                # grid, _psk = psk[_i]
                f, ax[_i] = complex_plot(grid, 1/psrf, rng, fig=(f, ax[_i]), mode='line', lwid=0.5, lsty='dashed')
                f, ax[_i] = complex_plot(grid, 1/_pss, rng, fig=(f, ax[_i]), mode='line', lwid=2, lsty='dotted')
                # f, ax[_i] = complex_plot(grid, 1/_psk, rng, fig=(f, ax[_i]), mode='line', lwid=1)

        for _i in range(5):
            ax[_i].set_xlim([-4.0, 0.5])
            ax[_i].set_ylim([-4.5, 4.5])

    if ifres:
        ## Residuals
        stys = ['bo', 'r^', 'gs', 'md', 'c*']
        f, ax = plt.subplots()
        for _i in range(5):
            ax.semilogy(np.abs(sas[_i]._wd), sas[_i]._res, stys[_i], label=lbs[_i], markerfacecolor='none')
        ax.set_xlabel('Norm of eigenvalue')
        ax.set_ylabel('Residual')
        ax.legend()

    if ifspe:
        # Spectral measure
        stys = ['b-', 'r-', 'g--', 'm--', 'c-']
        def func_obs(x):
            _x1, _x2 = x.T
            return _x1+_x2
        vgs = []
        for _s in sas:
            _t, _v = _s.estimate_measure(func_obs, 6, 0.1, thetas=501)
            vgs.append((_t, _v))

        _arg = np.angle(wa)
        _amp = np.max(vgs[0][1])

        f = plt.figure()
        for _i in range(5):
            th, vg = vgs[_i]
            plt.plot(th, vg, stys[_i], label=lbs[_i], markerfacecolor='none')
        plt.plot([_arg[0], _arg[0]], [0, _amp], 'k:', label='System frequency')
        plt.plot([_arg[1], _arg[1]], [0, _amp], 'k:')
        plt.legend()
        plt.xlabel('Angle, rad')
        plt.ylabel('Spectral measure')

    if ifegf:
        ## Eigenfunctions
        rngs = [[-np.pi/2.5, np.pi/2.5], [-1.4, 1.4]]
        Ns = [101, 121]
        md = 'angle'

        f, ax = plt.subplots(nrows=5, ncols=4, sharex=True, sharey=True, figsize=(10,10))
        for i in range(5):
            _n = min(sas[i]._Nrank, 4)
            sas[i].plot_eigfun_2d(rngs, Ns, _n, mode=md, fig=(f, ax[i]))
            ax[i][0].set_ylabel(lbs[i])

plt.show()