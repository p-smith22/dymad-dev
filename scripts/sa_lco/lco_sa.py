# from pysako import resolventAnalysis, estimatePSpec

import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
import torch

from dymad.models import DKBF, KBF
from dymad.numerics import complex_plot
from dymad.sako import SpectralAnalysis
from dymad.training import LinearTrainer, NODETrainer
from dymad.utils import load_model, plot_trajectory, setup_logging, TrajectorySampler

B = 500
N = 41
t_grid = np.linspace(0, 4, N)
dt = t_grid[1] - t_grid[0]

mu = 1.0
def f(t, x):
    _x, _y = x
    dx = np.array([
        _y,
        mu * (1-_x**2)*_y - _x
    ])
    return dx
g = lambda t, x: x

# Reference trajectory
_Nt = 161
_ts = np.linspace(0, 40.0, 8*_Nt)
_res = spi.solve_ivp(f, [0,_ts[-1]], [2,2], t_eval=_ts)
_ref = _res.y[:,-240:].T

# Reference frequencies
_tmp = _res.y[0,-4*_Nt:]
_dt = _ts[1]           # dt from reference
sp = np.fft.fft(_tmp)
fr = np.fft.fftfreq(4*_Nt)/_dt*(2*np.pi)
ii = np.argmax(np.abs(sp))
w0 = np.abs(fr[ii])
wa = np.exp(np.array([-5,-4,-3,-2,-1,1,2,3,4,5]) * (1j*w0*dt)) # Use dt from data

# Transition to LCO
db = 0.2
# # LCO
# db = 0.001

mdl_kb = {
    "name" : 'sa_model',
    "encoder_layers" : 2,
    "decoder_layers" : 2,
    "latent_dimension" : 32,
    "koopman_dimension" : 32,
    "activation" : "tanh",
    # "autoencoder_type" : "cat",
    "weight_init" : "xavier_uniform",
    "predictor_type" : "exp",}

mdl_kl = {
    "name" : 'sa_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0,
    "koopman_dimension" : 64,
    "activation" : "none",
    "weight_init" : "xavier_uniform",
    "predictor_type" : "exp",}
trn_kl = [
        {"type": "scaler", "mode": "-11"},
        {"type": "lift", "fobs": "poly", "Ks": [8, 8]}
    ]

trn_nd = {
    "n_epochs": 2000,
    "save_interval": 10,
    "load_checkpoint": False,
    "learning_rate": 1e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "sweep_lengths": [2, 11, 21, 41],
    "sweep_epoch_step": 400,
    "chop_mode": "unfold",
    "chop_step": 0.5,
    "ls_update": {
        "method": "sako",
        "params": 9,
        "interval": 100,
        "times": 2,
        "start_with_ls": False}
        }

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
        "params": 15,
        "remove_one": True}}
trn_sa.update(ref)

smpl = {'x0': {
    'kind': 'perturb',
    'params': {'bounds': [-db, db], 'ref': _ref}}
    }
config_path = 'sa_model.yaml'

cfgs = [
    ('kbf_nd',  KBF,  NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ('dkbf_nd', DKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ('dkbf_ln', DKBF, LinearTrainer,   {"model": mdl_kl, "transform_x" : trn_kl, "training" : trn_ln}),
    ('dkbf_tr', DKBF, LinearTrainer,   {"model": mdl_kl, "transform_x" : trn_kl, "training" : trn_tr}),
    ('dkbf_sa', DKBF, LinearTrainer,   {"model": mdl_kl, "transform_x" : trn_kl, "training" : trn_sa}),
    ]

IDX = [2, 3, 4]
labels = [cfgs[i][0] for i in IDX]

ifdat = 0
iftrn = 0
ifprd = 0
ifint = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config='sa_data.yaml', config_mod=smpl)
    ts, xs, ys = sampler.sample(t_grid, batch=B, save='./data/sa.npz')

    for i in range(B):
        plt.plot(ys[i, :, 0], ys[i, :, 1])

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"sa_{mdl}"
        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifprd:
    sampler = TrajectorySampler(f, g, config='sa_data.yaml', config_mod=smpl)
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
    saln = SpectralAnalysis(DKBF, 'sa_dkbf_ln.pt', dt=dt, reps=1e-10, etol=None)
    satr = SpectralAnalysis(DKBF, 'sa_dkbf_tr.pt', dt=dt, reps=1e-10, etol=None)
    sasa = SpectralAnalysis(DKBF, 'sa_dkbf_sa.pt', dt=dt, reps=1e-10, etol=None)
    sact = SpectralAnalysis(KBF,  'sa_kbf_nd.pt',  dt=dt, reps=1e-10)

    sas = [saln, satr, sasa, sact]
    lbs = ['DT-LN', 'DT-TR', 'DT-SA', 'CT-ND']

    Ns  = len(sas)

    ifprd = 1
    ifeig, ifeic, ifpsp, ifres = 0, 0, 1, 0
    ifspe, ifegf = 0, 0

    if ifprd:
        J = 16
        sampler = TrajectorySampler(f, g, config='sa_data.yaml', config_mod=smpl)
        ts, xs, _ = sampler.sample(t_grid, batch=J)
        x0s = xs[:, 0, :].squeeze()

        for _i in range(Ns):
        # for _i in [2]:
            sas[_i].plot_pred_x(x0s, ts[0], ref=xs, idx='all', figsize=(6,8), title=lbs[_i])

    if ifeig:
        ## Eigenvalues
        MRK = 15
        fig, ax = plt.subplots(ncols=Ns, sharey=True, figsize=(12,5))
        for _i in range(Ns):
            fig, ax[_i], _ls = sas[_i].plot_eigs(fig=(fig, ax[_i]), plot_filt=None)
            _l, = ax[_i].plot(wa.real, wa.imag, 'kx', markersize=MRK)
            ax[_i].set_title(f'{lbs[_i]}\nMax res: {sas[_i]._res[-1]:4.3e}')
            ax[_i].legend(_ls+[_l], ["Full-Order", "Truth"], loc=1)

        if ifpsp:
            # Pseudospectra
            xs = np.linspace(-1.3, 1.3, 51)
            gg = np.vstack([xs, xs])
            rng = np.array([0.1, 0.25])

            # Predicted
            pss, psk = [], []
            for _s in sas:
                grid, _pss = _s.estimate_ps(gg, mode='disc', method='standard', return_vec=False)
                grid, _psk = _s.estimate_ps(gg, mode='disc', method='sako', return_vec=False)
                pss.append((grid, _pss))
                psk.append((grid, _psk))

            for _i in range(Ns):
                grid, _pss = pss[_i]
                grid, _psk = psk[_i]
                f, ax[_i] = complex_plot(grid, 1/_pss, rng, fig=(f, ax[_i]), mode='line', lwid=2, lsty='dotted')
                f, ax[_i] = complex_plot(grid, 1/_psk, rng, fig=(f, ax[_i]), mode='line', lwid=1)

        for _i in range(Ns):
            ax[_i].set_xlim([-0.1, 1.3])
            ax[_i].set_ylim([-1.1, 1.1])

    if ifeic:
        ## Eigenvalues
        MRK = 15
        fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(15,5))
        for _i in range(Ns):
            fig, ax[_i], _ls = sas[_i].plot_eigs(fig=(fig, ax[_i]), mode='cont', plot_filt=None)
            _l, = ax[_i].plot(w0.real, w0.imag, 'kx', markersize=MRK)
            ax[_i].set_title(f'{lbs[_i]}\nMax res: {sas[_i]._res[-1]:4.3e}')
            ax[_i].legend(_ls+[_l], ["Full-Order", "Truth"], loc=1)

        if ifpsp:
            # Pseudospectra
            zs = np.linspace(-4.0,0.5,51)
            ws = np.linspace(-6.6,6.6,51)
            gg = np.vstack([zs,ws])
            rng = np.array([0.25, 0.5])
            # Predicted
            pss, psk = [], []
            for _s in sas:
                grid, _pss = _s.estimate_ps(gg, mode='cont', method='standard', return_vec=False)
                grid, _psk = _s.estimate_ps(gg, mode='cont', method='sako', return_vec=False)
                pss.append((grid, _pss))
                psk.append((grid, _psk))

            for _i in range(Ns):
                grid, _pss = pss[_i]
                grid, _psk = psk[_i]
                fig, ax[_i] = complex_plot(grid, 1/_pss, rng, fig=(fig, ax[_i]), mode='line', lwid=2, lsty='dotted')
                fig, ax[_i] = complex_plot(grid, 1/_psk, rng, fig=(fig, ax[_i]), mode='line', lwid=1)

        for _i in range(2):
            ax[_i].set_xlim([-4.0, 0.5])
            ax[_i].set_ylim([-6.6, 6.6])

    if ifres:
        ## Residuals
        stys = ['bo', 'r^', 'gs', 'md', 'c*']
        fig, ax = plt.subplots()
        for _i in range(Ns):
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
        for _i in range(Ns):
            th, vg = vgs[_i]
            plt.plot(th, vg, stys[_i], label=lbs[_i], markerfacecolor='none')
        plt.plot([_arg[0], _arg[0]], [0, _amp], 'k:', label='System frequency')
        for _a in _arg[1:]:
            plt.plot([_a, _a], [0, _amp], 'k:')
        plt.legend()
        plt.xlabel('Angle, rad')
        plt.ylabel('Spectral measure')

    if ifegf:
        ## Eigenfunctions
        rngs = [[-np.pi/2.5, np.pi/2.5], [-1.4, 1.4]]
        Ne = [101, 121]

        f1, a1 = plt.subplots(nrows=Ns, ncols=4, sharex=True, sharey=True, figsize=(10,10))
        f2, a2 = plt.subplots(nrows=Ns, ncols=4, sharex=True, sharey=True, figsize=(10,10))
        for i in range(Ns):
            _n = min(sas[i]._Nrank, 4)
            sas[i].plot_eigfun_2d(rngs, Ne, _n, mode='abs', fig=(f1, a1[i]))
            a1[i][0].set_ylabel(lbs[i])
            sas[i].plot_eigfun_2d(rngs, Ne, _n, mode='angle', fig=(f2, a2[i]))
            a2[i][0].set_ylabel(lbs[i])

plt.show()




"""
Limit Cycle Oscillations, which should consist of only point spectrum.

Rule of thumb:
1. All methods do not perform well in predicting transient responses.
2. ResDMD and K-ResDMD should perform similarly, with K-ResDMD slightly better in prediction and spectrum.
3. EDMD gives reasonable eigenfunctions regardless of data type; others do well for transient data (as trajs cover more space)

@author Dr. Daning Huang
@date 07/06/24
"""

