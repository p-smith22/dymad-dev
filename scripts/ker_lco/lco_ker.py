import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
import torch

from dymad.models import KM
from dymad.training import LinearTrainer, NODETrainer
from dymad.utils import load_model, plot_trajectory, setup_logging, TrajectorySampler

B = 50
N = 81
t_grid = np.linspace(0, 8, N)
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
_ref = _res.y[:,-220:].T

# Transition to LCO
db = 0.4

# Training options
RIDGE = 1e-10
opt_rbf1 = {
    "type": "sc_rbf",
    "input_dim": 2,
    "lengthscale_init": 1.0
}
opt_share = {
    "type": "share",
    "kernel": opt_rbf1,
    "dtype": torch.float64,
    "ridge_init": RIDGE
}

mdl_kl = {
    "name" : 'ker_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0
    }
mdl_kl.update(**opt_share)

trn_ln = {
    "n_epochs": 1,
    "save_interval": 100,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "ls_update": {
        "method": "raw",
        "interval": 500,
        "times": 1}
        }

smpl = {'x0': {
    'kind': 'perturb',
    'params': {'bounds': [-db, db], 'ref': _ref}}
    }
config_path = 'ker_model.yaml'

cfgs = [
    ('km_ln',  KM,  LinearTrainer,     {"model": mdl_kl, "training" : trn_ln}),
    # ('dkbf_nd', DKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_dt}),
    # ('dkbf_ln', DKBF, LinearTrainer,   {"model": mdl_kl, "transform_x" : trn_kl, "training" : trn_ln}),
    # ('dkbf_tr', DKBF, LinearTrainer,   {"model": mdl_kl, "transform_x" : trn_kl, "training" : trn_tr}),
    # ('dkbf_sa', DKBF, LinearTrainer,   {"model": mdl_kl, "transform_x" : trn_kl, "training" : trn_sa}),
    ]

# IDX = [0, 1, 2, 3, 4]
IDX = [0]
# IDX = [2, 3, 4]
labels = [cfgs[i][0] for i in IDX]

ifdat = 0
iftrn = 1
ifprd = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config='ker_data.yaml', config_mod=smpl)
    ts, xs, ys = sampler.sample(t_grid, batch=B, save='./data/ker.npz')

    for i in range(B):
        plt.plot(ys[i, :, 0], ys[i, :, 1])
    plt.plot(_ref[:,0], _ref[:,1], 'k--', linewidth=2)

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"ker_{mdl}"
        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifprd:
    sampler = TrajectorySampler(f, g, config='ker_data.yaml', config_mod=smpl)
    ts, xs, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]

    res = [x_data]
    for i in IDX:
        mdl, MDL, _, _ = cfgs[i]
        _, prd_func = load_model(MDL, f'ker_{mdl}.pt')
        with torch.no_grad():
            pred = prd_func(x_data, t_data)
        res.append(pred)

    plot_trajectory(
        np.array(res), t_data, "SA",
        labels=['Truth'] + labels, ifclose=False)

plt.show()
