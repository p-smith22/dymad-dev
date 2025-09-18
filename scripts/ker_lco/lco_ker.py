import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
import torch

from dymad.models import DKM, DKMSK, KM
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
opt_opk1 = {
    "type": "op_sep",
    "input_dim": 2,
    "output_dim": 2,
    "kopts": [opt_rbf1],
    "Ls": np.array([[[1, 0], [0, 1]]])
}
opt_share = {
    "type": "share",
    "kernel": opt_rbf1,
    "dtype": torch.float64,
    "ridge_init": RIDGE
}
opt_indep = {
    "type": "indep",
    "kernel": [opt_rbf1, opt_rbf1],
    "dtype": torch.float64,
    "ridge_init": RIDGE
}
opt_opval = {
    "type": "opval",
    "kernel": opt_opk1,
    "dtype": torch.float64,
    "ridge_init": RIDGE,
}

# opt_krr = opt_share
# opt_krr = opt_indep
opt_krr = opt_opval
mdl_kl = {
    "name" : 'ker_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0
    }
mdl_kl.update(**opt_krr)

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
trn_ct = {
    "n_epochs": 200,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "sweep_lengths": [4],
    "chop_mode": "initial",
    "chop_step": 0.5,
    "ls_update": {
        "method": "raw",
        "interval": 50,
        "times": 3,
        "reset": False}
        }
trn_dt = {
    "n_epochs": 400,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "ls_update": {
        "method": "raw",
        "interval": 100,
        "times": 5,
        "reset": False}
        }

smpl = {'x0': {
    'kind': 'perturb',
    'params': {'bounds': [-db, db], 'ref': _ref}}
    }
config_path = 'ker_model.yaml'

cfgs = [
    ('km_ln',  KM,     LinearTrainer,     {"model": mdl_kl, "training" : trn_ln}),
    ('km_nd',  KM,     NODETrainer,       {"model": mdl_kl, "training" : trn_ct}),
    ('dkm_ln', DKM,    LinearTrainer,     {"model": mdl_kl, "training" : trn_ln}),
    ('dkm_nd', DKM,    NODETrainer,       {"model": mdl_kl, "training" : trn_dt}),
    ('dks_ln', DKMSK,  LinearTrainer,     {"model": mdl_kl, "training" : trn_ln}),
    ('dks_nd', DKMSK,  NODETrainer,       {"model": mdl_kl, "training" : trn_dt}),
    ]

# IDX = [0, 1, 2, 3, 4, 5]
# IDX = [0, 1]
# IDX = [2, 3]
IDX = [3]
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
    J = 32
    sampler = TrajectorySampler(f, g, config='ker_data.yaml', config_mod=smpl)
    ts, xs, ys = sampler.sample(t_grid, batch=J)
    x_data = xs
    t_data = ts[0]

    res = [x_data]
    for i in IDX:
        mdl, MDL, _, _ = cfgs[i]
        _, prd_func = load_model(MDL, f'ker_{mdl}.pt')
        with torch.no_grad():
            pred = prd_func(x_data, t_data)
        res.append(pred)

    for _i in range(1, len(res)):
        print(labels[_i-1], np.linalg.norm(res[_i]-res[0])/np.linalg.norm(res[0]))

    stys = ['b-', 'r--', 'g:', 'm-.', 'c--', 'y:', 'k-.']
    f, ax = plt.subplots(nrows=2, sharex=True)
    for _i in range(J):
        for _j in range(len(res)):
            ax[0].plot(t_data, res[_j][_i][:, 0], stys[_j])
            ax[1].plot(t_data, res[_j][_i][:, 1], stys[_j])

    # plot_trajectory(
    #     np.array(res), t_data, "SA",
    #     labels=['Truth'] + labels, ifclose=False)

plt.show()
