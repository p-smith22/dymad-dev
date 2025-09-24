import logging
import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.models import DKMSK, KM, KMM
from dymad.training import LinearTrainer
from dymad.utils import load_model, plot_trajectory, setup_logging, TrajectorySampler

B = 20
N = 101
t_grid = np.linspace(0, 8, N)
t_pred = np.linspace(0, 16, N*2)

s5 = np.sqrt(5)
K, D = 3, 0.1
# \dot{\theta} = 3/2 - \cos(\theta)
def dyn(tt, K=K, D=D):
    vv = 2 * np.arctan(np.tan(s5*tt/4)/s5)
    rr = 1 + D*np.cos(K*vv)
    uu = np.array([
        rr*np.cos(vv),
        rr*np.sin(vv)]).T
    return vv, uu
t_ref = np.linspace(0, 6, 51)
_ref = dyn(t_ref)[1]

def f(t, x, u):
    _x = np.atleast_2d(x)
    _t = np.arctan2(_x[:,1], _x[:,0])
    _v = 1.5 - np.cos(_t) + u
    _r = 1 + D*np.cos(K*_t)
    _d = -K*D*np.sin(K*_t)
    _c, _s = np.cos(_t)*_v, np.sin(_t)*_v
    _T = np.vstack([
        -_r*_s+_d*_c, _r*_c+_d*_s]).T
    return _T.squeeze()
g = lambda t, x, u: x

config_chr = {
    "control" : {
        "kind": "chirp",
        "params": {
            "t1": 8.0,
            "freq_range": (0.25, 0.5),
            "amp_range": (0.5, 1.0),
            "phase_range": (0.0, 360.0)}},
    'x0': {
        'kind': 'perturb',
        'params': {'bounds': [0, 0], 'ref': _ref}}
    }

# Training options
RIDGE = 1e-10

## Multi-output shared scalar kernel
opt_rbf = {
    "type": "sc_rbf",
    "input_dim": 3,
    "lengthscale_init": 1.0
}
opt_share = {
    "type": "share",
    "kernel": opt_rbf,
    "dtype": torch.float64,
    "ridge_init": RIDGE
}
mdl_kl = {
    "name" : 'ker_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0,
    "kernel_dimension" : 2
    }
mdl_kl.update(**opt_share)

## Tangent kernel
opt_opk = {
    "type": "op_tan",
    "input_dim": 3,
    "output_dim": 2,
    "kopts": opt_rbf
}
opt_tange = {
    "type": "tangent",
    "kernel": opt_opk,
    "dtype": torch.float64,
    "ridge_init": RIDGE
}
mdl_mn = {
    "name" : 'ker_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0,
    "kernel_dimension" : 2,
    "manifold" : {
        "d" : 1,
        "T" : 4,
        "g" : 4
    }}
mdl_mn.update(**opt_tange)

trn_ln = {
    "n_epochs": 1,
    "save_interval": 100,
    "load_checkpoint": False,
    "ls_update": {
        "method": "raw",
        "interval": 500,
        "times": 1}
        }

config_path = 'ker_model.yaml'

cfgs = [
    ('km_ln',  KM,     LinearTrainer,     {"model": mdl_kl, "training" : trn_ln}),
    ('kmm_ln', KMM,    LinearTrainer,     {"model": mdl_mn, "training" : trn_ln}),
    ('dks_ln', DKMSK,  LinearTrainer,     {"model": mdl_kl, "training" : trn_ln}),
    ]

IDX = [0, 1, 2]
labels = [cfgs[i][0] for i in IDX]

ifdat = 1
iftrn = 1
ifprd = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config='ker_data.yaml', config_mod=config_chr)
    ts, xs, us, ys = sampler.sample(t_grid, batch=B, save='./data/ker.npz')

    fig, ax = plt.subplots(nrows=2, sharex=True)
    for i in range(B):
        ax[0].plot(ts[0], ys[i, :, 0], 'b-')
        ax[1].plot(ts[0], ys[i, :, 1], 'b-')

    fig = plt.figure()
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
    sampler = TrajectorySampler(f, g, config='ker_data.yaml', config_mod=config_chr)
    ts, xs, us, ys = sampler.sample(t_pred, batch=1, save='./data/ker.npz')
    x_data = xs[0]
    t_data = ts[0]
    u_data = us[0]

    res = [x_data]
    for _i in IDX:
        mdl, MDL, _, _ = cfgs[_i]
        _, prd_func = load_model(MDL, f'ker_{mdl}.pt')
        with torch.no_grad():
            pred = prd_func(x_data, u_data, t_data)
        res.append(pred)

    plot_trajectory(
        np.array(res), t_data, "S1U",
        us=u_data, labels=['Truth']+labels, ifclose=False)

    fig = plt.figure()
    plt.plot(_ref[:,0], _ref[:,1], 'k--', linewidth=2)
    for _i, _r in enumerate(res[1:]):
        plt.plot(_r[:,0], _r[:,1], '-', label=labels[_i])
    plt.legend()

plt.show()
