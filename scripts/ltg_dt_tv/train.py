import logging
import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model
from dymad.models import DLDMG
from dymad.training import NODETrainer, LinearTrainer
from dymad.utils import plot_summary, plot_trajectory, setup_logging, TrajectorySampler

mdl_kb = {
    "name" : 'kura_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0,
    "processor_layers" : 1,
    "latent_dimension" : 2,
    "gcl": "sage",
    # "gcl": "cheb",
    # "gcl_opts": {"K": 2},
    "activation" : "none",
    "weight_init" : "xavier_uniform"}

trn_nd = {
    "n_epochs": 2000,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "sweep_lengths": [2, 4, 8, 12, 16, 20],
    "sweep_epoch_step": 100,
    "chop_mode": "unfold",
    "chop_step": 0.5,
    "ls_update": {
        "method": "full",
        "interval": 200,
        "times": 1}
    }

config_path = 'config.yaml'
data_path = './data/data_n2_s3_k4_s10.npz'
cfgs = [
    ('dldmg', DLDMG, NODETrainer,        {"data": {"path": data_path}, "model": mdl_kb, "training" : trn_nd}),
    ]

IDX = [0]
labels = [cfgs[i][0] for i in IDX]

iftrn = 1
ifprd = 0

if iftrn:
    for _i in IDX:
        mdl, MDL, Trainer, opt = cfgs[_i]

        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

# if ifprd:
#     sampler = TrajectorySampler(f, g, config='lti_data.yaml', config_mod=config_gau)

#     ts, xs, us, ys = sampler.sample(t_grid, batch=1)
#     x_data = xs[0]
#     t_data = ts[0]
#     u_data = us[0]

#     res = [x_data]
#     for i in IDX:
#         MDL, mdl = cases[i]['model'], cases[i]['name']
#         _, prd_func = load_model(MDL, f'lti_{mdl}.pt', f'lti_{mdl}.yaml')

#         with torch.no_grad():
#             pred = prd_func(x_data, t_data, u=u_data)
#             res.append(pred)

#     plot_trajectory(
#         np.array(res), t_data, "LTI",
#         us=u_data, labels=['Truth']+labels, ifclose=False)

plt.show()
