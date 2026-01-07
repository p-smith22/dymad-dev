import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model
from dymad.models import CD_SDM, DSDMG, PredefinedModel
from dymad.training import NODETrainer
from dymad.utils import plot_summary, plot_trajectory

mdl_sdm = {
    # "name" : 'kura_model',
    # "encoder_layers" : 2,
    # "decoder_layers" : 2,
    # "hidden_dimension" : 32,
    # "koopman_dimension" : 16,
    # "gcl": "sage",
    # # "gcl": "cheb",
    # # "gcl_opts": {"K": 2},
    "activation" : "prelu",
    "weight_init" : "xavier_uniform"
    }

trn_nd = {
    "n_epochs": 1000,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "sweep_lengths": [2, 3, 5, 7],
    "sweep_epoch_step": 1000,
    "chop_mode": "unfold",
    "chop_step": 0.5,
    }

DSDMSKG = PredefinedModel(False, "node_raw", "none",  "graph_skip", "node",  CD_SDM)

config_path = 'kur_seq.yaml'
data_path = './data/data_n4_s5_k4_s5.npz'
cfgs = [
    ('sdm_node', DSDMG,   NODETrainer,      {"data": {"path": data_path}, "model": mdl_sdm, "training" : trn_nd}),
    ('sdm_skip', DSDMSKG, NODETrainer,      {"data": {"path": data_path}, "model": mdl_sdm, "training" : trn_nd}),
    ]

IDX = [0]
labels = [cfgs[i][0] for i in IDX]

iftrn = 1
ifprd = 0

if iftrn:
    for _i in IDX:
        mdl, MDL, Trainer, opt = cfgs[_i]
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
