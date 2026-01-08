import matplotlib.pyplot as plt
import numpy as np
import textwrap as tw
import torch

from dymad.io import load_model
from dymad.models import ComposedDynamics, ENC_MAP, DEC_MAP, get_dims, predict_discrete
from dymad.modules import make_network
from dymad.training import NODETrainer
from dymad.utils import plot_summary, plot_trajectory


class DSDMSKG(ComposedDynamics):
    GRAPH = True
    CONT = False

    def __init__(self, model_config, data_meta, dtype=None, device=None):
        super().__init__()

        # Dimensions
        dims = get_dims(model_config, data_meta)
        self.dim_x = dims['x'] // dims['seq']
        self.seq_len = dims['seq']

        # Common options
        opts = {
            'activation'     : model_config.get('activation', 'prelu'),
            'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
            'bias_init'      : model_config.get('bias_init', 'zeros'),
            'gain'           : model_config.get('gain', 1.0),
            'end_activation' : model_config.get('end_activation', True),
            'dtype'          : dtype,
            'device'         : device
        }

        # Autoencoder as in DyMAD: used in predict, do nothing
        self.encoder_net = None
        self.decoder_net = None
        self._encoder = ENC_MAP['iden']
        self._decoder = DEC_MAP['iden']

        # Components of dynamics
        # "Encoder": sequentially applied to each time step
        self.step_encoder = make_network(
            "seq_mlp_smp",
            input_dim  = dims['e'],
            hidden_dim = dims['h'],
            output_dim = dims['z'],
            n_layers   = dims['enc'],
            seq_len    = dims['seq'],
            **opts
        )

        # "Decoder": map only the last step to original space
        self.step_decoder = make_network(
            "mlp_smp",
            input_dim  = dims['r'],
            hidden_dim = dims['h'],
            output_dim = self.dim_x,
            n_layers   = dims['dec'],
            **opts
        )

        # Processor in the dynamics
        opts['gcl']      = model_config.get('gcl', 'sage')
        opts['gcl_opts'] = model_config.get('gcl_opts', {})
        self.processor_net = make_network(
            "gnn_smp",
            input_dim  = dims['s'],
            hidden_dim = dims['h'],
            output_dim = dims['r'],
            n_layers   = dims['prc'],
            **opts
        )

        # Prediction options
        self.input_order = None
        self._predict = predict_discrete

    def dynamics(self, z, w):   # z is x here
        ss = self.step_encoder(w.g(z), w.ug)
        dz = self.processor_net(ss, w.ei, w.ew, w.ea)
        dx = self.step_decoder(w.g(dz))
        x_next = w.xg[..., -self.dim_x:] + dx
        x_pred = torch.cat([w.xg[..., self.dim_x:], x_next], dim=-1)
        return w.G(x_pred)

    def diagnostic_info(self) -> str:
        """
        Return diagnostic information about the model.

        Returns:
            str: String with model details
        """
        ind = "          "
        fin = lambda net: tw.indent(f"{net}", ind)
        return f"Model parameters: {sum(p.numel() for p in self.parameters())}\n" + \
               f"Encoder:  \n{fin(self.step_encoder)}\n" + \
               f"Processor: \n{fin(self.processor_net)}\n" + \
               f"Decoder:  \n{fin(self.step_decoder)}\n" + \
               f"Prediction: {self._predict.__name__}\n" + \
               f"Continuous-time: {self.CONT}, Graph-compatible: {self.GRAPH}, " + \
               f"Sequence length: {self.seq_len}\n"


mdl_sdm = {
    "activation" : "prelu",
    "weight_init" : "xavier_uniform",
    "gain" : 0.01,
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

config_path = 'kur_seq.yaml'
data_path = './data/data_n4_s5_k4_s5.npz'
cfgs = [
    ('sdm_skip', DSDMSKG, NODETrainer, {"data": {"path": data_path}, "model": mdl_sdm, "training" : trn_nd}),
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
