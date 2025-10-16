import numpy as np
import torch
from typing import Dict, Union

from dymad.io import DynData
from dymad.models.model_temp_uenc import ModelTempUEncGraphDyn
from dymad.models.prediction import predict_continuous, predict_discrete
from dymad.modules import GNN

class LDMG(ModelTempUEncGraphDyn):
    """Latent Dynamics Model on Graph (LDMG).

    Uses MLP for node-wise encoder/decoder and GNN for dynamics.
    """
    GRAPH = True
    CONT  = True

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__(model_config, data_meta, dtype=dtype, device=device)

        enc_out_dim, dec_inp_dim = self._build_autoencoder(model_config, dtype, device)

        # The dynamics
        # Options can be different from autoencoder
        proc_depth = model_config.get('processor_layers', 2)
        opts = {
            'activation'     : model_config.get('activation', 'prelu'),
            'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
            'bias_init'      : model_config.get('bias_init', 'zeros'),
            'gain'           : model_config.get('gain', 1.0),
            'gcl'            : model_config.get('gcl', 'sage'),
            'gcl_opts'       : model_config.get('gcl_opts', {}),
            'end_activation' : model_config.get('end_activation', True),
            'dtype'          : dtype,
            'device'         : device
        }

        self.dynamics_net = GNN(
            input_dim  = enc_out_dim,
            latent_dim = self.latent_dimension,
            output_dim = dec_inp_dim,
            n_layers   = proc_depth,
            **opts
        )

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], method: str = 'dopri5', **kwargs) -> torch.Tensor:
        return predict_continuous(
            self, x0, ts,
            us=w.u, edge_index=w.ei, edge_weights=w.ew, edge_attr=w.ea,
            method=method, order=self.input_order, **kwargs)

class DLDMG(LDMG):
    """Discrete version of LDMG.
    """
    GRAPH = True
    CONT  = False

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__(model_config, data_meta, dtype=dtype, device=device)

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        return predict_discrete(
            self, x0, ts,
            us=w.u, edge_index=w.ei, edge_weights=w.ew, edge_attr=w.ea)
