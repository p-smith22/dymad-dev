import numpy as np
import torch
from typing import Dict, Union

from dymad.io import DynData
from dymad.models import ModelTempUEnc, ModelTempUEncGraph, \
    predict_continuous, predict_discrete, predict_graph_continuous, predict_graph_discrete
from dymad.modules import MLP

class LDM(ModelTempUEnc):
    """Latent Dynamics Model (LDM)

    The encoder, dynamics, and decoder networks are implemented as MLPs.
    """
    GRAPH = False
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
            'end_activation' : model_config.get('end_activation', True),
            'dtype'          : dtype,
            'device'         : device
        }

        self.dynamics_net = MLP(
            input_dim  = enc_out_dim,
            latent_dim = self.latent_dimension,
            output_dim = dec_inp_dim,
            n_layers   = proc_depth,
            **opts
        )

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor],
                method: str = 'dopri5', **kwargs) -> torch.Tensor:
        return predict_continuous(self, x0, ts, us=w.u, method=method, order=self.input_order, **kwargs)

class DLDM(LDM):
    """Discrete Latent Dynamics Model (DLDM) - discrete-time version.

    In this case, the forward pass effectively does the following:

    ```
    z_n = self.encoder(w_n)
    z_{n+1} = self.dynamics(z_n, w_n)
    x_hat_n = self.decoder(z_n, w_n)
    ```
    """
    GRAPH = False
    CONT  = False

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super(DLDM, self).__init__(model_config, data_meta, dtype=dtype, device=device)

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        return predict_discrete(self, x0, ts, us=w.u)

class GLDM(ModelTempUEncGraph):
    """Graph Latent Dynamics Model (GLDM).

    Uses GNN for encoder/decoder and MLP for dynamics.
    """
    GRAPH = True
    CONT  = True

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super(GLDM, self).__init__(model_config, data_meta, dtype=dtype, device=device)

        enc_out_dim, dec_inp_dim = self._build_autoencoder(model_config, dtype, device)

        # The dynamics
        # Options can be different from autoencoder
        proc_depth = model_config.get('processor_layers', 2)
        opts = {
            'activation'     : model_config.get('activation', 'prelu'),
            'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
            'bias_init'      : model_config.get('bias_init', 'zeros'),
            'gain'           : model_config.get('gain', 1.0),
            'end_activation' : model_config.get('end_activation', True),
            'dtype'          : dtype,
            'device'         : device
        }

        self.dynamics_net = MLP(
            input_dim  = enc_out_dim,
            latent_dim = self.latent_dimension,
            output_dim = dec_inp_dim,
            n_layers   = proc_depth,
            **opts
        )

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], method: str = 'dopri5', **kwargs) -> torch.Tensor:
        return predict_graph_continuous(self, x0, ts, w.edge_index, us=w.u, method=method, order=self.input_order, **kwargs)

class DGLDM(GLDM):
    """Discrete Graph Latent Dynamics Model (DGLDM) - discrete-time version.

    Same idea as DKBF vs KBF.
    """
    GRAPH = True
    CONT  = False

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super(DGLDM, self).__init__(model_config, data_meta, dtype=dtype, device=device)

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        return predict_graph_discrete(self, x0, ts, w.edge_index, us=w.u)
