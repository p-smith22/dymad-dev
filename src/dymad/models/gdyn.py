import numpy as np
import torch
from typing import Dict, Union

from dymad.io import DynData
from dymad.models.model_temp_uenc import ModelTempUEncGraphDyn
from dymad.models.prediction import predict_continuous, predict_discrete, predict_discrete_exp
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
            self, x0, ts, w,
            method=method, order=self.input_order, **kwargs)

class DLDMG(LDMG):
    """Discrete version of LDMG.
    """
    GRAPH = True
    CONT  = False

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__(model_config, data_meta, dtype=dtype, device=device)

        self._predictor_type = model_config.get('predictor_type', 'ode')
        if self.n_total_control_features > 0:
            if self._predictor_type == "exp":
                raise ValueError("Exponential predictor is not supported for model with control inputs.")

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        if self._predictor_type == "exp":
            return predict_discrete_exp(self, x0, ts, w, **kwargs)
        return predict_discrete(self, x0, ts, w, **kwargs)
