import numpy as np
import torch
from typing import Dict, Union

from dymad.io import DynData
from dymad.models.temp_uenc import TemplateUEnc, TemplateUEncGraphAE
from dymad.models.prediction import predict_continuous, predict_discrete, predict_discrete_exp
from dymad.modules import MLP

class LDM(TemplateUEnc):
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
        return predict_continuous(self, x0, ts, w, method=method, order=self.input_order, **kwargs)

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

        self._predictor_type = model_config.get('predictor_type', 'ode')
        if self.n_total_control_features > 0:
            if self._predictor_type == "exp":
                raise ValueError("Exponential predictor is not supported for model with control inputs.")

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        if self._predictor_type == "exp":
            return predict_discrete_exp(self, x0, ts, w, **kwargs)
        return predict_discrete(self, x0, ts, w, **kwargs)

class GLDM(TemplateUEncGraphAE):
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
        return predict_continuous(
            self, x0, ts, w,
            method=method, order=self.input_order, **kwargs)

class DGLDM(GLDM):
    """Discrete Graph Latent Dynamics Model (DGLDM) - discrete-time version.

    Same idea as DKBF vs KBF.
    """
    GRAPH = True
    CONT  = False

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super(DGLDM, self).__init__(model_config, data_meta, dtype=dtype, device=device)

        self._predictor_type = model_config.get('predictor_type', 'ode')
        if self.n_total_control_features > 0:
            if self._predictor_type == "exp":
                raise ValueError("Exponential predictor is not supported for model with control inputs.")

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        if self._predictor_type == "exp":
            return predict_discrete_exp(self, x0, ts, w, **kwargs)
        return predict_discrete(self, x0, ts, w, **kwargs)
