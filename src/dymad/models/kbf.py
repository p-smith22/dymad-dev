import numpy as np
import torch
from typing import Dict, Union

from dymad.io import DynData
from dymad.models.model_temp_ucat import ModelTempUCat, ModelTempUCatGraph
from dymad.models.prediction import predict_continuous, predict_continuous_exp, \
    predict_discrete, predict_discrete_exp
from dymad.modules import FlexLinear

class KBF(ModelTempUCat):
    """
    Koopman Bilinear Form (KBF) model.
    Uses MLP encoder/decoder and KBF operators for dynamics.
    """
    GRAPH = False
    CONT  = True

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__(model_config, data_meta, dtype=dtype, device=device)
        self.koopman_dimension = model_config.get('koopman_dimension', 16)
        self.const_term = model_config.get('const_term', True)

        self._predictor_type = model_config.get('predictor_type', 'ode')
        if self.n_total_control_features > 0:
            if self._predictor_type == "exp":
                raise ValueError("Exponential predictor is not supported for KBF with control inputs.")

        self._build_autoencoder(self.koopman_dimension, model_config, dtype, device)

        # Create KBF operators, concatenated
        if self.n_total_control_features > 0:
            if self.const_term:
                dyn_dim = self.koopman_dimension * (self.n_total_control_features + 1) + self.n_total_control_features
            else:
                dyn_dim = self.koopman_dimension * (self.n_total_control_features + 1)
        else:
            dyn_dim = self.koopman_dimension
        self.dynamics_net = FlexLinear(dyn_dim, self.koopman_dimension, bias=False, dtype=dtype, device=device)

        self.set_linear_weights = self.dynamics_net.set_weights

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.dynamics_net(self._zu_cat(z, w))

    def _zu_cat_ctrl(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        z_u = (z.unsqueeze(-1) * w.u.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
        if self.const_term:
            return torch.cat([z, z_u, w.u], dim=-1)
        return torch.cat([z, z_u], dim=-1)

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor],
                method: str = 'dopri5', **kwargs) -> torch.Tensor:
        if self._predictor_type == "exp":
            return predict_continuous_exp(self, x0, ts, **kwargs)
        return predict_continuous(self, x0, ts, us=w.u, method=method, order=self.input_order, **kwargs)

class DKBF(KBF):
    """Discrete Koopman Bilinear Form (DKBF) model - discrete-time version.

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
        super(DKBF, self).__init__(model_config, data_meta, dtype=dtype, device=device)

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        if self._predictor_type == "exp":
            return predict_discrete_exp(self, x0, ts, **kwargs)
        return predict_discrete(self, x0, ts, us=w.u)

class GKBF(ModelTempUCatGraph):
    """Graph Koopman Bilinear Form (GKBF) model - graph-specific version.
    Uses GNN encoder/decoder and KBF operators for dynamics.

    Koopman dimension is defined per node.
    """
    GRAPH = True
    CONT  = True

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__(model_config, data_meta, dtype=dtype, device=device)
        self.koopman_dimension = model_config.get('koopman_dimension', 16)
        self.const_term = model_config.get('const_term', True)

        # Method for input handling
        self.input_order = model_config.get('input_order', 'cubic')

        self._build_autoencoder(self.koopman_dimension, model_config, dtype, device)

        # Create KBF operators, concatenated
        if self.n_total_control_features > 0:
            if self.const_term:
                dyn_dim = self.koopman_dimension * (self.n_total_control_features + 1) + self.n_total_control_features
            else:
                dyn_dim = self.koopman_dimension * (self.n_total_control_features + 1)
        else:
            dyn_dim = self.koopman_dimension
        self.dynamics_net = FlexLinear(dyn_dim, self.koopman_dimension, bias=False, dtype=dtype, device=device)

        self.set_linear_weights = self.dynamics_net.set_weights

    def _zu_cat_ctrl(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """Concatenate state and control inputs."""
        u_reshaped = w.ug
        z_u = (z.unsqueeze(-1) * u_reshaped.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
        if self.const_term:
            return torch.cat([z, z_u, u_reshaped], dim=-1)
        return torch.cat([z, z_u], dim=-1)

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], method: str = 'dopri5', **kwargs) -> torch.Tensor:
        return predict_continuous(
            self, x0, ts,
            us=w.u, edge_index=w.ei, edge_weights=w.ew, edge_attr=w.ea,
            method=method, order=self.input_order, **kwargs)

class DGKBF(GKBF):
    """Discrete Graph Koopman Bilinear Form (DGKBF) model - discrete-time version.

    Same idea as DKBF vs KBF.
    """
    GRAPH = True
    CONT  = False

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super(DGKBF, self).__init__(model_config, data_meta, dtype=dtype, device=device)

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        return predict_discrete(
            self, x0, ts,
            us=w.u, edge_index=w.ei, edge_weights=w.ew, edge_attr=w.ea)