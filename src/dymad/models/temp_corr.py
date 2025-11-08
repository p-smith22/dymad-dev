import numpy as np
import torch
from typing import Dict, Union, Tuple

from dymad.io import DynData
from dymad.models.model_base import ModelBase
from dymad.models.prediction import predict_continuous_np, predict_discrete_exp

class TemplateCorrAlg(ModelBase):
    """
    Base class for dynamics modeling with algebraic corrections.
    Consider a base dynamics model

    - f = Residual_Force(x, u)
    - x_dot/x_next = Base_Dynamics_With_Correction(x, u, f, p)
    """
    GRAPH = None
    CONT = None

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__()
        self.n_total_state_features = data_meta.get('n_total_state_features')
        self.n_total_control_features = data_meta.get('n_total_control_features')
        self.n_total_features = data_meta.get('n_total_features')
        self.latent_dimension = model_config.get('latent_dimension', 64)
        self.residual_dimension = model_config.get('residual_dimension', self.n_total_state_features)
        self.dtype = dtype
        self.device = device

        # Method for input handling
        self.input_order = model_config.get('input_order', 'cubic')

        # Input concatenation
        if self.n_total_control_features == 0:
            self.residual = self._residual_auto
        else:
            self.residual = self._residual_ctrl

        # Cache
        self.residual_net = None

        assert self.CONT is not None, "CONT flag must be set in derived class."

    def diagnostic_info(self) -> str:
        model_info = super().diagnostic_info()
        model_info += f"Dynamics: {self.residual_net}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def _residual_ctrl(self, w: DynData) -> torch.Tensor:
        """
        Map features to latent space for systems with inputs.\
        """
        return self.residual_net(torch.cat([w.x, w.u], dim=-1))

    def _residual_auto(self, w: DynData) -> torch.Tensor:
        """
        Map features to latent space for autonomous systems.
        """
        return self.residual_net(w.x)

    def base_dynamics(self, x: torch.Tensor, u: torch.Tensor, f: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in derived class.")

    def encoder(self, w: DynData) -> torch.Tensor:
        """
        Map features to latent space for autonomous systems.
        """
        return w.x

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Compute latent dynamics (derivative).
        """
        if z.ndim == 3:
            w_p = w.p.unsqueeze(-2)
        else:
            w_p = w.p
        _l = self.residual(w)
        _f = self.base_dynamics(z, w.u, _l, w_p)
        return _f

    def decoder(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Map from latent space back to state space.
        """
        return z

    def forward(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        """
        z = w.x    # Encoder simplified
        z_dot = self.dynamics(z, w)
        x_hat = z  # Decoder simplified
        return z, z_dot, x_hat

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor],
                method: str = 'dopri5', **kwargs) -> torch.Tensor:
        """
        Predict trajectory using CT or DT integration.
        """
        if self.CONT:
            return predict_continuous_np(self, x0, ts, w, method=method, order=self.input_order, **kwargs)
        return predict_discrete_exp(self, x0, ts, w, **kwargs)
