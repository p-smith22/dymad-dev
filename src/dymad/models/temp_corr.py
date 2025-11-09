import numpy as np
import torch
from typing import Dict, Union, Tuple

from dymad.io import DynData
from dymad.models.model_base import ModelBase
from dymad.models.prediction import predict_continuous_np, predict_discrete_exp
from dymad.modules import make_autoencoder, MLP

class TemplateCorrAlg(ModelBase):
    """
    Template class for dynamics modeling with algebraic corrections.
    Consider a base dynamics model with parameters p

    - x_dot/x_next = Base_Dynamics(x, u, p)

    The corrected dynamics model with residual force f is given by

    - f = Residual_Force(x, u)
    - x_dot/x_next = Base_Dynamics_With_Correction(x, u, f, p)

    Here the user needs to provide Base_Dynamics_With_Correction that takes
    the residual force as an additional input.
    """
    GRAPH = False
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

        opts = {
            'activation'     : model_config.get('activation', 'prelu'),
            'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
            'bias_init'      : model_config.get('bias_init', 'zeros'),
            'gain'           : model_config.get('gain', 1.0),
            'end_activation' : model_config.get('end_activation', True),
            'dtype'          : dtype,
            'device'         : device
        }
        self.residual_net = MLP(
            input_dim  = self.n_total_features,
            latent_dim = self.latent_dimension,
            output_dim = self.residual_dimension,
            n_layers   = model_config.get('residual_layers', 2),
            **opts
        )

        self.extra_setup()

        assert self.CONT is not None, "CONT flag must be set in derived class."

    def extra_setup(self):
        """
        Additional setup in derived classes.  Called at end of __init__.
        """
        pass

    def diagnostic_info(self) -> str:
        model_info = super().diagnostic_info()
        model_info += f"Residual: {self.residual_net}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def _residual_ctrl(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Map features to residual force for systems with inputs.
        """
        return self.residual_net(torch.cat([z, w.u], dim=-1))

    def _residual_auto(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Map features to residual force for autonomous systems.
        """
        return self.residual_net(z)

    def base_dynamics(self, x: torch.Tensor, u: torch.Tensor, f: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        The minimal that the user must implement.

        `f` is the residual correction term computed by `self.residual`, and needs
        to be incorporated into the base dynamics.
        """
        raise NotImplementedError("Implement in derived class.")

    def encoder(self, w: DynData) -> torch.Tensor:
        """
        Dummy interface.
        """
        return w.x

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Compute corrected dynamics.
        """
        if z.ndim == 3:
            w_p = w.p.unsqueeze(-2)
        else:
            w_p = w.p
        _l = self.residual(z, w.u)
        _f = self.base_dynamics(z, w.u, _l, w_p)
        return _f

    def decoder(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Dummy interface.
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
        Predict trajectory using cont-time or disc-time integration.
        """
        if self.CONT:
            return predict_continuous_np(self, x0, ts, w, method=method, order=self.input_order, **kwargs)
        return predict_discrete_exp(self, x0, ts, w, **kwargs)

class TemplateCorrDif(ModelBase):
    """
    Template class for dynamics modeling with differential corrections.
    Consider a base dynamics model with parameters p

    - x_dot/x_next = Base_Dynamics(x, u, p)

    Add a hidden dynamics state s for correction

    - s_dot/s_next = Hidden_Dynamics(x, s, u)
    - f = Residual_Force(x, s, u)

    Then full dynamics is

    - z = [x, Encoder(x,u)]
    - x_dot/x_next = Base_Dynamics_With_Correction(x, u, f, p)
    - s_dot/s_next = Hidden_Dynamics(z=[x, s], u)
    - f = Residual_Force(z=[x, s], u)
    - x_hat = Decoder(z=[x, s])

    Here the user needs to provide Base_Dynamics_With_Correction that takes
    the residual force as an additional input.

    Due to the special structure, the autoencoder dimensions are fixed,

    - Encoder input: n_total_features (x or x+u)
    - Encoder output: hidden_dimension (s)
    - Decoder input: n_total_state_features + hidden_dimension (x+s)
    - Decoder output: n_total_state_features (x)
    """
    GRAPH = False
    CONT = None

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__()
        self.n_total_state_features = data_meta.get('n_total_state_features')
        self.n_total_control_features = data_meta.get('n_total_control_features')
        self.n_total_features = data_meta.get('n_total_features')
        self.latent_dimension = model_config.get('latent_dimension', 64)
        self.hidden_dimension = model_config.get('hidden_dimension', 1)
        self.residual_dimension = model_config.get('residual_dimension', 1)
        self.dtype = dtype
        self.device = device

        # Method for input handling
        self.input_order = model_config.get('input_order', 'cubic')

        # Input concatenation
        if self.n_total_control_features == 0:
            self.encoder  = self._encoder_auto
            self.residual = self._residual_auto
            self.hidden   = self._hidden_auto
        else:
            self.encoder  = self._encoder_ctrl
            self.residual = self._residual_ctrl
            self.hidden   = self._hidden_ctrl

        # Build components
        opts = {
            'activation'     : model_config.get('activation', 'prelu'),
            'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
            'bias_init'      : model_config.get('bias_init', 'zeros'),
            'gain'           : model_config.get('gain', 1.0),
            'end_activation' : model_config.get('end_activation', True),
            'dtype'          : dtype,
            'device'         : device
        }

        self.encoder_net = MLP(
            input_dim  = self.n_total_features,
            latent_dim = self.latent_dimension,
            output_dim = self.hidden_dimension,
            n_layers   = model_config.get('encoder_layers', 2),
            **opts
        )

        _dim = self.n_total_state_features + self.hidden_dimension + self.n_total_control_features
        self.residual_net = MLP(
            input_dim  = _dim,
            latent_dim = self.latent_dimension,
            output_dim = self.residual_dimension,
            n_layers   = model_config.get('residual_layers', 2),
            **opts
        )

        self.dynamics_net = MLP(
            input_dim  = _dim,
            latent_dim = self.latent_dimension,
            output_dim = self.hidden_dimension,
            n_layers   = model_config.get('hidden_layers', 2),
            **opts
        )

        self.decoder_net = MLP(
            input_dim  = self.n_total_state_features + self.hidden_dimension,
            latent_dim = self.latent_dimension,
            output_dim = self.n_total_state_features,
            n_layers   = model_config.get('decoder_layers', 2),
            **opts
        )

        self.extra_setup()

        assert self.CONT is not None, "CONT flag must be set in derived class."

    def extra_setup(self):
        """
        Additional setup in derived classes.  Called at end of __init__.
        """
        pass

    def diagnostic_info(self) -> str:
        model_info = super().diagnostic_info()
        model_info += f"Encoder: {self.encoder_net.diagnostic_info()}\n"
        model_info += f"Residual: {self.residual_net}\n"
        model_info += f"Hidden: {self.dynamics_net}\n"
        model_info += f"Decoder: {self.decoder_net.diagnostic_info()}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def _encoder_ctrl(self, w: DynData) -> torch.Tensor:
        """
        Map features to latent space for systems with inputs.
        """
        return torch.cat(
            [w.x, self.encoder_net(torch.cat([w.x, w.u], dim=-1))],
            dim=-1)

    def _encoder_auto(self, w: DynData) -> torch.Tensor:
        """
        Map features to latent space for autonomous systems.
        """
        return torch.cat([w.x, self.encoder_net(w.x)], dim=-1)

    def _residual_ctrl(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Map states to residual force for systems with inputs.
        """
        return self.residual_net(torch.cat([z, u], dim=-1))

    def _residual_auto(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Map states to residual force for autonomous systems.
        """
        return self.residual_net(z)

    def _hidden_ctrl(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Dynamics of the hidden state for systems with inputs.
        """
        return self.dynamics_net(torch.cat([z, u], dim=-1))

    def _hidden_auto(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Dynamics of the hidden state for autonomous systems.
        """
        return self.dynamics_net(z)

    def base_dynamics(self, x: torch.Tensor, u: torch.Tensor, f: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        The minimal that the user must implement.

        `f` is the residual correction term as output from the hidden dynamics, and needs
        to be incorporated into the base dynamics.
        """
        raise NotImplementedError("Implement in derived class.")

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Compute corrected dynamics.
        """
        if z.ndim == 3:
            w_p = w.p.unsqueeze(-2)
        else:
            w_p = w.p
        _x = z[..., :self.n_total_state_features]
        _f = self.residual(z, w.u)
        _dx = self.base_dynamics(_x, w.u, _f, w_p)
        _ds = self.hidden(z, w.u)
        return torch.cat([_dx, _ds], dim=-1)

    def decoder(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Map from latent space back to state space.
        """
        return self.decoder_net(z)

    def forward(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        """
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        x_hat = self.decoder(z, w)
        return z, z_dot, x_hat

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor],
                method: str = 'dopri5', **kwargs) -> torch.Tensor:
        """
        Predict trajectory using cont-time or disc-time integration.
        """
        if self.CONT:
            return predict_continuous_np(self, x0, ts, w, method=method, order=self.input_order, **kwargs)
        return predict_discrete_exp(self, x0, ts, w, **kwargs)
