import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Dict, Union, Tuple

from dymad.io import DynData
from dymad.models.components import DEC_MAP, ENC_MAP, FZU_MAP
from dymad.models.helpers import get_dims
from dymad.models.model_base import ComposedDynamics, Dynamics, Encoder
from dymad.models.prediction import predict_continuous_np, predict_discrete_exp
from dymad.modules import MLP

class DynCorrAlg(Dynamics):
    def __init__(self, net: nn.Module, func: Callable):
        super().__init__(net)
        self.func = func

    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        if z.ndim == 3:
            w_p = w.p.unsqueeze(-2)
        else:
            w_p = w.p
        _l = self.net(self.features(z, w))
        _f = self.func(z, w.u, _l, w_p)
        return _f

class TemplateCorrAlg(ComposedDynamics):
    """
    Template class for dynamics modeling with algebraic corrections.
    Consider a base dynamics model with parameters p

    - x' = Base_Dynamics(x, u, p)

    The corrected dynamics model with residual force f is given by

    - f = Residual_Force(x, u)
    - x' = Base_Dynamics_With_Correction(x, u, f, p)

    Here the user needs to provide Base_Dynamics_With_Correction that takes
    the residual force as an additional input.
    """
    GRAPH = False
    CONT = None

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__()

        # Dimensions
        dims = get_dims(model_config, data_meta)
        dims['z'] = dims['x']
        dims['s'] = dims['e']
        dims['r'] = dims['x']
        dims['prc'] = model_config.get('residual_layers', 2)

        # Autoencoder
        self.encoder_net = None
        self.decoder_net = None
        self._encoder = ENC_MAP['iden']
        self._decoder = DEC_MAP['iden']

        # Features in the dynamics
        fzu_type = 'cat' if dims['u'] > 0 else 'none'

        # Processor in the dynamics
        opts = {
            'activation'     : model_config.get('activation', 'prelu'),
            'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
            'bias_init'      : model_config.get('bias_init', 'zeros'),
            'gain'           : model_config.get('gain', 1.0),
            'end_activation' : model_config.get('end_activation', True),
            'dtype'          : dtype,
            'device'         : device
        }
        processor_net = MLP(
            input_dim  = dims['s'],
            latent_dim = dims['l'],
            output_dim = dims['r'],
            n_layers   = dims['prc'],
            **opts
        )
        self.dynamics = DynCorrAlg(processor_net, self.base_dynamics)
        self.dynamics.features = FZU_MAP[fzu_type]

        # Prediction options
        self.input_order = model_config.get('input_order', 'cubic')

        assert self.CONT is not None, "CONT flag must be set in derived class."
        if self.CONT:
            self._predict = predict_continuous_np
        else:
            self._predict = predict_discrete_exp

        self.extra_setup()

    def extra_setup(self):
        """
        Additional setup in derived classes.  Called at end of __init__.
        """
        pass

    def base_dynamics(self, x: torch.Tensor, u: torch.Tensor, f: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        The minimal that the user must implement.

        `f` is the residual correction term computed by `self.dynamics.net`, and needs
        to be incorporated into the base dynamics.
        """
        raise NotImplementedError("Implement in derived class.")


class DynCorrDif(Dynamics):
    def __init__(self, net: nn.Module, hidden: nn.Module, func: Callable, dimx: int):
        super().__init__(net)
        self.hidden = hidden
        self.func = func
        self.n_total_state_features = dimx

    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        if z.ndim == 3:
            w_p = w.p.unsqueeze(-2)
        else:
            w_p = w.p
        _x = z[..., :self.n_total_state_features]
        _f = self.net(self.features(z, w))
        _dx = self.func(_x, w.u, _f, w_p)
        _ds = self.hidden(self.features(z, w))
        return torch.cat([_dx, _ds], dim=-1)

def enc_corr_dif_ctrl(self, w: DynData) -> torch.Tensor:
    """Encodes states and controls."""
    return torch.cat(
        [w.x, self.net(torch.cat([w.x, w.u], dim=-1))],
        dim=-1)

def enc_corr_dif_auto(self, w: DynData) -> torch.Tensor:
    """Encodes states."""
    return torch.cat([w.x, self.net(w.x)], dim=-1)

class TemplateCorrDif(ComposedDynamics):
    """
    Template class for dynamics modeling with differential corrections.
    Consider a base dynamics model with parameters p

    - x' = Base_Dynamics(x, u, p)

    Add a hidden dynamics state s for correction

    - s' = Hidden_Dynamics(x, s, u)
    - f = Residual_Force(x, s, u)

    Then full dynamics is

    - z = [x, Encoder(x,u)]
    - x' = Base_Dynamics_With_Correction(x, u, f, p)
    - s' = Hidden_Dynamics(z=[x, s], u)
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

        # Dimensions
        dims = get_dims(model_config, data_meta)
        dims['z'] = dims['x'] + model_config.get('hidden_dimension', 1)
        dims['s'] = dims['z']
        dims['r'] = model_config.get('residual_dimension', 1)
        dims['prc'] = model_config.get('residual_layers', 2)

        # NN options
        opts = {
            'activation'     : model_config.get('activation', 'prelu'),
            'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
            'bias_init'      : model_config.get('bias_init', 'zeros'),
            'gain'           : model_config.get('gain', 1.0),
            'end_activation' : model_config.get('end_activation', True),
            'dtype'          : dtype,
            'device'         : device
        }

        # Autoencoder
        self.encoder_net = MLP(
            input_dim  = dims['e'],
            latent_dim = dims['l'],
            output_dim = dims['z'] - dims['x'],
            n_layers   = dims['enc'],
            **opts
        )
        if dims['u'] > 0:
            self._encoder = enc_corr_dif_ctrl
        else:
            self._encoder = enc_corr_dif_auto

        self.decoder_net = MLP(
            input_dim  = dims['z'],
            latent_dim = dims['l'],
            output_dim = dims['x'],
            n_layers   = dims['dec'],
            **opts
        )
        self._decoder = DEC_MAP['auto']

        # Features in the dynamics
        fzu_type = 'cat' if dims['u'] > 0 else 'none'

        # Processor in the dynamics
        _dim = dims['z'] + dims['u']
        processor_net = MLP(
            input_dim  = _dim,
            latent_dim = dims['l'],
            output_dim = dims['r'],
            n_layers   = dims['prc'],
            **opts
        )
        hidden_net = MLP(
            input_dim  = _dim,
            latent_dim = dims['l'],
            output_dim = dims['z'] - dims['x'],
            n_layers   = model_config.get('hidden_layers', 2),
            **opts
        )
        self.dynamics = DynCorrDif(processor_net, hidden_net, self.base_dynamics, dims['x'])
        self.dynamics.features = FZU_MAP[fzu_type]

        # Prediction options
        self.input_order = model_config.get('input_order', 'cubic')

        assert self.CONT is not None, "CONT flag must be set in derived class."
        if self.CONT:
            self._predict = predict_continuous_np
        else:
            self._predict = predict_discrete_exp

        self.extra_setup()

    def extra_setup(self):
        """
        Additional setup in derived classes.  Called at end of __init__.
        """
        pass

    def base_dynamics(self, x: torch.Tensor, u: torch.Tensor, f: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        The minimal that the user must implement.

        `f` is the residual correction term as output from the hidden dynamics, and needs
        to be incorporated into the base dynamics.
        """
        raise NotImplementedError("Implement in derived class.")
