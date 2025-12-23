import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Dict, Tuple, Union

from dymad.io import DynData


class Encoder(nn.Module):
    GRAPH = None
    AUTO = None
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
    def forward(self, w: DynData) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

class Dynamics(nn.Module):
    GRAPH = None
    features = None
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

class Decoder(nn.Module):
    GRAPH = None
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")


class ComposedDynamics(nn.Module):
    r"""
    Base class for dynamic models.

    Notation:

    - x: Physical state/observation space; can be time-delayed
    - u: Control input; can be time-delayed
    - z: Embedded space, where dynamics is learned.
    - s: Features for dynamics, composing z and u as needed
    - r: Output of processor, might be lower dimensional than z
    - z': next step of z (discrete-time) or z_dot (continuous-time)

    Full model:

    - z = encoder(x, u)
    - z' = dynamics(z, u)
    - x = decoder(z)

    Details in dynamics:

    - s = features(z, u); e.g., concatenation of linear or bilinear terms
    - r = processor(s, u); e.g., NN or linear transform
    - z' = composition(s, r); Direct or Skip-Connection

    Linear training assumes:

    - processor is linear
    - linear_targets = r = W @ features(z, u)
    - and fits W only

    Signature for predict:

    - `predict(x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> Tuple[torch.Tensor, torch.Tensor]`
    - Usually comes from `dymad/models/prediction`
    """
    GRAPH = None   # True for graph compatible models
    CONT  = None   # True for continuous-time models, otherwise discrete-time

    def __init__(
            self,
            encoder: Encoder,
            dynamics: Dynamics,
            decoder: Decoder,
            predict: Tuple[Callable, str] | None = None,
            model_config: Dict | None = None,
            dims: Dict | None = None):
        super().__init__()
        self.encoder  = encoder
        self.dynamics = dynamics
        self.decoder  = decoder
        if predict is not None:
            self._predict, self.input_order = predict
        # else use the default predict method

        self.n_total_state_features = dims['x']
        self.latent_dimension = dims['z']

        self.dtype  = next(self.parameters()).dtype
        self.device = next(self.parameters()).device

    @classmethod
    def build_core(cls, model_config, dtype, device, ifgnn=False):
        raise NotImplementedError("This is the base class.")

    def diagnostic_info(self) -> str:
        """
        Return diagnostic information about the model.

        Returns:
            str: String with model details
        """
        return f"Model parameters: {sum(p.numel() for p in self.parameters())}\n" + \
               f"Encoder: {self.encoder}\n" + \
               f"Dynamics: {self.dynamics}\n" + \
               f"Decoder: {self.decoder}\n"

    def forward(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        x_hat = self.decoder(z, w)
        return z, z_dot, x_hat

    def linear_eval(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear evaluation, dz, and states, z, for the model.

        dz = Af(z)

        z is the encoded state, which will be used to compute the expected output.
        """
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        return z_dot, z

    def linear_features(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear features, f, and outputs, dz, for the model.

        dz = Af(z)

        dz is the output of the dynamics, z_dot for cont-time, z_next for disc-time.
        """
        z = self.encoder(w)
        return self.dynamics.features(z, w), z

    def set_linear_weights(self,
        W: torch.Tensor | None = None, b: torch.Tensor | None = None,
        U: torch.Tensor | None = None, V: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Set the weights of the linear dynamics module."""
        return self.dynamics.net.set_weights(W, b, U, V)

    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This is the base class.")

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor],
                method= 'dopri5', **kwargs) -> torch.Tensor:
        return self._predict(self, x0, ts, w, method=method, order=self.input_order, **kwargs)
