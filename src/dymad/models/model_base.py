import torch
import torch.nn as nn
from typing import Callable, Protocol, Tuple

from dymad.io import DynData


class Encoder(Protocol):
    GRAPH = None
    AUTO = None
    def __init__(self, net: nn.Module): super().__init__(); self.net = net
    def forward(self, w: DynData) -> torch.Tensor: ...

class Dynamics(Protocol):
    GRAPH = None
    zu_cat = None
    def __init__(self, net: nn.Module): super().__init__(); self.net = net
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor: ...
    def linear_eval(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def linear_features(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def set_linear_weights(self, W: torch.Tensor) -> None: ...

class Decoder(Protocol):
    GRAPH = None
    def __init__(self, net: nn.Module): super().__init__(); self.net = net
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor: ...


class ComposedDynamics(nn.Module):
    r"""
    Base class for dynamic models.

    Notation:

    - x: Physical state/observation space; can be time-delayed
    - u: Control input; can be time-delayed
    - z: Embedding (latent) space.

    Discrete-time model:

    - z_k = encoder(x_k, u_k)
    - z_{k+1} = dynamics(z_k, u_k)
    - x_{k+1} = decoder(z_{k+1})

    Continuous-time model:

    - z = encoder(x, u)
    - \dot{z} = dynamics(z, u)
    - x = decoder(z)

    Linear training assumes:

    - linear_targets = dynamics = W @ linear_features(z, u)
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
            predict: Callable):
        super().__init__()
        self.encoder  = encoder
        self.dynamics = dynamics
        self.decoder  = decoder
        self.predict  = predict

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
        raise NotImplementedError("This is the base class.")

    def linear_features(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This is the base class.")

    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This is the base class.")

    def set_linear_weights(self, W: torch.Tensor) -> None:
        raise NotImplementedError("This is the base class.")
