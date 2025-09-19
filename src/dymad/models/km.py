import numpy as np
import torch
from typing import Dict, Union, Tuple

from dymad.data import DynData
from dymad.models import ModelTempUCat, ModelTempUCatGraph
from dymad.modules import make_krr
from dymad.utils import predict_continuous, predict_discrete

class KM(ModelTempUCat):
    """
    Kernel machine, where dynamics is given by KRR.
    """
    GRAPH = False
    CONT = True

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__(model_config, data_meta, dtype=dtype, device=device)
        self.kernel_dimension = model_config.get('kernel_dimension', 16)

        self._build_autoencoder(self.kernel_dimension, model_config, dtype, device)

        # Build kernel-based dynamics
        opts = {
            'type'       : model_config.get('type', 'share'),
            'kernel'     : model_config.get('kernel', None),
            'ridge_init' : model_config.get('ridge_init', 1e-10),
            'dtype'      : dtype,
            'device'     : device
        }
        self.dynamics_net = make_krr(**opts)

    def _zu_cat_ctrl(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """Concatenate state and control inputs."""
        return torch.cat([z, w.u], dim=-1)

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor],
                method: str = 'dopri5', **kwargs) -> torch.Tensor:
        return predict_continuous(self, x0, ts, us=w.u, method=method, order=self.input_order, **kwargs)

    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit the kernel dynamics using input-output pairs.
        """
        self.dynamics_net.set_train_data(inp, out)
        residual = self.dynamics_net.fit()
        return self.dynamics_net._alphas, residual

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        KRR relies on data, and when initialized some parameters are placeholders.
        Here we first update the shapes of those parameters to match the checkpoint,
        then call the standard load_state_dict to load values and do checks.
        """
        with torch.no_grad():
            for name, p in self.named_parameters(recurse=True):
                if name in state_dict:
                    saved = state_dict[name]
                    if p.shape != saved.shape:
                        # keep the same Parameter object (so it's still registered,
                        # and any optimizer state tied to id(p) can be preserved)
                        p.set_(torch.empty_like(saved))

        # Then do standard loading and checks
        return super().load_state_dict(state_dict, strict=strict)

class DKM(KM):
    """
    Dynamics based on kernel machine, discrete-time version.

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
        super(DKM, self).__init__(model_config, data_meta, dtype=dtype, device=device)

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        return predict_discrete(self, x0, ts, us=w.u)

class DKMSK(KM):
    """
    Dynamics based on kernel machine with skip connections in dynamics.
    """
    GRAPH = False
    CONT  = False

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super(DKMSK, self).__init__(model_config, data_meta, dtype=dtype, device=device)

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Compute latent dynamics (derivative).
        """
        return z + self.dynamics_net(self._zu_cat(z, w))

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        return predict_discrete(self, x0, ts, us=w.u)

    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit the kernel dynamics using input-output pairs.
        """
        self.dynamics_net.set_train_data(inp, out-inp[..., :self.kernel_dimension])
        residual = self.dynamics_net.fit()
        return self.dynamics_net._alphas, residual
