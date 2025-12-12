import numpy as np
import torch
from typing import Dict, Union, Tuple

from dymad.io import DynData
from dymad.models.temp_ucat import TemplateUCat, TemplateUCatGraphAE
from dymad.models.prediction import predict_continuous, predict_continuous_fenc, predict_discrete, predict_discrete_exp
from dymad.modules import make_krr
from dymad.numerics import Manifold

class KM(TemplateUCat):
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
            'jitter'     : model_config.get('jitter', 1e-12),
            'dtype'      : dtype,
            'device'     : device
        }
        self.dynamics_net = make_krr(**opts)

    def _zu_cat_ctrl(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """Concatenate state and control inputs."""
        return torch.cat([z, w.u], dim=-1)

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor],
                method: str = 'dopri5', **kwargs) -> torch.Tensor:
        return predict_continuous(self, x0, ts, w, method=method, order=self.input_order, **kwargs)

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

M_KEYS = ['data', 'd', 'K', 'g', 'T', 'iforit', 'extT']
class KMM(KM):
    """
    KM with Manifold constraints.

    The model is based on Geometrically constrained KRR,
    The prediction uses the normal correction scheme.

    See more in Huang, He, Harlim & Li ICLR2025.
    """
    GRAPH = False
    CONT  = True

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__(model_config, data_meta, dtype=dtype, device=device)

        self._man_opts = model_config.get('manifold', {})

        # Register buffers for Manifold parameters
        self.register_buffer(f"_m_data", torch.empty(0, dtype=torch.float64))
        self.register_buffer(f"_m_d", torch.empty(0, dtype=torch.int64))
        self.register_buffer(f"_m_K", torch.empty(0, dtype=torch.int64))
        self.register_buffer(f"_m_g", torch.empty(0, dtype=torch.int64))
        self.register_buffer(f"_m_T", torch.empty(0, dtype=torch.int64))
        self.register_buffer(f"_m_iforit", torch.tensor(False, dtype=torch.bool))
        self.register_buffer(f"_m_extT", torch.empty(0, dtype=torch.float64))

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Compute latent dynamics.
        """
        return self.dynamics_net(self._zu_cat(z, w))

    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit the kernel dynamics using input-output pairs.
        """
        # Build manifold from input data
        # This is a Numpy object, and we register buffers to reload it later
        self._manifold = Manifold(inp[:,:self.n_total_state_features], **self._man_opts)
        self._manifold.precompute()
        ts = self._manifold.to_tensors()
        for _k, _v in ts.items():
            setattr(self, f"_m_{_k}", _v)

        # Fit KRR with the manifold constraint
        self.dynamics_net.set_train_data(inp, out)
        self.dynamics_net.set_manifold(self._manifold)
        residual = self.dynamics_net.fit()
        return self.dynamics_net._alphas, residual

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        return predict_continuous_fenc(self, x0, ts, w, **kwargs)

    def fenc_step(self, z: torch.Tensor, w: DynData, dt: float) -> torch.Tensor:
        """
        First-order Euler step with Normal Correction.
        """
        dz = self.dynamics(z, w) * dt
        dn = self._manifold._estimate_normal(z.detach().cpu().numpy(), dz.detach().cpu().numpy())
        return z + dz + torch.as_tensor(dn, dtype=self.dtype, device=z.device)

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        KMM relies on the Numpy-based object Manifold, and
        the defining parameters of the latter are registered as buffers in KMM.
        When self is initialized these buffers are placeholders.
        Here we first update the shapes of those buffers to match the checkpoint,
        then call the standard load_state_dict to load values and do checks.
        In the end we reconstruct the Manifold object from the loaded buffers,
        and set this object in appropriate locations.
        """
        with torch.no_grad():
            for name, p in self.named_buffers(recurse=True):
                if name in state_dict:
                    saved = state_dict[name]
                    if p.shape != saved.shape:
                        p.set_(torch.empty_like(saved))

        res = super().load_state_dict(state_dict, strict=strict)

        t = {_k : getattr(self, f"_m_{_k}") for _k in M_KEYS}
        self._manifold = Manifold.from_tensors(t)
        self.dynamics_net._manifold = self._manifold
        self.dynamics_net.kernel._manifold = self._manifold

        return res

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
        super().__init__(model_config, data_meta, dtype=dtype, device=device)

        self._predictor_type = model_config.get('predictor_type', 'ode')
        if self.n_total_control_features > 0:
            if self._predictor_type == "exp":
                raise ValueError("Exponential predictor is not supported for model with control inputs.")

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        if self._predictor_type == "exp":
            return predict_discrete_exp(self, x0, ts, w, **kwargs)
        return predict_discrete(self, x0, ts, w)

class DKMSK(KM):
    """
    Dynamics based on kernel machine with skip connections in dynamics.
    """
    GRAPH = False
    CONT  = False

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__(model_config, data_meta, dtype=dtype, device=device)

        self._predictor_type = model_config.get('predictor_type', 'ode')
        if self.n_total_control_features > 0:
            if self._predictor_type == "exp":
                raise ValueError("Exponential predictor is not supported for model with control inputs.")

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Compute latent dynamics.
        """
        return z + self.dynamics_net(self._zu_cat(z, w))

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        if self._predictor_type == "exp":
            return predict_discrete_exp(self, x0, ts, w, **kwargs)
        return predict_discrete(self, x0, ts, w)

    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit the kernel dynamics using input-output pairs.
        """
        self.dynamics_net.set_train_data(inp, out-inp[..., :self.kernel_dimension])
        residual = self.dynamics_net.fit()
        return self.dynamics_net._alphas, residual


class GKM(TemplateUCatGraphAE):
    """Graph version of KM.

    Uses GNN encoder/decoder instead of MLP.

    Dynamics is defined per node.
    """
    GRAPH = True
    CONT  = True

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__(model_config, data_meta, dtype=dtype, device=device)
        self.kernel_dimension = model_config.get('kernel_dimension', 16)
        self.const_term = model_config.get('const_term', True)

        # Method for input handling
        self.input_order = model_config.get('input_order', 'cubic')

        self._build_autoencoder(self.kernel_dimension, model_config, dtype, device)

        # Build kernel-based dynamics
        opts = {
            'type'       : model_config.get('type', 'share'),
            'kernel'     : model_config.get('kernel', None),
            'ridge_init' : model_config.get('ridge_init', 1e-10),
            'jitter'     : model_config.get('jitter', 1e-12),
            'dtype'      : dtype,
            'device'     : device
        }
        self.dynamics_net = make_krr(**opts)

    def _zu_cat_ctrl(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """Concatenate state and control inputs."""
        u_reshaped = w.ug
        z_u = (z.unsqueeze(-1) * u_reshaped.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
        if self.const_term:
            return torch.cat([z, z_u, u_reshaped], dim=-1)
        return torch.cat([z, z_u], dim=-1)

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], method: str = 'dopri5', **kwargs) -> torch.Tensor:
        return predict_continuous(
            self, x0, ts, w,
            method=method, order=self.input_order, **kwargs)

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

class DGKM(GKM):
    """Discrete-time version of GKM.
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
        if self._predictor_type == "exp":
            return predict_discrete_exp(self, x0, ts, w, **kwargs)
        return predict_discrete(self, x0, ts, w, **kwargs)

class DGKMSK(GKM):
    """Graph version of DKMSK.
    """
    GRAPH = True
    CONT  = False

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__(model_config, data_meta, dtype=dtype, device=device)

        self._predictor_type = model_config.get('predictor_type', 'ode')
        if self.n_total_control_features > 0:
            if self._predictor_type == "exp":
                raise ValueError("Exponential predictor is not supported for model with control inputs.")

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return z + self.dynamics_net(self._zu_cat(z, w))

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        if self._predictor_type == "exp":
            return predict_discrete_exp(self, x0, ts, w, **kwargs)
        return predict_discrete(self, x0, ts, w, **kwargs)

    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        self.dynamics_net.set_train_data(inp, out-inp[..., :self.kernel_dimension])
        residual = self.dynamics_net.fit()
        return self.dynamics_net._alphas, residual
