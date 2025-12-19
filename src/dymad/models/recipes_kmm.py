import numpy as np
import torch
from typing import Callable, Union, Tuple

from dymad.io import DynData
from dymad.models.helpers import fzu_selector
from dymad.models.model_base import ComposedDynamics, Decoder, Dynamics, Encoder
from dymad.models.prediction import predict_continuous_fenc
from dymad.modules import make_krr
from dymad.numerics import Manifold


M_KEYS = ['data', 'd', 'K', 'g', 'T', 'iforit', 'extT']
class CD_KMM(ComposedDynamics):
    """
    KM with Manifold constraints.

    The model is based on Geometrically constrained KRR,
    The prediction uses the normal correction scheme.

    See more in Huang, He, Harlim & Li ICLR2025.
    """
    GRAPH = False
    CONT  = True

    def __init__(
            self,
            encoder: Encoder,
            dynamics: Dynamics,
            decoder: Decoder,
            predict: Callable | None = None,
            model_config: dict | None = None):
        super().__init__(encoder, dynamics, decoder, predict, model_config)

        self._man_opts = model_config.get('manifold', {})

        # Register buffers for Manifold parameters
        self.register_buffer(f"_m_data", torch.empty(0, dtype=torch.float64))
        self.register_buffer(f"_m_d", torch.empty(0, dtype=torch.int64))
        self.register_buffer(f"_m_K", torch.empty(0, dtype=torch.int64))
        self.register_buffer(f"_m_g", torch.empty(0, dtype=torch.int64))
        self.register_buffer(f"_m_T", torch.empty(0, dtype=torch.int64))
        self.register_buffer(f"_m_iforit", torch.tensor(False, dtype=torch.bool))
        self.register_buffer(f"_m_extT", torch.empty(0, dtype=torch.float64))

    @classmethod
    def build_core(cls, model_config, enc_type, fzu_type, dec_type, dtype, device, ifgnn=False):
        n_total_control_features = model_config.get('n_total_control_features')
        const_term = model_config.get('const_term', True)

        opts = {
            'type'       : model_config.get('type', 'share'),
            'kernel'     : model_config.get('kernel', None),
            'ridge_init' : model_config.get('ridge_init', 1e-10),
            'jitter'     : model_config.get('jitter', 1e-12),
            'dtype'      : dtype,
            'device'     : device
        }
        dynamics_net = make_krr(**opts)

        fzu_func = fzu_selector(fzu_type, n_total_control_features, const_term)

        return dynamics_net, enc_type, fzu_func, dec_type

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
        self.dynamics.net.set_train_data(inp, out)
        self.dynamics.net.set_manifold(self._manifold)
        residual = self.dynamics.net.fit()
        return self.dynamics.net._alphas, residual

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
        self.dynamics.net._manifold = self._manifold
        self.dynamics.net.kernel._manifold = self._manifold

        return res
