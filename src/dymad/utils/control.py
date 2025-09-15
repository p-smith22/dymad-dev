from scipy.interpolate import interp1d
import torch
import torch.nn as nn

class ControlInterpolator(nn.Module):
    """
    Interpolates the sampled control signal u(t_k) when the ODE solver
    requests u(t_query).

    Args:
        t (torch.Tensor): 1-D tensor of shape (N,). Sampling times (must be ascending).
        u (torch.Tensor): Tensor of shape (..., N, m). Control samples, m inputs per step.
        order (str): Interpolation mode. One of {'zoh', 'linear', 'cubic', etc}.

    Note:
        Not to be confused with `dymad.utils.sampling._build_interpolant`,
        which is for data generation, esp. with Numpy.
        `ControlInterpolator` is meant to be used in a Torch setting.
    """
    def __init__(self, t, u, order='linear'):
        super().__init__()

        assert u.ndim >= 2, "Control signal must have at least 2 dimensions"

        self.order = order.lower()
        self.register_buffer('t', t)
        self.register_buffer('u', u)

        if self.order == 'zoh':
            self._interp = self._interp_0
        elif self.order == 'linear':
            self._interp = self._interp_1
        else:
            # Assuming option for 'scipy' interpolation
            self._cpu_t  = t.detach().cpu().numpy()
            self._cpu_u  = u.detach().cpu().numpy()
            self._spl    = interp1d(self._cpu_t,
                                    self._cpu_u,
                                    kind=order,
                                    axis=-2,
                                    fill_value="extrapolate",
                                    assume_sorted=True)
            self._interp = self._interp_s

    def forward(self, t_query: torch.Tensor) -> torch.Tensor:
        return self._interp(t_query)

    def _interp_0(self, t_query: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(self.t, t_query).clamp(1, self.t.numel()-1)
        return self.u[..., idx-1, :]

    def _interp_1(self, t_query: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(self.t, t_query).clamp(1, self.t.numel()-1)
        t0, t1   = self.t[idx-1], self.t[idx]
        u0, u1   = self.u[..., idx-1, :], self.u[..., idx, :]
        w        = (t_query - t0) / (t1 - t0)
        return (1. - w) * u0 + w * u1

    def _interp_s(self, t_query: torch.Tensor) -> torch.Tensor:
        uq = self._spl(t_query.detach().cpu().numpy())
        return torch.as_tensor(uq,
                                device=t_query.device,
                                dtype=self.u.dtype)
