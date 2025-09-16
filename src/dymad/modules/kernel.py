from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

# --------------------
# Utils
# --------------------
def scaled_cdist(X: torch.Tensor, Z: torch.Tensor, scale: Union[float, torch.Tensor], p: float) -> torch.Tensor:
    """
    Pairwise distance ||X/scale - Z/scale||^p with broadcasting-friendly scaling.

    Args:
        X (torch.Tensor): (N,d)
        Z (torch.Tensor): (M,d)
        scale (float or torch.Tensor): (d,) or scalar, positive
        p (float): order of the norm
    """
    Xn, Zn = X / scale, Z / scale
    dists = torch.cdist(Xn, Zn, p=p)  # (N,M)
    return dists

# --------------------
# Kernels
#
# Besides base classes, naming convention: Kernel[A][B]
#   A: Sc (scalar) or Op (operator-valued)
#   B: Specific type of kernel, e.g., RBF, Separable, etc.
# --------------------

# Bases
class KernelAbstract(nn.Module, ABC):
    """
    Base interface for all kernels (scalar or operator-valued).
    """
    def __init__(self, in_dim: int, dtype=None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.dtype = dtype if dtype is not None else torch.float64

    @abstractmethod
    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel between X (N,d) and Z (M,d).
        Returns:
          - Scalar kernels: (N, M)
          - Operator-valued kernels: (N, M, Dy, Dy)
        """
        pass

    @property
    @abstractmethod
    def is_operator_valued(self) -> bool:
        """True for operator-valued kernels; False for scalar kernels."""
        pass

    def set_reference_data(self, Xref: torch.Tensor) -> None:
        """
        Prepare data-dependent structures from Xref (N,d).
        Must be differentiable if kernel params are learnable.

        By default the kernel is data-independent and does nothing.
        """
        pass


# Drived Bases
class KernelScalarValued(KernelAbstract, ABC):
    def __init__(self, in_dim: int, dtype=None):
        super().__init__(in_dim, dtype=dtype)
        self.out_dim = 1

    @property
    def is_operator_valued(self) -> bool:
        return False

class KernelOperatorValued(KernelAbstract, ABC):
    def __init__(self, in_dim: int, out_dim: int, dtype=None):
        super().__init__(in_dim, dtype=dtype)
        self.out_dim = int(out_dim)

    @property
    def is_operator_valued(self) -> bool:
        return True

class KernelOperatorValuedScalars(KernelOperatorValued):
    """
    Operator-valued kernel induced by scalar kernels
    Output shape: (N, M, Dy, Dy)
    """
    def __init__(self, kernels: Union[KernelScalarValued, List], out_dim: int, dtype=None):
        if isinstance(kernels, KernelScalarValued):
            kernels = nn.ModuleList([kernels])
        elif isinstance(kernels, list):
            kernels = nn.ModuleList(kernels)
        self.n_kernels = len(kernels)
        self.in_dim = kernels[0].in_dim
        for k in kernels:
            assert isinstance(k, KernelScalarValued)
            assert k.in_dim == self.in_dim

        super().__init__(self.in_dim, out_dim, dtype=dtype)
        self.scalar_kernels = kernels

    def set_reference_data(self, Xref: torch.Tensor) -> None:
        for _k in self.scalar_kernels:
            _k.set_reference_data(Xref)

# Actual kernels
## Scalar kernels
class KernelScRBF(KernelScalarValued):
    """
    Scalar RBF: k(x,z) = exp(-0.5 * ||x - z||^2 / ell^2)
    Learnable positive lengthscale.
    """
    def __init__(self, in_dim: int, lengthscale_init: float = 1.0, dtype=None):
        super().__init__(in_dim, dtype=dtype)
        self._log_ell = nn.Parameter(torch.tensor(float(lengthscale_init), dtype=self.dtype).log())

    def __repr__(self) -> str:
        return f"KernelScRBF(in_dim={self.in_dim}, ell={self.ell.item():.4f}, dtype={self.dtype})"

    @property
    def ell(self):
        # positive via softplus
        return F.softplus(self._log_ell)

    def forward(self, X, Z):
        sq = scaled_cdist(X, Z, self.ell, 2)
        return torch.exp(-0.5 * sq)

class KernelScDM(KernelScalarValued):
    """
    Symmetric-normalized diffusion kernel via diffusion maps.

    Steps:
      - Build K_eps on Xref with Gaussian affinity using ε (learnable).
      - Symmetric normalize: A = D^{-1/2} K D^{-1/2}
      - Eigendecompose: A = U Λ U^T (take top-m)
      - Features Φ_t = D^{-1/2} U Λ^{t/2}, where t>0 (learnable)
      - Kernel: k_t(x,z) = Φ_t(x) · Φ_t(z); out-of-sample via Nyström.

    Everything keeps autograd for ε and t.
    """
    def __init__(self, in_dim: int, n_eigs: int = 64, eps_init: float = 1.0, t_init: float = 1.0, jitter: float = 1e-9, dtype=None):
        super().__init__(in_dim)
        self.n_eigs = int(n_eigs)
        self.jitter = float(jitter)
        self._log_eps = nn.Parameter(torch.tensor(float(eps_init), dtype=self.dtype).log())
        self._log_t   = nn.Parameter(torch.tensor(float(t_init), dtype=self.dtype).log())

        # caches
        self._Xref = None
        self._d_ref = None      # (N,)
        self._U = None          # (N,m)
        self._lam = None        # (m,)
        self._Phi_ref_t = None  # (N,m)

    def __repr__(self) -> str:
        return f"KernelScDM(in_dim={self.in_dim}, n_eigs={self.n_eigs}, eps={self.eps.item():.4f}, t={self.t.item():.4f}, dtype={self.dtype})"

    @property
    def eps(self):  # ε > 0
        return F.softplus(self._log_eps)

    @property
    def t(self):    # t > 0
        return F.softplus(self._log_t)

    def _gauss_affinity(self, X, Z):
        # K_eps = exp(-||x-z||^2 / (4 ε))  (scaled so that bandwidth uses eps directly)
        scale = (2.0 * self.eps).sqrt()
        sq = scaled_cdist(X, Z, scale, 2)
        return torch.exp(-sq)

    def set_reference_data(self, Xref: torch.Tensor) -> None:
        self._Xref = Xref
        N = Xref.shape[0]

        K = self._gauss_affinity(Xref, Xref)      # (N,N), symmetric
        d = K.sum(dim=1).clamp_min(self.jitter)   # (N,)
        Dinvs = d.pow(-0.5)
        A = Dinvs[:, None] * K * Dinvs[None, :]   # symmetric normalization
        A = 0.5 * (A + A.T)

        lam, U = torch.linalg.eigh(A)             # ascending
        m = min(self.n_eigs, N)
        lam_m = lam[-m:].clamp_min(self.jitter)   # (m,)
        U_m   = U[:, -m:]                         # (N,m)

        lt = lam_m.pow(self.t / 2.0)              # (m,)
        Phi_ref_t = (Dinvs[:, None] * U_m) * lt[None, :]  # (N,m)

        self._d_ref = d
        self._U = U_m
        self._lam = lam_m
        self._Phi_ref_t = Phi_ref_t

    def _features(self, X: torch.Tensor) -> torch.Tensor:
        assert self._Xref is not None, "Call set_reference_data(X_train) before using the kernel."
        if X.data_ptr() == self._Xref.data_ptr():
            return self._Phi_ref_t

        R = self._gauss_affinity(X, self._Xref)            # (M,N)
        d_new = R.sum(dim=1).clamp_min(self.jitter)        # (M,)
        A_new_ref = R / (d_new[:, None].sqrt() * self._d_ref[None, :].sqrt())

        U_new = (A_new_ref @ self._U) / self._lam          # (M,m)
        lt = self._lam.pow(self.t / 2.0)                   # (m,)
        Phi_new_t = (d_new.pow(-0.5)[:, None] * U_new) * lt[None, :]
        return Phi_new_t

    def forward(self, X, Z):
        PhiX = self._features(X)   # (N,m)
        PhiZ = self._features(Z)   # (M,m)
        return PhiX @ PhiZ.T       # (N,M)


## Operator kernels
class KernelOpSeparable(KernelOperatorValuedScalars):
    """
    Separable operator-valued kernel K(x,z) = sum_i k_i(x,z; ell) * B_i
    where B_i = L_i L_i^T is PSD and learnable.
    Output shape: (N, M, Dy, Dy)
    """
    def __init__(self,
                 kernels: Union[KernelScalarValued, List], out_dim: int,
                 Ls: Union[torch.Tensor, List[torch.Tensor]]=None, dtype=None):
        super().__init__(kernels, out_dim, dtype=dtype)

        if Ls is None:
            L0 = torch.stack([torch.eye(out_dim, dtype=self.dtype) for _ in range(self.n_kernels)], dim=0)
            self.Ls = nn.Parameter(L0.clone())  # (n_kernels, Dy, Dy)
        else:
            Ls = torch.atleast_3d(torch.as_tensor(Ls, dtype=self.dtype))
            assert Ls.ndim == 3
            assert Ls.shape[0] == self.n_kernels and Ls.shape[1] == out_dim and Ls.shape[2] == out_dim
            self.Ls = nn.Parameter(Ls.clone())

    def __repr__(self) -> str:
        _s = [self.scalar_kernels[i].__repr__() for i in range(self.n_kernels)]
        return f"KernelOpSeparable(in_dim={self.in_dim}, out_dim={self.out_dim}, n_kernels={self.n_kernels}, dtype={self.dtype})\n" \
               f"\t\tLs_shapes={[self.Ls.shape]}\n\twith:\n\t\t" + "\n\t\t".join(_s)

    def forward(self, X, Z):
        k = torch.stack([_k(X, Z) for _k in self.scalar_kernels], dim=0)  # (n_kernels, N, M)
        L = torch.tril(self.Ls)
        B = torch.matmul(L, L.transpose(-1, -2))      # (n_kernels, Dy, Dy)

        # Output: (N, M, Dy, Dy) = sum_i k_i(x,z) * B_i
        out = torch.einsum('i n m, i a b -> n m a b', k, B)
        return out