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
# Besides base classes, naming convention: Kernel[A][B][C]
#   A: Sc (scalar) or Op (operator-valued)
#   B: Dp (data-dependent) or nothing (data-independent)
#   C: Specific type of kernel, e.g., RBF, Separable, etc.
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

class KernelDataDependent(ABC):
    """
    Mixin interface for data-dependent kernels (e.g., diffusion maps).
    Implement set_reference_data so solvers can notify once per training set.
    """
    @abstractmethod
    def set_reference_data(self, Xref: torch.Tensor) -> None:
        """
        Prepare data-dependent structures from Xref (N,d).
        Must be differentiable if kernel params are learnable.
        """
        pass


# Drived Bases
class KernelScalarValued(KernelAbstract, ABC):
    def __init__(self, in_dim: int, out_dim: int, dtype=None):
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
    def __init__(self, kernels: Union[KernelScalarValued, List[KernelScalarValued]], out_dim: int, dtype=None):
        if isinstance(kernels, KernelScalarValued):
            kernels = [kernels]
        self.scalar_kernels = kernels
        self.n_kernels = len(kernels)
        self.in_dim = kernels[0].in_dim
        for k in kernels:
            assert isinstance(k, KernelScalarValued)
            assert k.in_dim == self.in_dim

        super().__init__(self.in_dim, out_dim, dtype=dtype)


# Actual kernels
# Data-INDEPENDENT
class KernelScRBF(KernelScalarValued):
    """
    Scalar RBF: k(x,z) = exp(-0.5 * ||x - z||^2 / ell^2)
    Learnable positive lengthscale.
    """
    def __init__(self, in_dim: int, lengthscale_init: float = 1.0, dtype=None):
        super().__init__(in_dim, dtype=dtype)
        self._log_ell = nn.Parameter(torch.tensor(float(lengthscale_init)).log(), dtype=self.dtype)

    @property
    def ell(self):
        # positive via softplus
        return F.softplus(self._log_ell)

    def forward(self, X, Z):
        sq = scaled_cdist(X, Z, self.ell, 2)
        return torch.exp(-0.5 * sq)

class KernelOpSeparable(KernelOperatorValuedScalars):
    """
    Separable operator-valued kernel K(x,z) = sum_i k_i(x,z; ell) * B_i
    where B_i = L_i L_i^T is PSD and learnable.
    Output shape: (N, M, Dy, Dy)
    """
    def __init__(self, kernels: Union[KernelScalarValued, List[KernelScalarValued]], out_dim: int, dtype=None):
        super().__init__(kernels, out_dim, dtype=dtype)

        L0 = torch.stack([torch.eye(out_dim) for _ in range(self.n_kernels)], dim=0)
        self.Ls = nn.Parameter(L0.clone(), dtype=self.dtype)  # (n_kernels, Dy, Dy)

    def forward(self, X, Z):
        k = torch.stack([_k(X, Z) for _k in self.scalar_kernels], dim=0)  # (n_kernels, N, M)
        L = torch.tril(self.Ls)
        B = torch.matmul(L, L.transpose(-1, -2))      # (n_kernels, Dy, Dy)

        # Output: (N, M, Dy, Dy) = sum_i k_i(x,z) * B_i
        out = torch.einsum('i n m, i a b -> n m a b', k, B)
        return out


# Actual kernels
# Data-DEPENDENT
class KernelScDpDM(KernelScalarValued, KernelDataDependent):
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
        self._log_eps = nn.Parameter(torch.tensor(float(eps_init)).log(), dtype=dtype)
        self._log_t   = nn.Parameter(torch.tensor(float(t_init)).log(), dtype=dtype)

        # caches
        self._Xref = None
        self._d_ref = None      # (N,)
        self._U = None          # (N,m)
        self._lam = None        # (m,)
        self._Phi_ref_t = None  # (N,m)

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

class KernelOpDpSeparable(KernelOpSeparable, KernelDataDependent):
    """
    Data-dependent version of SeparableOKernel.
    Output shape: (N, M, Dy, Dy)
    """
    def set_reference_data(self, Xref: torch.Tensor) -> None:
        for _k in self.scalar_kernels:
            if isinstance(_k, KernelDataDependent):
                _k.set_reference_data(Xref)
