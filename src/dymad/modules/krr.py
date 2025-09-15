import torch
import torch.nn as nn
import torch.nn.functional as F

from dymad.modules.kernel import KernelDataDependent

def _flatten_block_kernel(K):  # (N,M,Dy,Dy) -> (N*Dy, M*Dy)
    return K.permute(0, 2, 1, 3).reshape(K.size(0) * K.size(2), K.size(1) * K.size(3))

class KRRBase(nn.Module):
    """
    Base class for Kernel Ridge Regression, in particular:

        - Multi-output single scalar kernel (the most common case)
        - Multi-output multiple scalar kernel (i.e., one kernel per output)
        - True operator-valued kernel (i.e., matrix-valued)

    Subclasses must implement:

        - _ensure_solved(self)
        - _predict_from_solution(self, Xnew) -> (M, Dy)
    """
    def __init__(self, ridge_init=1e-2, jitter=1e-8, dtype=None, device=None):
        super().__init__()
        self.jitter = float(jitter)
        # Ridge params are defined by subclasses (scalar, vector, etc.)
        self._ridge_unconstrained = None

        # Train data & caches
        self.X_train = None
        self.Y_train = None
        self._solved = False

        # Other settings
        self.dtype   = dtype
        self.device  = device

    @property
    def ridge(self):
        return F.softplus(self._ridge_unconstrained)

    def set_train_data(self, X: torch.Tensor, Y: torch.Tensor):
        assert X.ndim == 2 and Y.ndim == 2
        self.X_train = torch.tensor(X, dtype=self.dtype, device=self.device)
        self.Y_train = torch.tensor(Y, dtype=self.dtype, device=self.device)
        self._solved = False

        if isinstance(self.kernel, KernelDataDependent):
            self.kernel.set_reference_data(self.X_train)
        if isinstance(self.kernel, list):
            for k in self.kernel:
                if isinstance(k, KernelDataDependent):
                    k.set_reference_data(self.X_train)

        self._on_set_train_data()  # hook for subclasses

    def _on_set_train_data(self):
        pass  # optional in subclasses

    def fit(self):
        """
        Precompute the linear solve, which can be backprop'd.
        """
        self._ensure_solved()

    def forward(self, Xnew: torch.Tensor):
        return self._predict_from_solution(Xnew)

    def _ensure_solved(self):
        raise NotImplementedError("This is the base class.")

    def _predict_from_solution(self, Xnew: torch.Tensor):
        raise NotImplementedError("This is the base class.")

class KRRMultiOutputScalar(KRRBase):
    """
    Scalar KRR for the following cases, with `Dy` outputs:

        - shared=True:

            - Multi-output single kernel (the most common case)
            - One NxN Cholesky; solve `Dy` outputs together. One lambda (scalar) by default.

        - shared=False:

            - Multi-output multiple kernels
            - A ModuleList of `Dy` scalar kernels (one per output).
            - `Dy` independent NxN Choleskys; `Dy` ridges (vector).
    """
    def __init__(self, kernel, ridge_init=1e-2, shared=True, jitter=1e-8, dtype=None, device=None):
        super().__init__(ridge_init=ridge_init, jitter=jitter, dtype=dtype, device=device)
        self.kernel = kernel
        self.shared = bool(shared)

        if self.shared:
            self._ridge_unconstrained = nn.Parameter(torch.tensor(ridge_init).log(),
                                                     dtype=self.dtype, device=self.device)
        else:
            self._ridge_unconstrained = None  # created after seeing Dy

        # caches
        self._alphas = None      # (N, Dy)

    def _on_set_train_data(self):
        self._alphas = None
        self._Ndat, self._Dy = self.Y_train.shape

        if not self.shared:
            # per-output ridge vector (Dy,)
            if self._ridge_unconstrained is None or self._ridge_unconstrained.numel() != self._Dy:
                self._ridge_unconstrained = nn.Parameter(torch.full((self._Dy,), 1e-2).log(),
                                                         dtype=self.dtype, device=self.device)

    def _solve_shared(self):
        X, Y = self.X_train, self.Y_train
        Kxx = self.kernel(X, X)  # (N,N)
        I = torch.eye(self._Ndat, dtype=self.dtype, device=self.device)
        L = torch.linalg.cholesky(Kxx + (self.ridge + self.jitter) * I)
        A = torch.cholesky_solve(Y, L)  # (N,Dy)
        self._alphas = A

    def _solve_per_output(self):
        X, Y = self.X_train, self.Y_train
        assert isinstance(self.kernel, (nn.ModuleList, list)) and len(self.kernel) == self._Dy
        A = torch.empty_like(Y)
        I = torch.eye(self._Ndat, dtype=self.dtype, device=self.device)
        for d in range(self._Dy):
            Kxx = self.kernel[d](X, X)  # (N,N)
            L = torch.linalg.cholesky(Kxx + (self.ridge[d] + self.jitter) * I)
            A[:, d] = torch.cholesky_solve(Y[:, d:d+1], L).squeeze(-1)
        self._alphas = A

    def _ensure_solved(self):
        if self._solved:
            return

        assert self.X_train is not None and self.Y_train is not None, "Call set_train_data first."

        if self.shared:
            self._solve_shared()
        else:
            self._solve_per_output()
        self._solved = True

    def _predict_from_solution(self, Xnew: torch.Tensor):
        M = Xnew.shape[0]
        if self.shared:
            Kxz = self.kernel(Xnew, self.X_train)  # (M,N)
            return Kxz @ self._alphas              # (M,Dy)
        else:
            Yhat = torch.empty((M, self._Dy), device=Xnew.device, dtype=Xnew.dtype)
            for d in range(self._Dy):
                Kxz = self.kernel[d](Xnew, self.X_train)
                Yhat[:, d] = (Kxz @ self._alphas[:, d]).view(-1)
            return Yhat

class KRROperatorValued(KRRBase):
    """
    Operator-valued kernel K(X,Z) -> (N,M,Dy,Dy).

    Solves (Kxx + lambda I) vec(alpha) = vec(Y), using a single (N*Dy)x(N*Dy) Cholesky.
    """
    def __init__(self, kernel, ridge_init=1e-2, jitter=1e-8, dtype=None, device=None):
        super().__init__(ridge_init=ridge_init, jitter=jitter, dtype=dtype, device=device)
        self.kernel = kernel
        self._ridge_unconstrained = nn.Parameter(torch.tensor(ridge_init).log(), dtype=self.dtype, device=self.device)
        self._alpha_vec = None  # (N*Dy,)

    def _on_set_train_data(self):
        self._alpha_vec = None
        self._Ndat, self._Dy = self.Y_train.shape

    def _ensure_solved(self):
        if self._solved:
            return

        assert self.X_train is not None and self.Y_train is not None, "Call set_train_data first."

        X, Y = self.X_train, self.Y_train

        Kxx = self.kernel(X, X)               # (N,N,Dy,Dy)
        Kflat = _flatten_block_kernel(Kxx)    # (N*Dy, N*Dy)
        A = Kflat + self.ridge * torch.eye(self._Ndat * self._Dy, dtype=self.dtype, device=self.device)
        L = torch.linalg.cholesky(A + self.jitter * torch.eye(A.size(0), dtype=self.dtype, device=self.device))
        _tmp = torch.cholesky_solve(Y.reshape(-1, 1), L).squeeze(-1)  # (N*Dy,)

        # _alpha_vec might be used for backprop, so keep as Parameter
        self._alpha_vec = nn.Parameter(_tmp.clone(), requires_grad=True, dtype=self.dtype, device=self.device)

        self._solved = True

    def _predict_from_solution(self, Xnew: torch.Tensor):
        M = Xnew.shape[0]
        Kxz = self.kernel(Xnew, self.X_train) # (M,N,Dy,Dy)
        Kflat = _flatten_block_kernel(Kxz)    # (M*Dy, N*Dy)
        ynew_vec = Kflat @ self._alpha_vec    # (M*Dy,)
        return ynew_vec.reshape(M, self._Dy)
