import numpy as np
import scipy.spatial as sps
import sklearn.manifold as skm
from sklearn.preprocessing import PolynomialFeatures
from typing import List

from dymad.numerics import DM, DMF, VBDM
from dymad.transform.base import Transform

Array = List[np.ndarray]

class TransformKernel(Transform):
    """
    Kernel-based transforms, including some manifold learning methods.
    These are non-parametric and the Transform instance carries the entire dataset.

    Rigorously, the decoding would require solving the pre-image problem.
    Here we implement some approximate methods to bypass the pre-image solver:

        - Pseudo-inverse
        - (modified) Generalized Moving Least Squares (GMLS)

    Args:
        inverse: Type of inverse transform, pinv or gmls
        Knn: Number of nearest neighbors in reconstruction
             GMLS: Usually same as encoding algorithm
        Kphi: GMLS only - Order of polynomial
        order: Order of SVD truncation.
               GMLS: Suggest the intrinsic dimension
        rcond: Pinv only - Condition number for pinv.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._inv   = kwargs.pop('inverse', None)
        self._Knn   = kwargs.pop('Knn', None)
        self._Kphi  = kwargs.pop('Kphi', None)
        self._order = kwargs.pop('order', None)
        self._rcond = kwargs.pop('rcond', None)

        self._tree  = None

    def _prepare_inverse(self):
        """
        To be called after proc_data to create the inverse transform.
        """
        _inv = self._inv.lower()
        if _inv == "gmls":
            assert self._tree is not None
            self._fphi = PolynomialFeatures(self._Kphi, include_bias=True)
            self.inverse_transform = self._gmls
        elif _inv == "pinv":
            _Z = np.vstack(self._Zs)
            self._C = np.linalg.pinv(_Z, rcond=self._rcond, hermitian=False).T.dot(self._X)
            self.inverse_transform = self._pinv
        else:
            raise ValueError(f"Unknown inverse transform {self._inv}")

    def _pinv(self, Z):
        return Z.dot(self._C)

    def _gmls(self, s):
        _s = np.real(s)
        _, _i = self._tree.query(_s, k=self._Knn)

        # Arc lengths in intrinsic space
        _U = self._Z[_i] - self._Z[_i[0]]
        _, _, _Vh = np.linalg.svd(_U, full_matrices=False)
        _V = _Vh[:self._order].conj().T
        _t = _U.dot(_V)
        _c = (_s-self._Z[_i[0]]).dot(_V).reshape(1,-1)
        _mn, _mx = np.min(_t), np.max(_t)
        _t = (_t-_mn) / (_mx-_mn)
        _c = (_c-_mn) / (_mx-_mn)

        # Fit in physical space
        _P = self._fphi.fit_transform(_t)
        _C = np.linalg.pinv(_P).dot(self._X[_i])
        _p = self._fphi.fit_transform(_c)
        _X = _p.dot(_C).T.squeeze()

        return _X

class Isomap(TransformKernel):
    """
    Manifold embedding by Isometric Mapping.

    Args:
        edim: Embedding dimension.
        Kiso: Number of nearest neighbors.
        offset: Whether to include constant.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Ncmp = kwargs.pop('edim', None)
        self._Kiso = kwargs.pop('Kiso', None)
        self._off  = kwargs.pop('offset', False)
        self._Ncmp = Ncmp
        if self._off:
            self._out_dim = self._Ncmp + 1
        else:
            self._out_dim = self._Ncmp

    def __str__(self):
        return "isomap"

    def fit(self, X: Array) -> None:
        """"""
        self._isom = skm.Isomap(n_neighbors=self._Kiso, n_components=self._Ncmp)

        _X = np.vstack([_ for _ in X])
        self._inp_dim = _X.shape[-1]

        _Z = self._isom.fit_transform(_X)
        if self._off:
            _one = np.ones((len(_Z),1))
            _Z = np.hstack([_one, _Z])
        self._tree = sps.KDTree(_Z, leafsize=self._Kiso)

        self._prepare_inverse()

    def transform(self, X: Array) -> Array:
        """"""
        _res = self._isom.transform(X.reshape(-1,self._Ninp))
        if self._off:
            _one = np.ones((len(_res),1))
            return np.hstack([_one, _res])
        return _res

class DiffMap(TransformKernel):
    """
    Manifold embedding by regular diffusion map.

    Args:
        edim: Embedding dimension.
        Kdm: Number of nearest neighbors.
        Kb: Number of nearest neighbors for density estimation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._out_dim = kwargs.pop('edim', None)
        self._Kdm  = kwargs.pop('Kdm', None)
        self._Kb   = kwargs.pop('Kb', None)

    def __str__(self):
        return "dm"
    
    def fit(self, X: Array) -> None:
        """"""
        if self._Kdm is None:
            self._dm = DMF(
                n_components=self._out_dim, Kb=self._Kb, operator='lb')
        else:
            self._dm = DM(
                n_neighbors=self._Kdm, n_components=self._out_dim,
                Kb=self._Kb, operator='lb')

        _X = np.vstack([_ for _ in X])
        self._inp_dim = _X.shape[-1]

        self._dm.fit(_X)
        self._tree = sps.KDTree(self._dm._psi, leafsize=self._Kdm)

        self._prepare_inverse()

    def transform(self, X: Array) -> Array:
        """"""
        return self._dm.transform(X.reshape(-1,self._inp_dim))

class DiffMapVB(DiffMap):
    """
    Manifold embedding by variable bandwidth diffusion map.

    Args:
        edim: Embedding dimension.
        Kdm: Number of nearest neighbors.
        Kb: Number of nearest neighbors for density estimation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._out_dim = kwargs.pop('edim', None)
        self._Kdm  = kwargs.pop('Kdm', None)
        self._Kb   = kwargs.pop('Kb', None)

    def __str__(self):
        return "vbdm"
    
    def fit(self, X: Array) -> None:
        """"""
        self._vbdm = VBDM(
            n_neighbors=self._Kdm, n_components=self._out_dim,
            Kb=self._Kb, operator='lb')

        _X = np.vstack([_ for _ in X])
        self._inp_dim = _X.shape[-1]

        self._vbdm.fit(_X)
        self._tree = sps.KDTree(self._vbdm._psi, leafsize=self._Kdm)

        self._prepare_inverse()
