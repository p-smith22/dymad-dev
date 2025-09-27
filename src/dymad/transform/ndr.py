import logging
import numpy as np
import scipy.spatial as sps
import sklearn.manifold as skm
from sklearn.preprocessing import PolynomialFeatures
from typing import Any, List

from dymad.numerics import DM, DMF, Manifold, VBDM
from dymad.transform.base import Transform

Array = List[np.ndarray]

logger = logging.getLogger(__name__)

class TransformKernel(Transform):
    """
    Kernel-based transforms, including some manifold learning methods.
    These are non-parametric and the Transform instance carries the entire dataset.

    Rigorously, the decoding would require solving the pre-image problem.
    Here we implement some approximate methods to bypass the pre-image solver:

        - Pseudo-inverse
        - (modified) Generalized Moving Least Squares (GMLS)

    Args:
        edim: Embedding dimension.
        inverse: Type of inverse transform, pinv or gmls
        Knn: GMLS only - Number of nearest neighbors in reconstruction
             GMLS: Usually same as encoding algorithm
        Kphi: GMLS only - Order of polynomial
        order: Order of SVD truncation.
               GMLS: Suggest the intrinsic dimension
        rcond: Pinv only - Condition number for pinv.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._out_dim = kwargs.pop('edim', None)
        self._inv   = kwargs.pop('inverse', None)
        self._Knn   = kwargs.pop('Knn', None)
        self._Kphi  = kwargs.pop('Kphi', None)
        self._order = kwargs.pop('order', None)
        self._rcond = kwargs.pop('rcond', None)

        self._ndr   = None
        self._tree  = None

    def _prepare_inverse(self):
        """
        To be called after proc_data to create the inverse transform.
        """
        _inv = self._inv.lower()
        if _inv == "gmls":
            self._man = Manifold(self._Z, self._order, K=self._Knn, g=self._Kphi, T=self._Kphi)
            self.inverse_transform = self._gmls
        elif _inv == "pinv":
            self._C = np.linalg.pinv(self._Z, rcond=self._rcond, hermitian=False).dot(self._X)
            self.inverse_transform = self._pinv
        else:
            raise ValueError(f"Unknown inverse transform {self._inv}")

    def _pinv(self, Z):
        return [_Z.dot(self._C) for _Z in Z]

    def _gmls(self, Z):
        return [self._man.gmls(_Z, self._X) for _Z in Z]
    
    def fit(self, X: Array) -> None:
        """"""
        self._make_ndr()

        self._X = np.vstack([_ for _ in X])
        self._inp_dim = self._X.shape[-1]
        self._Z = self._ndr.fit_transform(self._X)

        self._prepare_inverse()

    def transform(self, X: Array) -> Array:
        """"""
        _res = [self._ndr.transform(_X) for _X in X]
        return _res

    def state_dict(self) -> dict[str, Any]:
        """"""
        return {
            "inv":   self._inv,
            "Knn":   self._Knn,
            "Kphi":  self._Kphi,
            "order": self._order,
            "rcond": self._rcond,
            "inp":   self._inp_dim,
            "out":   self._out_dim,
            "X":     self._X,
            "Z":     self._Z,
            }

    def load_state_dict(self, d) -> None:
        """"""
        logger.info(f"{self.__str__}: Loading parameters from checkpoint :{d}")
        self._inv   = d["inv"]
        self._Knn   = d["Knn"]
        self._Kphi  = d["Kphi"]
        self._order = d["order"]
        self._rcond = d["rcond"]
        self._inp_dim = d["inp"]
        self._out_dim = d["out"]
        self._X     = d["X"]
        self._Z     = d["Z"]

class Isomap(TransformKernel):
    """
    Manifold embedding by Isometric Mapping.
    """
    def __str__(self):
        return "isomap"

    def _make_ndr(self) -> None:
        """"""
        self._ndr = skm.Isomap(n_neighbors=self._Knn, n_components=self._out_dim)

    def load_state_dict(self, d) -> None:
        """"""
        super().load_state_dict(d)
        self.fit([d["X"]])
        assert np.allclose(self._Z, d["Z"]), "Loaded Z does not match computed Z"

class DiffMap(TransformKernel):
    """
    Manifold embedding by regular diffusion map.

    Args:
        alpha: for DM.
        epsilon: Kernel scale for DM. If None, it will be estimated.
        mode: 'full' or 'knn'. 'full' uses dense matrix.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._alpha   = kwargs.pop('alpha', 1)
        self._epsilon = kwargs.pop('epsilon', None)
        self._mode    = kwargs.pop('mode', 'full')

    def __str__(self):
        return "dm"

    def _make_ndr(self) -> None:
        """"""
        if self._mode == 'full':
            self._ndr = DMF(
                n_components=self._out_dim, alpha=self._alpha, epsilon=self._epsilon)
        else:
            self._ndr = DM(
                n_components=self._out_dim, n_neighbors=self._Knn,
                alpha=self._alpha, epsilon=self._epsilon)

    def state_dict(self) -> dict[str, Any]:
        """"""
        _d = super().state_dict()
        _d.update({
                "alpha":    self._alpha,
                "epsilon":  self._epsilon,
                "mode":     self._mode,
                "DM":       self._ndr.state_dict()
            })
        return _d

    def load_state_dict(self, d) -> None:
        """"""
        self._alpha   = d["alpha"]
        self._epsilon = d["epsilon"]
        self._mode    = d["mode"]
        super().load_state_dict(d)
        self._make_ndr()
        self._ndr.load_state_dict(d["DM"])
        self._prepare_inverse()

class DiffMapVB(DiffMap):
    """
    Manifold embedding by variable bandwidth diffusion map.

    Args:
        Kb: Number of nearest neighbors for density estimation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._Kb   = kwargs.pop('Kb', None)

    def __str__(self):
        return "vbdm"

    def _make_ndr(self) -> None:
        """"""
        self._ndr = VBDM(
            n_neighbors=self._Knn, n_components=self._out_dim,
            Kb=self._Kb, operator='lb')

    def state_dict(self) -> dict[str, Any]:
        """"""
        _d = super().state_dict()
        _d.update({
                "Kb":     self._Kb,
            })
        return _d

    def load_state_dict(self, d) -> None:
        """"""
        self._Kb = d["Kb"]
        super().load_state_dict(d)
