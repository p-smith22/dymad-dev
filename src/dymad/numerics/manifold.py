import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize_scalar
import scipy.spatial as sps
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)

def tangent_1circle(x):
    _x = np.atleast_2d(x)
    _t = np.arctan2(_x[:,1], _x[:,0])
    _T = np.vstack([np.sin(_t), -np.cos(_t)]).T
    return _T.reshape(len(_x), 1, 2)

def tangent_2torus(x, R):
    _X = np.atleast_2d(x)
    _x, _y, _z = _X[:,0], _X[:,1], _X[:,2]
    _p = np.arctan2(_y, _x)
    _c, _s = np.cos(_p), np.sin(_p)
    _r = np.sqrt(_x**2 + _y**2)
    _T1 = np.vstack([
        -_s,
        _c,
        np.zeros_like(_s)]).T
    _T2 = np.vstack([
        -_z*_c,
        -_z*_s,
        _r-R]).T
    _T2 /= np.linalg.norm(_T2, axis=1).reshape(-1,1)
    _T = np.swapaxes(np.array([_T1, _T2]), 0, 1)
    return _T.reshape(len(_x), 2, 3)

class DimensionEstimator:
    """
    Estimate the intrinsic dimension of a point cloud using kernel method.

    Based on https://doi.org/10.1016/j.acha.2015.01.001
    See Fig. 6

    In implementation, we analytically evaluate

        d(log(S(e)))/d(log(e))
    """
    def __init__(self, data, bracket=[-20, 5], tol=0.2):
        self._data = np.array(data)
        self._Ndat, self._Ndim = self._data.shape

        self._bracket = bracket
        self._tol = tol

        # cache
        self._S   = None
        self._dS  = None
        self._res = None

    def __call__(self):
        # Normalized squared distances
        dst = sps.distance.pdist(self._data)**2
        dmx = np.max(dst)
        dst /= dmx
        # Tuning functions
        def S(e):
            tmp = np.sum(np.exp(- dst / e))*2 + self._Ndat
            return tmp / self._Ndat**2
        def dS(e):
            _k = np.exp(- dst / e)
            _S = 2 * np.sum(_k) + self._Ndat
            _d = 2 * dst.dot(_k)
            return _d/_S/e
        # Find the intrinsic dimension as the max of slope
        func = lambda e: -2*dS(2.**e)
        res = minimize_scalar(func, bracket=self._bracket)
        est = -func(res.x)
        # For fractional dimensions, we do a biased rounding
        if np.ceil(est)-est <= self._tol:
            dim = int(np.ceil(est))
        else:
            dim = int(np.floor(est))
        tmp = (2**res.x * dmx)**(1/est)

        # Key results
        self._dim = dim
        self._est = est
        self._ref_bandwidth = tmp
        self._ref_l2dist = dmx
        self._ref_scalar = 2**res.x

        logger.info(f"Dimension estimation of {self._Ndat} points in {self._Ndim}-dim")
        logger.info(f"Estimated intrinsic dimension: {dim} (est={est:4.3f})")
        logger.info(f"Reference bandwidth: {tmp:4.3e} (l2dist={dmx:4.3e}, scalar={2**res.x:4.3e})")

        # Update cache
        self._S   = S
        self._dS  = dS
        self._res = res

        return dim

    def plot(self, N=20):
        eps = 2.**np.linspace(self._bracket[0], self._bracket[1], N)

        val = [self._S(_e) for _e in eps]
        slp = [2*self._dS(_e) for _e in eps]

        f, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].loglog(eps, val)
        ax[1].semilogx(eps, slp)
        ax[1].semilogx(2**self._res.x, self._est, 'bo', markerfacecolor='none', \
            label=f"Estimated dim: {self._est:4.3f}")
        ax[0].set_ylabel(r'$S(\epsilon)$')
        ax[1].set_xlabel(r'$\epsilon$')
        ax[1].set_ylabel(r'$d$')
        ax[1].legend()
        return f, ax

    def sanity_check(self, K=None, ifref=True, ifnrm=True):
        # Number of kNN points in local PCA
        tmp = int(np.sqrt(self._Ndat))
        Nknn = tmp if K is None else K

        # KD tree for kNN
        _leaf = max(20, Nknn)
        self._tree = sps.KDTree(self._data, leafsize=_leaf)

        # Local PCA
        svs = []
        for _x in self._data:
            _, _i = self._tree.query(_x, k=Nknn)
            _V = self._data[_i] - _x
            _, _s, _ = np.linalg.svd(_V, full_matrices=False)
            svs.append(_s)
        _avr = np.mean(svs, axis=0)
        _std = np.std(svs, axis=0)
        # Global PCA, as reference
        _tmp = self._data - np.mean(self._data, axis=0)
        _, sv, _ = np.linalg.svd(_tmp, full_matrices=False)

        scl = np.max(_avr) if ifnrm else 1.0
        ds = np.arange(len(_avr))+1
        f = plt.figure()
        plt.plot(ds, _avr/scl)
        plt.fill_between(ds, (_avr+_std)/scl, (_avr-_std)/scl, alpha=0.4)
        if ifref:
            scl = np.max(sv) if ifnrm else 1.0
            plt.plot(np.arange(len(sv))+1, sv/scl, 'r--')
        plt.xlabel("SV Index")
        if ifnrm:
            plt.ylabel("Normalized SV")
        else:
            plt.ylabel("SV")

        return (_avr, _std), f

class Manifold:
    def __init__(self, data, d, K=None, g=None, T=None, iforit=False, extT=None):
        self._data = np.array(data)
        self._Ndat, self._Ndim = self._data.shape

        # Number of kNN points in local PCA
        tmp = int(np.sqrt(self._Ndat))
        self._Nknn = tmp if K is None else K

        # KD tree for kNN
        _leaf = max(20, self._Nknn)
        self._tree = sps.KDTree(self._data, leafsize=_leaf)

        # Intrinsic dimension
        self._Nman = d

        # Order of GMLS
        self._Nlsq = 2 if g is None else g
        self._fphi = PolynomialFeatures(self._Nlsq, include_bias=False)
        self._fpsi = PolynomialFeatures(self._Nlsq, include_bias=True) # General GMLS

        # Order of tangent space estimation
        self._Ntan = 0 if T is None else T
        if self._Ntan > 0:
            self._ftau = PolynomialFeatures(self._Ntan, include_bias=False)
            self._estimate_tangent = self._estimate_tangent_ho
        else:
            self._estimate_tangent = self._estimate_tangent_1

        # Orientation of tangent vectors
        self._iforit = iforit

        if extT is None:
            self._ifprecomp = False  # Not precomputed yet
        else:
            self._ifprecomp = True
            self._T = np.array(extT)

        logger.info(f"Manifold info: {self._Ndat} points of {self._Ndim}-dim, intrinsic {self._Nman}-dim")

    def precompute(self):
        logger.info("  Precomputing")
        if self._ifprecomp:
            assert self._T.shape == (self._Ndat, self._Nman, self._Ndim)
            logger.info("  Already done, or T supplied externally; skipping")
            return

        self._T = self._estimate_tangent(self._data)
        if self._iforit:
            logger.info("  Orienting tangent vectors")
            if self._Nman == 1:
                rems = list(range(1,self._Ndat))
                curr = 0
                while len(rems) > 0:
                    _, _i = self._tree.query(self._data[curr], k=self._Nknn)
                    _T = self._T[curr]
                    for _j in _i:
                        if _j in rems:
                            _d = self._T[_j].dot(_T.T)
                            if _d < 0:
                                self._T[_j] = -self._T[_j]
                            rems.remove(_j)
                            curr = _j
            else:
                raise NotImplementedError("Orientation for higher-dim manifold not implemented; \
                    supply oriented T externally")
        self._ifprecomp = True
        logger.info("  Done")

    def gmls(self, x, Y):
        _, _i = self._tree.query(x, k=self._Nknn)
        _T, _V = self._estimate_tangent(x, ret_V=True)
        _B = np.matmul(_V, np.swapaxes(_T, -2, -1))
        _P = self._poly_eval(self._fpsi, _B)
        _C = np.matmul(np.linalg.pinv(_P), Y[_i][..., None])
        _tmp = self._fpsi.fit_transform(np.zeros((1,self._Nman)))
        _r = np.matmul(_tmp, _C)
        return _r.squeeze()

    def _poly_eval(self, f, B):
        # PolynomialFeatures only supports 2D input
        # so we reshape input to 2D and then reshape back
        _s = B.shape[:-1]
        _P = f.fit_transform(B.reshape(-1, self._Nman)).reshape(*_s, -1)
        return _P

    def _estimate_normal(self, base, dx):
        _T, _V = self._estimate_tangent(base, ret_V=True)
        _B = np.matmul(_V, np.swapaxes(_T, -2, -1))
        _P = self._poly_eval(self._fphi, _B)
        _b = np.atleast_2d(np.matmul(dx, np.swapaxes(_T, -2, -1)))
        _p = self._poly_eval(self._fphi, _b)
        _tmp = np.matmul(_p, np.linalg.pinv(_P))
        _n = np.matmul(_tmp, _V - np.matmul(_B, _T))
        return _n.squeeze()

    def _estimate_tangent_1(self, x, ret_V=False):
        _, _i = self._tree.query(x, k=self._Nknn)
        _V = self._data[_i] - x[..., None, :]
        _, _, _Vh = np.linalg.svd(_V, full_matrices=False)
        _T = _Vh.conj()[..., :self._Nman, :]
        if ret_V:
            return _T, _V
        return _T

    def _estimate_tangent_ho(self, x, ret_V=False):
        _T, _V = self._estimate_tangent_1(x, ret_V=True)
        _B = np.matmul(_V, np.swapaxes(_T, -2, -1))
        _P = self._poly_eval(self._ftau, _B)
        _C = np.matmul(np.linalg.pinv(_P), _V - np.matmul(_B, _T))
        _T += _C[..., :self._Nman, :]
        _tmp = np.linalg.qr(np.swapaxes(_T, -2, -1), mode='reduced')[0]
        _T = np.swapaxes(_tmp, -2, -1)
        if ret_V:
            return _T, _V
        return _T

    def plot2d(self, N, scl=1):
        assert self._Ndim == 2
        _d = self._data

        f, ax = plt.subplots(nrows=1, ncols=1)
        plt.plot(_d[:,0], _d[:,1], 'b.', markersize=1)
        for _i in range(N):
            _p = _d[_i] + scl*self._T[_i]
            _c = np.vstack([_d[_i], _p]).T
            plt.plot(_c[0], _c[1], 'k-')
        return f, ax

    def plot3d(self, N, scl=1):
        assert self._Ndim == 3
        _d = self._data

        f = plt.figure()
        ax = f.add_subplot(projection='3d')
        ax.plot(_d[:,0], _d[:,1], _d[:,2], 'b.', markersize=1)
        for _i in range(N):
            _p = _d[_i] + scl*self._T[_i]
            for _j in range(2):
                _c = np.vstack([_d[_i], _p[_j]]).T
                ax.plot(_c[0], _c[1], _c[2], 'k-')
        return f, ax

class ManifoldAnalytical(Manifold):
    def __init__(self, data, d, K=None, g=None, fT=None):
        self._data = np.array(data)
        self._Ndat, self._Ndim = self._data.shape

        # Number of kNN points in local PCA
        tmp = int(np.sqrt(self._Ndat))
        self._Nknn = tmp if K is None else K

        # KD tree for kNN
        _leaf = max(20, self._Nknn)
        self._tree = sps.KDTree(self._data, leafsize=_leaf)

        # Intrinsic dimension
        self._Nman = d

        # Order of GMLS
        self._Nlsq = 2 if g is None else g
        self._fphi = PolynomialFeatures(self._Nlsq, include_bias=False)
        self._fpsi = PolynomialFeatures(self._Nlsq, include_bias=True) # General GMLS

        # Order of tangent space estimation
        self._tangent_func = fT

        # Possible data members
        self._T = []   # Tangent space basis for every data point
        self._ifprecomp = False  # Not precomputed yet

        logger.info(f"Manifold info: {self._Ndat} points of {self._Ndim}-dim, intrinsic {self._Nman}-dim")

    def precompute(self):
        logger.info("  Precomputing")
        self._T = self._estimate_tangent(self._data)
        self._ifprecomp = True
        logger.info("  Done")

    def _estimate_tangent(self, x, ret_V=False):
        _T = self._tangent_func(x)
        if ret_V:
            _, _i = self._tree.query(x, k=self._Nknn)
            _V = self._data[_i] - x[..., None, :]
            return _T, _V
        return _T
