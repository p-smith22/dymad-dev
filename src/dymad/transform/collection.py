import logging
import numpy as np
from typing import Any, Dict, List, Union

from dymad.transform.base import DelayEmbedder, Identity, Lift, Scaler, SVD, Transform
from dymad.transform.ndr import DiffMap, DiffMapVB, Isomap

Array = List[np.ndarray]

logger = logging.getLogger(__name__)

class Compose(Transform):
    """Apply transforms in order.  Inverse is applied in reverse."""
    def __init__(self, transforms: List[Transform] = None):
        if transforms is None:
            # Reload from state_dict is expected.
            return

        self.T = transforms
        self._T_names = [str(t) for t in transforms]
        self.NT = len(transforms)

        _n = self._T_names.count("delay")
        if _n > 1:
            raise ValueError(f"Compose: Multiple delay transforms ({_n}) are not allowed. "
                             "Please use only one delay transform in the composition.")
            # This is to reduce bookkeeping complexity in trajectory manager.
        elif _n == 1:
            _k = self._T_names.index("delay")
            self.delay = self.T[_k].delay
        else:
            self.delay = 0

    def __str__(self):
        return "compose"
    
    def _proc_rng(self, rng: Union[List, None]) -> List:
        if rng is not None:
            assert len(rng) == 2, "Range should be a list of two integers [start, end]."
            assert 0 <= rng[0] < rng[1] <= self.NT, f"Range should be within [0, {self.NT}]."
            return rng
        else:
            return [0, self.NT]

    def fit(self, data: Array) -> None:
        """"""
        _d = data
        for t in self.T:
            t.fit(_d)
            _d = t.transform(_d)

        self._inp_dim = self.T[0]._inp_dim
        self._out_dim = self.T[-1]._out_dim

        for _i in range(len(self.T)-1):
            assert self.T[_i]._out_dim == self.T[_i+1]._inp_dim, \
                f"Compose: Output dimension of transform {_i} ({self.T[_i]._out_dim}) " \
                f"does not match input dimension of transform {_i+1} ({self.T[_i+1]._inp_dim})."

    def transform(self, data: Array, rng: Union[List, None] = None) -> Array:
        """"""
        _rng = self._proc_rng(rng)
        for _i in range(_rng[0], _rng[1]):
            data = self.T[_i].transform(data)
        return data

    def inverse_transform(self, data: Array, rng: Union[List, None] = None) -> Array:
        """"""
        _rng = self._proc_rng(rng)
        for _i in range(_rng[1]-1, _rng[0]-1, -1):
            data = self.T[_i].inverse_transform(data)
        return data

    def state_dict(self) -> dict[str, Any]:
        """"""
        return {
            "type": "Compose",
            "names": self._T_names,
            "delay": self.delay,
            "children": [t.state_dict() for t in self.T],
            "inp": self._inp_dim,
            "out": self._out_dim,
            }

    def load_state_dict(self, d) -> None:
        """"""
        logger.info(f"Compose: Loading parameters from checkpoint")
        self._T_names = d["names"]
        self.T = []
        for name, sd in zip(self._T_names, d["children"]):
            if name not in _TRN_MAP:
                raise ValueError(f"Unknown transform type in Compose: {name}")
            self.T.append(_TRN_MAP[name]())
            self.T[-1].load_state_dict(sd)
        self.NT = len(self.T)
        self.delay = d["delay"]
        self._inp_dim = d["inp"]
        self._out_dim = d["out"]

_TRN_MAP = {
    str(Compose()):       Compose,
    str(DelayEmbedder()): DelayEmbedder,
    str(DiffMap()):       DiffMap,
    str(DiffMapVB()):     DiffMapVB,
    str(Identity()):      Identity,
    str(Isomap()):        Isomap,
    str(Lift()):          Lift,
    str(Scaler()):        Scaler,
    str(SVD()):           SVD,
}

def make_transform(config: List[Dict[str, Any]]) -> Transform:
    """
    Create a transform object based on the provided configuration.

    Args:
        config (List[Dict[str, Any]]): List of dictionaries containing transform configurations.

    Returns:
        Transform: An instance of a Transform class.
    """
    if config is None or len(config) == 0:
        return Identity()

    if isinstance(config, dict):
        config = [config]

    transforms = []
    for t in config:
        trn_type = t.get("type", "").lower()
        if trn_type not in _TRN_MAP:
            raise ValueError(f"Unknown transform type: {trn_type}")
        tmp = dict(t)
        tmp.pop("type", None)
        transforms.append(_TRN_MAP[trn_type](**tmp))
    return Compose(transforms)
