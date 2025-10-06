import numpy as np

from dymad.numerics import complex_step
from dymad.transform import make_transform

def check_data(out, ref, label=''):
    for _s, _t in zip(out, ref):
        assert np.allclose(_s, _t), f"{label} failed: {_s} != {_t}"
    print(f"{label} passed.")

Xs = [
    np.array([
        [1., 2.],
        [1.1, 3.],
        [1.2, 4.],
        [1.3, 5.],
        [1.4, 6.],
        [1.5, 7.]]),
    np.array([
        [2.2, 3.4],
        [2.3, 3.5],
        [2.4, 3.6],
        [2.5, 3.7]])]
Xn = np.array(
    [1.32, 2.4])

def test_modes():
    # ----------
    # Initialize
    mktr = make_transform([
        {'type': 'scaler', 'mode': 'std'},
        {"type": "lift", "fobs": "poly", "Ks": [2, 3]},
        {'type': 'svd', 'order': 2, 'ifcen': True}
    ])
    mktr.fit(Xs)

    # ----------
    # Forward
    forward = lambda x: mktr.transform([x])[0].squeeze()
    modes_f = mktr.get_forward_modes(ref=Xn)
    modes_c = complex_step(forward, Xn)
    assert np.allclose(modes_f, modes_c), f"Forward modes failed: {modes_f} != {modes_c}"

    # ----------
    # Backward
    backward = lambda z: mktr.inverse_transform([z])[0].squeeze()
    modes_f = mktr.get_backward_modes(ref=Xn)
    modes_c = complex_step(backward, Xn).T
    assert np.allclose(modes_f, modes_c), f"Backward modes failed: {modes_f} != {modes_c}"

test_modes()