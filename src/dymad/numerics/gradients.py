import numpy as np

def complex_step(f, x, h=1e-20, v=None):
    """Complex step differentiation.

    Args:
        f: function handle that accepts and returns numpy arrays.
        x: input array.
        h: step size.
        v: directions for directional derivative. If None, return full Jacobian.

    Returns:
        df: derivative of f at x, possibly directional.
    """
    # Process x
    x = np.asarray(x).squeeze()
    if np.iscomplexobj(x):
        raise ValueError("Input x should be real.")
    n = x.size
    # Process v
    if v is None:
        v = np.eye(n)
    else:
        v = np.asarray(v)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        assert v.shape[1] == n, "v should have shape (m,n) where n is the size of x."
    # Complex step
    J = []
    for _v in v:
        J.append(np.imag(f(x + 1j*h*_v)) / h)
    return np.array(J).T
