import numpy as np

from dymad.numerics import complex_step

def test_complex_step():
    np.random.seed(0)
    eps = 1e-10

    # Test 1: Scalar function, scalar input
    f = lambda x: x**2 + 3*x + 2
    x = 1.0
    df_analytical = 2*x + 3
    df_cs = complex_step(f, x)
    assert np.abs(df_cs - df_analytical) < eps, "Scalar func"

    # Test 2: Vector function, vector input
    f = lambda x: np.array([x[0]**2 + x[1], np.sin(x[0]) + np.cos(x[1])])
    x = np.array([1.0, 0.5])
    df_analytical = np.array([[2*x[0], 1], [np.cos(x[0]), -np.sin(x[1])]])
    df_cs = complex_step(f, x)
    assert np.all(np.abs(df_cs - df_analytical) < eps), "Vector func, full"

    # Test 3: Directional derivative
    v = np.array([0.5, 0.5])
    df_analytical = df_analytical @ v.reshape(-1,1)
    df_cs = complex_step(f, x, v=v)
    assert np.all(np.abs(df_cs - df_analytical) < eps), "Vector func, directional"
