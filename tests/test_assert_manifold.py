import numpy as np

from dymad.numerics import DimensionEstimator, Manifold, ManifoldAnalytical, \
    tangent_1circle, tangent_2torus

a, b = 1.0, 2.0
def torus_sample(Nsmp):
    tmp = np.random.rand(2,Nsmp)*2*np.pi
    x = (a*np.cos(tmp[0])+b) * np.cos(tmp[1])
    y = (a*np.cos(tmp[0])+b) * np.sin(tmp[1])
    z = a*np.sin(tmp[0])
    tar = np.vstack([x, y, z]).T
    return tar
dat_t2 = torus_sample(2000)

def fit_curve_s1(man, idx):
    base = man._data[idx]
    T = man._T[idx]
    s = np.linspace(0, 1, 11).reshape(-1,1)
    dx = np.matmul(s, T)
    dn = man._estimate_normal(base, dx)
    xs = base[...,None,:] + dx + dn
    return base[...,None,:] + dx, xs

def fit_curve_t2(man, idx):
    base = man._data[idx]
    T = man._T[idx]
    s = np.linspace(0, 1, 11)
    s = np.vstack([s,s]).T
    dx = np.matmul(s, T)
    dn = man._estimate_normal(base, dx)
    xs = base[...,None,:] + dx + dn
    return base[...,None,:] + dx, xs

def run_dim_est():
    est = DimensionEstimator(dat_t2, bracket=[-20, 5], tol=0.2)
    est()
    return est

def test_dim_est():
    est = run_dim_est()
    assert est._dim == 2

def run_tan_s1():
    t = np.random.rand(200)*2*np.pi
    dat = np.vstack([np.cos(t), np.sin(t)]).T
    tan = np.vstack([-np.sin(t), np.cos(t)]).T
    nrm = dat

    cmpNrm = lambda tru, man: np.abs(np.sum(tru*man._T.squeeze(), axis=1))

    m1 = Manifold(dat, d=1, T=0)
    m1.precompute()
    m2 = Manifold(dat, d=1, T=3)
    m2.precompute()
    m3 = Manifold(dat, d=1, T=5, iforit=True)
    m3.precompute()
    m4 = ManifoldAnalytical(dat, d=1, g=4, fT=tangent_1circle)
    m4.precompute()

    res = np.sum(m3._T.squeeze()*tan, axis=1)

    e1 = cmpNrm(nrm, m1)
    e2 = cmpNrm(nrm, m2)
    e3 = cmpNrm(nrm, m3)
    e4 = cmpNrm(nrm, m4)

    return (m1, m2, m3, m4), (e1, e2, e3, e4), res

def test_tan_s1():
    _, es, res = run_tan_s1()

    if res[0] < 0:
        dif = res + 1
    else:
        dif = res - 1

    assert es[0].mean() < 0.04, "Local PCA estimate"
    assert es[1].mean() < 0.0005, "GMLS T=3 estimate"
    assert es[2].mean() < 8e-6, "GMLS T=5 estimate"
    assert es[3].mean() < 1e-15, "Analytical"
    assert dif.mean() < 1e-9, "Orientation of tangent vector"

def run_tan_t2():
    fT = lambda x: tangent_2torus(x, R=2)

    m1 = Manifold(dat_t2, d=2, T=4)
    m1.precompute()
    m2 = ManifoldAnalytical(dat_t2, d=2, fT=fT)
    m2.precompute()

    tmp = np.linalg.det(np.matmul(m1._T, np.swapaxes(m2._T, -2, -1)))

    return (m1, m2), np.array(tmp)

def test_tan_t2():
    _, res = run_tan_t2()
    dif = np.abs(np.abs(res) - 1)
    assert np.max(dif) < 0.003, "GMLS T=4 estimate"

def run_fit_t2():
    dat = dat_t2[:500]
    tar = dat_t2[500:1000]
    Ytrn = np.linalg.norm(np.sin(dat), axis=1)
    Ytst = np.linalg.norm(np.sin(tar), axis=1)
    man = Manifold(dat, d=2, g=4)
    man.precompute()
    Yprd = man.gmls(tar, Ytrn)

    return Ytst, Yprd

def test_fit_t2():
    Ytst, Yprd = run_fit_t2()
    err = np.linalg.norm(Ytst-Yprd)/np.linalg.norm(Ytst)
    assert err < 0.03, "GMLS fit error"

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ifest = 1
    ifts1 = 1
    iftt2 = 1
    iffit = 1

    if ifest:
        est = run_dim_est()
        est.plot()
        dim2 = est.sanity_check()

    if ifts1:
        ms, es, _ = run_tan_s1()

        for _m in ms:
            f, ax = _m.plot2d(8, scl=0.5)
            ax.set_aspect('equal', adjustable='box')

        x1, xs = fit_curve_s1(ms[2], [0, 1])
        for _i in range(2):
            ax.plot(xs[_i, :, 0], xs[_i, :, 1], 'r-')
            ax.plot(x1[_i, :, 0], x1[_i, :, 1], 'b-')

        x1, xs = fit_curve_s1(ms[3], 0)
        ax.plot(xs[:,0], xs[:,1], 'r--')
        ax.plot(x1[:,0], x1[:,1], 'b--')

        f = plt.figure()
        plt.semilogy(es[0], 'b-', label='Local PCA')
        plt.semilogy(es[1], 'r-', label='GMLS T=3')
        plt.semilogy(es[2], 'g-', label='GMLS T=5, orit')
        plt.legend()
        plt.xlabel("Sample index")
        plt.ylabel(r"$n\cdot t$")

    if iftt2:
        ms, res = run_tan_t2()
        dif = np.abs(np.abs(res) - 1)

        for _m in ms:
            f, ax = _m.plot3d(30, scl=0.5)
            ax.plot([-2,2], [-2,2], [-2,2], 'w.')

        x1, xs = fit_curve_t2(ms[-1], [0, 1, 2])
        for _i in range(3):
            ax.plot(xs[_i, :, 0], xs[_i, :, 1], xs[_i, :, 2], 'r-')
            ax.plot(x1[_i, :, 0], x1[_i, :, 1], x1[_i, :, 2], 'b-')

        f = plt.figure()
        plt.semilogy(dif, 'b.')

    if iffit:
        Ytst, Yprd = run_fit_t2()

        f = plt.figure()
        plt.plot(Ytst, 'bo', markerfacecolor='none')
        plt.plot(Yprd, 'rs', markerfacecolor='none')

    plt.show()
