import numpy as np
import torch

from dymad.modules import KRRMultiOutputIndep, KRRMultiOutputShared, KRROperatorValued, \
    KernelScRBF, KernelOpSeparable

def func(inp):
    x  = np.atleast_2d(inp)
    y1 = x[:,0]**2 + np.sin(x[:,1])
    y2 = 0.5*y1
    return np.vstack([y1, y2]).T.squeeze()

xx = np.linspace(0, 1, 6)
X, Y = np.meshgrid(xx, xx)
Xtrn = np.vstack([X.reshape(-1), Y.reshape(-1)]).T
Ytrn = func(Xtrn)
Ns = 21
s = np.linspace(0,1,Ns)
S, T = np.meshgrid(s, s)
Xtst = np.vstack([S.reshape(-1), T.reshape(-1)]).T
Ytst = func(Xtst)

def run_krr():
    k_rbf1 = KernelScRBF(in_dim=2, lengthscale_init=1.0)
    k_rbf2 = KernelScRBF(in_dim=2, lengthscale_init=5.0)
    k_opk1 = KernelOpSeparable(
        kernels=k_rbf1, out_dim=2, Ls=np.array([[[1, 0], [0.5, 0]]]))
    k_opk2 = KernelOpSeparable(
        kernels=[k_rbf1, k_rbf1], out_dim=2,
        Ls=np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]]))

    # Baseline
    krr_share = KRRMultiOutputShared(kernel=k_rbf1, ridge_init=1e-10)
    # Should be identical to Baseline
    krr_indp1 = KRRMultiOutputIndep(kernel=[k_rbf1, k_rbf1], ridge_init=1e-10)
    # Some difference in the second component
    krr_indp2 = KRRMultiOutputIndep(kernel=[k_rbf1, k_rbf2], ridge_init=1e-10)
    # Should be very close to Baseline
    krr_opva1 = KRROperatorValued(kernel=k_opk1, ridge_init=1e-10)
    # Should be identical to Baseline
    krr_opva2 = KRROperatorValued(kernel=k_opk2, ridge_init=1e-10)

    krrs = [krr_share, krr_indp1, krr_indp2, krr_opva1, krr_opva2]
    prds = []
    for _krr in krrs:
        _krr.set_train_data(Xtrn, Ytrn)
        _krr.fit()
        with torch.no_grad():
            Yprd = _krr(Xtst).cpu().numpy()
            prds.append(Yprd)
    return prds

def test_krr():
    prds = run_krr()
    ref = np.linalg.norm(Ytst, axis=1)
    ref[ref<1e-3] = 1.0
    for Yprd in prds:
        err = np.linalg.norm(Ytst - Yprd, axis=1) / ref
        assert np.mean(err) < 0.024

    assert np.linalg.norm(prds[0]-prds[1]) < 1e-14
    assert np.linalg.norm(prds[0][:,0]-prds[1][:,0]) < 1e-14
    assert np.linalg.norm(prds[0]-prds[3]) < 1e-9
    assert np.linalg.norm(prds[0]-prds[4]) < 1e-14

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    prds = run_krr()

    mdls = ['Shared', 'Indep1', 'Indep2', 'OpVal1', 'OpVal2']
    stys = ['bo', 'rs', 'g^', 'mv', 'cx']
    Nkrr = len(prds)

    # Mostly overlap, except Indep2 with slightly higher errors in some locations
    f, ax = plt.subplots()
    for m_idx, Yprd_m in enumerate(prds):
        err = np.linalg.norm(Ytst - Yprd_m, axis=1) / np.linalg.norm(Ytst, axis=1)
        ax.semilogy(err, stys[m_idx], label=mdls[m_idx], markerfacecolor='none')
    ax.legend()

    # Visually should be very similar
    _prds = [Ytst] + prds
    _mdls = ['Truth'] + mdls
    f, ax = plt.subplots(nrows=Nkrr+1, ncols=4, sharex=True, sharey=True, figsize=(12,2*Nkrr))
    for m_idx, Yprd_m in enumerate(_prds):
        yt0 = Ytst[:, 0].reshape(Ns, Ns)
        yp0 = Yprd_m[:, 0].reshape(Ns, Ns)
        yt1 = Ytst[:, 1].reshape(Ns, Ns)
        yp1 = Yprd_m[:, 1].reshape(Ns, Ns)

        cs1 = ax[m_idx, 0].contourf(S, T, yp0)
        cs2 = ax[m_idx, 1].contourf(S, T, np.abs(yt0 - yp0))
        ax[m_idx, 2].contourf(S, T, yp1, levels=cs1.levels)
        ax[m_idx, 3].contourf(S, T, np.abs(yt1 - yp1), levels=cs2.levels)

        ax[m_idx, 0].set_ylabel(_mdls[m_idx])
        plt.colorbar(ax[m_idx, 0].collections[0], ax=ax[m_idx], orientation='vertical', fraction=0.025, pad=0.04)

    ax[0, 0].plot(Xtrn[:, 0], Xtrn[:, 1], 'wo', markerfacecolor='none')
    ax[0, 0].set_title('y1')
    ax[0, 1].set_title('y1 error')
    ax[0, 2].set_title('y2')
    ax[0, 3].set_title('y2 error')

    plt.show()