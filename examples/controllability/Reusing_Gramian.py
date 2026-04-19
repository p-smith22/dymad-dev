import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time

# ── SYSTEM ────────────────────────────────────────────────────────────────────
n, m, N    = 4, 2, 20
n_targets  = 30
perturb_r  = 2.0        # radius of random target cloud around z_A
rng        = np.random.default_rng(42)

A = np.diag([0.90, 0.85, 0.80, 0.75])
B = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
N1_base = np.array([[ 0., 1., 0., 0.], [-1., 0., 0., 0.],
                    [ 0., 0., 0., 1.], [ 0., 0.,-1., 0.]])
N2_base = np.array([[ 0., 0., 1., 0.], [ 0., 0., 0., 1.],
                    [-1., 0., 0., 0.], [ 0.,-1., 0., 0.]])

z_0 = 3 * np.array([-3.0,  2.5, -2.0,  1.2])
z_A = 3 * np.array([ 3.0, -2.5,  2.0, -1.2])   # nominal IPOPT target

# Random target cloud — drawn once, same for every ratio:
dirs       = rng.standard_normal((n_targets, n))
dirs      /= np.linalg.norm(dirs, axis=1, keepdims=True)
mags       = rng.uniform(0.5, perturb_r, n_targets)
z_targets  = z_A + dirs * mags[:, None]          # (n_targets, n)

ratios = [0.01, 0.1, 0.5, 1.0, 2.0]

# ── DYNAMICS ──────────────────────────────────────────────────────────────────
def make_N(r):
    s = r * np.linalg.norm(A)
    return [s / np.linalg.norm(N1_base) * N1_base,
            s / np.linalg.norm(N2_base) * N2_base]

def sim(z0, U, Nm):
    z = z0.copy(); traj = [z.copy()]
    for u in U:
        z = A @ z + sum(Nm[i] * u[i] for i in range(m)) @ z + B @ u
        traj.append(z.copy())
    return np.array(traj)

def build_Co_an(z0, Nm):
    # Zero-input analytical Co — used only for warm-starting IPOPT:
    Ap = [np.eye(n)]
    for _ in range(N): Ap.append(Ap[-1] @ A)
    Co = np.zeros((n, N * m))
    for k in range(N):
        Co[:, k*m:(k+1)*m] = Ap[N-1-k] @ B
        for i in range(m):
            Co[:, k*m+i] += Ap[N-1-k] @ Nm[i] @ Ap[k] @ z0
    return Co

def build_Co_traj(z0, U_nom, Nm):
    # Analytical Co linearised along the actual trajectory under U_nom.
    # Propagates sensitivity through A_tilde_k = A + sum_i N_i u_k^i,
    # so it correctly accounts for the bilinear state-control coupling
    # at each step — no finite differences, no numerical noise.
    z  = z0.copy(); zs = [z.copy()]; At = []
    for u in U_nom:
        Ak = A + sum(Nm[i] * u[i] for i in range(m))
        At.append(Ak)
        z  = Ak @ z + B @ u
        zs.append(z.copy())

    # Backward pass: Phi[tau] = A_tilde_{N-1} @ ... @ A_tilde_{tau}
    Phi = [None] * (N + 1)
    Phi[N] = np.eye(n)
    for k in range(N - 1, -1, -1):
        Phi[k] = Phi[k + 1] @ At[k]

    Co = np.zeros((n, N * m))
    for tau in range(N):
        for j in range(m):
            Co[:, tau*m+j] = Phi[tau + 1] @ (B[:, j] + Nm[j] @ zs[tau])
    return Co

def gram_solve(Co, dz):
    # Minimum-norm solution: U = Co^+ dz = Co' (Co Co')^{-1} dz
    lam = np.linalg.lstsq(Co @ Co.T, dz, rcond=None)[0]
    return (Co.T @ lam).reshape(N, m)

def ipopt_solve(z0, U_init, Nm, z_tgt):
    opti  = ca.Opti()
    U_var = opti.variable(N * m)
    z     = ca.DM(z0)
    Nd    = [ca.DM(Ni) for Ni in Nm]
    for k in range(N):
        u_k = U_var[k*m:(k+1)*m]
        z   = ca.DM(A) @ z + sum(Nd[i]*u_k[i] for i in range(m)) @ z + ca.DM(B) @ u_k
    opti.minimize(ca.dot(U_var, U_var))
    opti.subject_to(z == ca.DM(z_tgt))
    opti.set_initial(U_var, U_init.flatten())
    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0,
                          'ipopt.tol': 1e-10, 'ipopt.constr_viol_tol': 1e-10,
                          'ipopt.nlp_scaling_method': 'gradient-based'})
    try:    return np.array(opti.solve().value(U_var)).reshape(N, m)
    except: return U_init

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
# Structure:
#   OFFLINE (once per ratio):
#     1. Solve IPOPT to z_A  →  U_A*, z_A_traj
#     2. Build Co analytically along U_A* trajectory  →  Co_A, W_A = Co_A Co_A'
#     3. Compute eigenvalues of W_A^{-1} for optimality certificate
#
#   ONLINE (per new target, no IPOPT):
#     4. For each z_new: dz = z_new - z_A_end
#                        U_new = U_A* + Co_A^+ dz   (one matrix-vector multiply)
#
#   GROUND TRUTH (for comparison):
#     5. Fresh IPOPT for each z_new

results = {}
for ratio in ratios:
    print(f"\n── ratio = {ratio} ──────────────────────────────")
    Nm = make_N(ratio)

    # ── OFFLINE ──────────────────────────────────────────────────────────────
    t_off0 = time.perf_counter()

    U_w_A  = gram_solve(build_Co_an(z_0, Nm),
                        z_A - sim(z_0, np.zeros((N, m)), Nm)[-1])
    U_A    = ipopt_solve(z_0, U_w_A, Nm, z_A)
    z_A_t  = sim(z_0, U_A, Nm)
    z_end_A = z_A_t[-1]

    # Analytical Co at U_A* — this is the key offline object:
    Co_A   = build_Co_traj(z_0, U_A, Nm)
    W_A    = Co_A @ Co_A.T
    W_inv  = np.linalg.inv(W_A)
    eigvals_A = np.sort(np.linalg.eigvalsh(W_inv))[::-1]

    t_offline = time.perf_counter() - t_off0
    err_A     = np.linalg.norm(z_end_A - z_A)
    print(f"  [offline]  IPOPT err={err_A:.2e}  "
          f"t={t_offline*1e3:.1f} ms  κ(W)={eigvals_A[0]/eigvals_A[-1]:.2e}")

    # ── ONLINE: Gramian reuse for all targets ─────────────────────────────────
    t_on0 = time.perf_counter()
    reuse_trajs, reuse_errs, reuse_costs, reuse_Us = [], [], [], []
    for z_t in z_targets:
        dz   = z_t - z_end_A
        dU   = (Co_A.T @ W_inv @ dz).reshape(N, m)   # Co_A^+ dz
        U_r  = U_A + dU
        z_r  = sim(z_0, U_r, Nm)
        reuse_trajs.append(z_r)
        reuse_Us.append(U_r)
        reuse_errs.append(np.linalg.norm(z_r[-1] - z_t))
        reuse_costs.append(float(np.sum(U_r**2)))
    t_online = time.perf_counter() - t_on0
    print(f"  [online]   reuse  mean_err={np.mean(reuse_errs):.2e}  "
          f"t={t_online*1e3:.1f} ms  ({n_targets} targets)")

    # ── GROUND TRUTH: IPOPT for each target ───────────────────────────────────
    t_gt0 = time.perf_counter()
    gt_trajs, gt_errs, gt_costs, gt_Us = [], [], [], []
    for z_t in z_targets:
        U_w  = gram_solve(build_Co_an(z_0, Nm),
                          z_t - sim(z_0, np.zeros((N, m)), Nm)[-1])
        U_gt = ipopt_solve(z_0, U_w, Nm, z_t)
        z_gt = sim(z_0, U_gt, Nm)
        gt_trajs.append(z_gt)
        gt_Us.append(U_gt)
        gt_errs.append(np.linalg.norm(z_gt[-1] - z_t))
        gt_costs.append(float(np.sum(U_gt**2)))
    t_gt = time.perf_counter() - t_gt0
    print(f"  [ground truth] IPOPT  mean_err={np.mean(gt_errs):.2e}  "
          f"t={t_gt*1e3:.1f} ms  ({n_targets} targets)")
    print(f"  speedup: {t_gt/t_online:.0f}x")

    results[ratio] = dict(
        U_A=U_A, z_A_t=z_A_t, eigvals_A=eigvals_A,
        reuse_trajs=reuse_trajs, reuse_errs=reuse_errs,
        reuse_costs=reuse_costs, reuse_Us=reuse_Us,
        gt_trajs=gt_trajs, gt_errs=gt_errs,
        gt_costs=gt_costs, gt_Us=gt_Us,
        t_offline=t_offline, t_online=t_online, t_gt=t_gt,
    )

# ── PLOT SETTINGS ─────────────────────────────────────────────────────────────
colors  = ['darkorange', 'steelblue', 'seagreen']
nr      = len(ratios)
steps   = np.arange(N + 1)
eig_idx = np.arange(1, n + 1)

# ── FIGURE 1: Trajectories ────────────────────────────────────────────────────
# Show nominal A trajectory, all reuse trajectories (faint), all GT trajectories
# (faint) — one column per ratio, one row per state, plus control rows.
fig1, axes1 = plt.subplots(n + m, nr, figsize=(4 * nr, 11), sharex=True)
fig1.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.06,
                     wspace=0.15, hspace=0.12)

for col, ratio in enumerate(ratios):
    axes1[0, col].set_title(f'$\\|N\\|/\\|A\\|={ratio}$',
                             fontsize=10, fontweight='bold')
for i in range(n):
    axes1[i, 0].set_ylabel(f'$z_{i+1}$')
for j in range(m):
    axes1[n+j, 0].set_ylabel(f'$u_{j+1}$')

for col, ratio in enumerate(ratios):
    res = results[ratio]
    for i in range(n):
        ax = axes1[i, col]
        for k, z_r in enumerate(res['reuse_trajs']):
            ax.plot(steps, z_r[:, i], color=colors[1], lw=0.8, alpha=0.25,
                    label='Reuse' if (col == 0 and i == 0 and k == 0) else None)
        for k, z_g in enumerate(res['gt_trajs']):
            ax.plot(steps, z_g[:, i], color=colors[2], lw=0.8, alpha=0.25,
                    label='IPOPT (GT)' if (col == 0 and i == 0 and k == 0) else None)
        ax.plot(steps, res['z_A_t'][:, i], color=colors[0], lw=2.0,
                label='IPOPT (nominal)' if (col == 0 and i == 0) else None)
        ax.axhline(z_A[i], color='r', ls=':', lw=1.0)
        ax.grid(True, alpha=0.25)

    for j in range(m):
        ax = axes1[n+j, col]
        for U_r in res['reuse_Us']:
            ax.step(np.arange(N), U_r[:, j], where='post',
                    color=colors[1], lw=0.8, alpha=0.25)
        for U_g in res['gt_Us']:
            ax.step(np.arange(N), U_g[:, j], where='post',
                    color=colors[2], lw=0.8, alpha=0.25)
        ax.step(np.arange(N), res['U_A'][:, j], where='post',
                color=colors[0], lw=2.0)
        ax.grid(True, alpha=0.25)
    axes1[-1, col].set_xlabel('Time step')

axes1[0, 0].legend(fontsize=8, loc='best')
fig1.tight_layout()
plt.show()

# ── FIGURE 2: Eigenvalues of W^{-1} at U_A* across ratios ────────────────────
fig2, axes2 = plt.subplots(1, nr, figsize=(4 * nr, 4), sharey=False)
fig2.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.15, wspace=0.15)

for col, ratio in enumerate(ratios):
    ax = axes2[col]
    ax.semilogy(eig_idx, results[ratio]['eigvals_A'],
                color=colors[0], marker='o', markersize=7, linestyle='')
    ax.set_xticks(eig_idx)
    ax.set_xticklabels([f'$\\lambda_{i}$' for i in eig_idx])
    ax.set_title(f'$\\|N\\|/\\|A\\|={ratio}$', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25, which='both')
    if col == 0:
        ax.set_ylabel(r'$\lambda_i(W^{-1})$')

fig2.tight_layout()
plt.show()

# ── FIGURE 3: Metrics across ratios ──────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(12, 4))

mean_reuse_err = [np.mean(results[r]['reuse_errs']) for r in ratios]
mean_gt_err    = [np.mean(results[r]['gt_errs'])    for r in ratios]
mean_reuse_cost = [np.mean(results[r]['reuse_costs']) for r in ratios]
mean_gt_cost    = [np.mean(results[r]['gt_costs'])    for r in ratios]
t_online_all   = [results[r]['t_online'] for r in ratios]
t_gt_all       = [results[r]['t_gt']     for r in ratios]

for ax, (y1, y2, ylabel, title, use_log) in zip(axes3, [
    (mean_reuse_err,  mean_gt_err,
     r'Mean terminal error $\|z_N - z_f\|$', 'Terminal Error',   True),
    (mean_reuse_cost, mean_gt_cost,
     r'Mean control cost $\sum_k \|u_k\|^2$', 'Control Cost',    False),
    (t_online_all,    t_gt_all,
     'Computation time (s)',                   'Computation Time', True),
]):
    ax.plot(ratios, y1, color=colors[1], lw=2, marker='s', ms=6,
            label='Gramian Reuse')
    ax.plot(ratios, y2, color=colors[2], lw=2, marker='^', ms=6,
            label='IPOPT (ground truth)')
    if use_log:
        ax.set_yscale('log')
    ax.set_xlabel(r'$\|N\|/\|A\|$')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig3.tight_layout()
plt.show()