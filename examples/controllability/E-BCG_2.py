# Import packages:
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time

# === DEFINE SYSTEM ===
# Define shapes:
n, m = 4, 2

# Define system matrices:
A       = np.diag([0.90, 0.85, 0.80, 0.75])
B       = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
N1_base = np.array([[ 0., 1., 0., 0.], [-1., 0., 0., 0.],
                    [ 0., 0., 0., 1.], [ 0., 0.,-1., 0.]])
N2_base = np.array([[ 0., 0., 1., 0.], [ 0., 0., 0., 1.],
                    [-1., 0., 0., 0.], [ 0.,-1., 0., 0.]])
C = np.eye(n)

# Define trajectory:
z_0 = 2 * np.array([-3.0,  2.5, -2.0,  1.2])
z_f = 2 * np.array([ 3.0, -2.5,  2.0, -1.2])
N   = 20

# Declare ratios we will sweep through:
ratios = [0.0, 1.0, 2.0, 5.0, 10.0]

# === BILINEAR DYNAMICS ===
def make_N_mats(ratio):
    s = ratio * np.linalg.norm(A)
    return [s / np.linalg.norm(N1_base) * N1_base,
            s / np.linalg.norm(N2_base) * N2_base]

def bilinear_step(z, u, N_mats):
    return A @ z + sum(N_mats[i] * u[i] for i in range(m)) @ z + B @ u

def simulate(z0, U, N_mats):
    z = z0.copy(); traj = [z.copy()]
    for u in U:
        z = bilinear_step(z, u, N_mats); traj.append(z.copy())
    return np.array(traj)

# === FIRST-ORDER FD (Jacobian) ===
def build_jacobian(z0, U_nom, N_mats, eps=1e-4):
    N_p = len(U_nom)
    Co  = np.zeros((n, N_p * m))
    for tau in range(N_p):
        for j in range(m):
            up = U_nom.copy(); up[tau, j] += eps
            un = U_nom.copy(); un[tau, j] -= eps
            Co[:, tau*m+j] = C @ (simulate(z0, up, N_mats)[-1]
                                  - simulate(z0, un, N_mats)[-1]) / (2*eps)
    dy = C @ z_f - C @ simulate(z0, U_nom, N_mats)[-1]
    return Co, dy

# === SECOND-ORDER FD ===
def build_hessian_sparse(z0, U_nom, N_mats, eps=1e-4):
    N_p = len(U_nom)
    H   = {}
    for tau in range(N_p):
        for j in range(m):
            for tau2 in range(tau, min(tau + 2, N_p)):
                for j2 in range(m):
                    if tau2 == tau and j2 < j:
                        continue
                    up_up = U_nom.copy(); up_up[tau,j]  += eps; up_up[tau2,j2] += eps
                    up_un = U_nom.copy(); up_un[tau,j]  += eps; up_un[tau2,j2] -= eps
                    un_up = U_nom.copy(); un_up[tau,j]  -= eps; un_up[tau2,j2] += eps
                    un_un = U_nom.copy(); un_un[tau,j]  -= eps; un_un[tau2,j2] -= eps
                    block = C @ (simulate(z0, up_up, N_mats)[-1]
                                 - simulate(z0, up_un, N_mats)[-1]
                                 - simulate(z0, un_up, N_mats)[-1]
                                 + simulate(z0, un_un, N_mats)[-1]) / (4*eps**2)
                    H[(tau,j,tau2,j2)] = block
                    H[(tau2,j2,tau,j)] = block
    return H

# === BUILD CORRECTED Co USING HESSIAN ===
def build_corrected_jacobian(Co, H, dU, N_p):
    Co_corrected = Co.copy()
    for (tau, j, tau2, j2), hblock in H.items():
        if (tau, j) <= (tau2, j2):
            Co_corrected[:, tau*m+j]   += 0.5 * hblock * dU[tau2, j2]
            Co_corrected[:, tau2*m+j2] += 0.5 * hblock * dU[tau,  j]
    return Co_corrected

# === SECOND-ORDER SOLVE: one first-order warm start, one corrected Gramian solve ===
def solve_second_order(Co, H, dy, N_p):

    # Warm start: first-order Gramian solve to get initial dU estimate:
    dU_0 = (Co.T @ np.linalg.lstsq(Co @ Co.T, dy, rcond=None)[0]).reshape(N_p, m)

    # Fold Hessian into Co using dU_0, then solve once with the corrected Gramian:
    Co_corr = build_corrected_jacobian(Co, H, dU_0, N_p)
    W_corr  = Co_corr @ Co_corr.T
    dU      = (Co_corr.T @ np.linalg.lstsq(W_corr, dy, rcond=None)[0]).reshape(N_p, m)

    return np.clip(dU.reshape(-1), -1.0, 1.0).reshape(N_p, m)

# === RH STEP SOLVERS ===
def ocg_second_order(z0, U_nom, N_mats):
    N_p    = len(U_nom)
    Co, dy = build_jacobian(z0, U_nom, N_mats)
    H      = build_hessian_sparse(z0, U_nom, N_mats)
    dU     = solve_second_order(Co, H, dy, N_p)
    return U_nom + dU

# === IPOPT ===
def build_Co(z0, N_mats):
    A_pows = [np.eye(n)]
    for _ in range(N):
        A_pows.append(A_pows[-1] @ A)
    C0_aug = np.zeros((n, N * m))
    for k in range(N):
        C0_aug[:, k*m:(k+1)*m] = A_pows[N-1-k] @ B
        for i in range(m):
            C0_aug[:, k*m+i] += A_pows[N-1-k] @ N_mats[i] @ A_pows[k] @ z0
    return C0_aug

def gramian_solve(Co, dz):
    lam = np.linalg.lstsq(Co @ Co.T, dz, rcond=None)[0]
    return (Co.T @ lam).reshape(-1, m)

def ipopt_solve(z0, N_rem, U_init, N_mats):
    opti  = ca.Opti()
    U_var = opti.variable(N_rem * m)
    z     = ca.DM(z0)
    N_dm  = [ca.DM(Ni) for Ni in N_mats]
    for k in range(N_rem):
        u_k = U_var[k*m:(k+1)*m]
        z   = ca.DM(A) @ z + sum(N_dm[i]*u_k[i] for i in range(m)) @ z + ca.DM(B) @ u_k
    opti.minimize(ca.dot(U_var, U_var))
    opti.subject_to(z == ca.DM(z_f))
    opti.set_initial(U_var, U_init.flatten())
    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0,
                          'ipopt.tol': 1e-10, 'ipopt.constr_viol_tol': 1e-10})
    try:
        return np.array(opti.solve().value(U_var)).reshape(N_rem, m)
    except Exception:
        return U_init

# === RH LOOP ===
def run_rh(solver_fn, N_mats):
    z      = z_0.copy()
    U_plan = np.zeros((N, m))
    U_hist = []
    for k in range(N):
        N_rem  = N - k
        U_plan = solver_fn(z, U_plan[:N_rem], N_mats)
        u_k    = U_plan[0]
        U_plan = U_plan[1:]
        z      = bilinear_step(z, u_k, N_mats)
        U_hist.append(u_k.copy())
    U_rh = np.array(U_hist)
    return simulate(z_0, U_rh, N_mats), U_rh

# === MAIN LOOP ===
results = {}

for ratio in ratios:
    res    = {}
    N_mats = make_N_mats(ratio)

    # IPOPT:
    t0    = time.perf_counter()
    Co_an = build_Co(z_0, N_mats)
    U_an0 = gramian_solve(Co_an, z_f - simulate(z_0, np.zeros((N, m)), N_mats)[-1])
    U_an  = ipopt_solve(z_0, N, U_an0, N_mats)
    t_an  = time.perf_counter() - t0
    z_an  = simulate(z_0, U_an, N_mats)
    res['ipopt'] = dict(U=U_an, z=z_an, err=np.linalg.norm(z_an[-1]-z_f), t=t_an)

    # Second-order RH-CG:
    t0       = time.perf_counter()
    z2, U_r2 = run_rh(ocg_second_order, N_mats)
    t2       = time.perf_counter() - t0
    res['rh_2nd'] = dict(U=U_r2, z=z2, err=np.linalg.norm(z2[-1]-z_f), t=t2)

    results[ratio] = res

# === PLOTTING ===
nr     = len(ratios)
steps  = np.arange(N + 1)
usteps = np.arange(N)
colors = {'ipopt': 'steelblue', 'rh_2nd': 'seagreen'}
labels = {'ipopt': 'IPOPT', 'rh_2nd': 'RH-CG (2nd order)'}

# --- Summary plots ---
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))

ax = axes1[0]
err_floor = 1e-15
for key, col in colors.items():
    errs = [max(results[r][key]['err'], err_floor) for r in ratios]
    ax.semilogy(ratios, errs, '-o', color=col, lw=2, label=labels[key])
ax.axhline(err_floor, color='gray', ls=':', lw=1.0, alpha=0.5, label='Machine precision')
ax.set_xlabel(r'$\|N\|/\|A\|$'); ax.set_ylabel(r'Terminal error $\|z_N - z_f\|$')
ax.set_title('Terminal Error')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes1[1]
for key, col in colors.items():
    ax.plot(ratios, [results[r][key]['t']*1e3 for r in ratios],
            '-o', color=col, lw=2, label=labels[key])
ax.set_xlabel(r'$\|N\|/\|A\|$'); ax.set_ylabel('Computation Time (ms)')
ax.set_title('Computation Time')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes1[2]
for key, col in colors.items():
    ax.plot(ratios, [np.sum(results[r][key]['U']**2) for r in ratios],
            '-o', color=col, lw=2, label=labels[key])
ax.set_xlabel(r'$\|N\|/\|A\|$'); ax.set_ylabel(r'Control Cost $\|U\|^2$')
ax.set_title('Control Cost')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

fig1.tight_layout()

# --- State & control trajectories ---
fig2, axes2 = plt.subplots(n + m, nr, figsize=(4*nr, 10), sharex=True, sharey='row')
fig2.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.06, wspace=0.15, hspace=0.12)
for col, ratio in enumerate(ratios):
    axes2[0, col].set_title(fr'$\|N\|/\|A\|={ratio}$', fontsize=10, fontweight='bold')
for i in range(n): axes2[i, 0].set_ylabel(f'$z_{i+1}$')
for i in range(m): axes2[n+i, 0].set_ylabel(f'$u_{i+1}$')
for col, ratio in enumerate(ratios):
    res = results[ratio]
    for i in range(n):
        ax = axes2[i, col]
        for key, col_c in colors.items():
            ax.plot(steps, res[key]['z'][:, i], color=col_c, lw=1.8,
                    ls='--' if key == 'ipopt' else '-', label=labels[key])
        ax.axhline(z_f[i], color='k', ls=':', lw=1.2)
        ax.grid(True, alpha=0.25)
    for i in range(m):
        ax = axes2[n+i, col]
        for key, col_c in colors.items():
            ax.step(usteps, res[key]['U'][:, i], color=col_c, lw=1.8,
                    ls='--' if key == 'ipopt' else '-', label=labels[key])
        ax.grid(True, alpha=0.25)
    axes2[-1, col].set_xlabel('Time step')
axes2[0, 0].legend(fontsize=7, loc='upper right')
fig2.tight_layout()

# --- Eigenvalue cloud ---
theta = np.linspace(0, 2*np.pi, 300)
cmap  = plt.get_cmap('coolwarm')
fig3, axes3 = plt.subplots(2, nr, figsize=(4*nr, 7), sharex=True, sharey=True)
fig3.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12,
                     wspace=0.25, hspace=0.15)
for col, ratio in enumerate(ratios):
    N_mats_plt = make_N_mats(ratio)
    for row, key in enumerate(colors.keys()):
        ax  = axes3[row, col]
        U_k = results[ratio][key]['U']
        all_eigs = np.concatenate([
            np.linalg.eigvals(A + sum(N_mats_plt[i] * U_k[k, i] for i in range(m)))
            for k in range(len(U_k))
        ])
        lim = max(1.3, np.max(np.abs(all_eigs)) * 1.15)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1.0, alpha=0.5, zorder=1)
        ax.axhline(0, color='k', lw=0.3, alpha=0.2)
        ax.axvline(0, color='k', lw=0.3, alpha=0.2)
        for k in range(len(U_k)):
            A_tilde = A + sum(N_mats_plt[i] * U_k[k, i] for i in range(m))
            eigs    = np.linalg.eigvals(A_tilde)
            c       = cmap(k / max(len(U_k) - 1, 1))
            ax.scatter(eigs.real, eigs.imag, s=25, color=c,
                       edgecolors='k', linewidths=0.3, zorder=3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2); ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel(f'{labels[key]}\n\nImag', fontsize=9, fontweight='bold')
        if row == 0:
            ax.set_title(fr'$\|N\|/\|A\|={ratio}$', fontsize=10, fontweight='bold')
        if row == len(colors) - 1:
            ax.set_xlabel('Real', fontsize=8)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=N-1))
sm.set_array([])
cb = fig3.colorbar(sm, ax=axes3.ravel().tolist(),
                   orientation='horizontal', fraction=0.02, pad=0.15, aspect=60)
cb.set_label('Time step $k$', fontsize=9)

plt.show()
