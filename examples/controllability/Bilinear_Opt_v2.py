# Import packages:
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import time

# === HELPER FUNCTIONS ===
# Build matrices:
def build_matrices(ratio):

    # Scale N matrices so ||N||/||A|| = ratio exactly:
    A_norm = np.linalg.norm(A)
    N_mats = [ratio * A_norm / np.linalg.norm(N1_base) * N1_base,
              ratio * A_norm / np.linalg.norm(N2_base) * N2_base]

    # Precompute powers of A:
    A_pows = [np.eye(n)]
    for _ in range(N):
        A_pows.append(A_pows[-1] @ A)

    # Build C0_aug:
    C0_aug = np.zeros((n, N * m))
    for k in range(N):
        C0_aug[:, k*m:(k+1)*m] = A_pows[N-1-k] @ B
        for i in range(m):
            C0_aug[:, k*m+i] += A_pows[N-1-k] @ N_mats[i] @ A_pows[k] @ z_0

    # Prepare bilinear columns:
    bil_cols = {}
    for k in range(N):
        for j in range(k+1, N):
            for i in range(m):
                block = A_pows[N-1-j] @ N_mats[i] @ A_pows[j-1-k] @ B
                for l in range(m):
                    bil_cols[(j, k, i, l)] = block[:, l]

    # Define observation gap:
    delta_z = z_f - A_pows[N] @ z_0

    # Return values:
    return C0_aug, bil_cols, delta_z, N_mats

# Calculate the residual and Jacobian for newton's step:
def newton_step(U, C0_aug, bil_cols, delta_z):

    # Residual given current control:
    r = C0_aug @ U - delta_z
    J = C0_aug.copy()

    # Loop through each of the bilinear columns:
    for (j, k, i, l), col in bil_cols.items():
        r           += col * U[j*m+i] * U[k*m+l]
        J[:, j*m+i] += col * U[k*m+l]
        J[:, k*m+l] += col * U[j*m+i]

    # Return residual and Jacobian:
    return r, J

# Zeroth (linear) solve:
def solve_zeroth(C0_aug, delta_z, N_mats):
    t0 = time.perf_counter()
    W0 = C0_aug @ C0_aug.T
    U  = (C0_aug.T @ np.linalg.solve(W0, delta_z)).reshape(N, m)
    t  = time.perf_counter() - t0
    return U, simulate(U, N_mats), t

# Newton's step to apply bilinear correction:
def solve_newton(C0_aug, bil_cols, delta_z, N_mats):
    t0 = time.perf_counter()
    W0 = C0_aug @ C0_aug.T
    U  = C0_aug.T @ np.linalg.solve(W0, delta_z)
    for _ in range(n_iter):
        r, J = newton_step(U, C0_aug, bil_cols, delta_z)
        if np.linalg.norm(r) < tol:
            break
        U = U - J.T @ np.linalg.solve(J @ J.T, r)
    t = time.perf_counter() - t0
    return U.reshape(N, m), simulate(U.reshape(N, m), N_mats), t

# IPOPT to apply bilinear correction:
def solve_ipopt(C0_aug, bil_cols, delta_z, N_mats):
    t0    = time.perf_counter()
    opti  = ca.Opti()
    U_var = opti.variable(N * m)
    opti.minimize(ca.dot(U_var, U_var))
    r = ca.DM(C0_aug) @ U_var - ca.DM(delta_z)
    for (j, k, i, l), col in bil_cols.items():
        r = r + ca.DM(col) * U_var[j*m+i] * U_var[k*m+l]
    opti.subject_to(r == 0)
    W0     = C0_aug @ C0_aug.T
    U_init = C0_aug.T @ np.linalg.solve(W0, delta_z)
    opti.set_initial(U_var, U_init)
    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0,
                          'ipopt.tol': 1e-10, 'ipopt.constr_viol_tol': 1e-10})
    sol    = opti.solve()
    U_flat = np.array(sol.value(U_var)).reshape(N, m)
    t      = time.perf_counter() - t0
    return U_flat, simulate(U_flat, N_mats), t

# Simulate the dynamics:
def simulate(U, N_mats):
    z    = np.zeros((N + 1, n))
    z[0] = z_0
    for k in range(N):
        N_u    = sum(N_mats[i] * U[k, i] for i in range(m))
        z[k+1] = A @ z[k] + N_u @ z[k] + B @ U[k]
    return z

# Solve all methods for a given ratio:
def solve_all(ratio):
    C0_aug, bil_cols, delta_z, N_mats = build_matrices(ratio)
    U_lin,   z_lin,   t_lin   = solve_zeroth(C0_aug, delta_z, N_mats)
    U_newt,  z_newt,  t_newt  = solve_newton(C0_aug, bil_cols, delta_z, N_mats)
    U_ipopt, z_ipopt, t_ipopt = solve_ipopt(C0_aug, bil_cols, delta_z, N_mats)

    # Eigenvalues of A_tilde at each step:
    def eigs_from_U(U):
        return [np.linalg.eigvals(A + sum(N_mats[i]*U[k,i] for i in range(m)))
                for k in range(N)]

    # Spectral radius of A_tilde at each step — stability check:
    def spec_rad(U):
        return [np.max(np.abs(np.linalg.eigvals(A + sum(N_mats[i]*U[k,i] for i in range(m)))))
                for k in range(N)]

    return {
        'zeroth': (U_lin,   z_lin,   t_lin,   eigs_from_U(U_lin),   spec_rad(U_lin)),
        'newton': (U_newt,  z_newt,  t_newt,  eigs_from_U(U_newt),  spec_rad(U_newt)),
        'ipopt' : (U_ipopt, z_ipopt, t_ipopt, eigs_from_U(U_ipopt), spec_rad(U_ipopt)),
    }

# === SYSTEM DEFINITION ===
n_iter = 20
tol    = 1e-10
n, m, N = 4, 2, 20

# A is diagonal:
A = np.diag([0.90, 0.85, 0.80, 0.75])

# B maps controls into all states evenly:
B = np.array([
    [1.0,  0.0],
    [0.0,  1.0],
    [1.0,  0.0],
    [0.0,  1.0],
])

# N bases are strictly skew-symmetric:
N1_base = np.array([
    [ 0.0,  1.0,  0.0,  0.0],
    [-1.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0],
    [ 0.0,  0.0, -1.0,  0.0],
])
N2_base = np.array([
    [ 0.0,  0.0,  1.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0],
    [-1.0,  0.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0,  0.0],
])

# Initial and final states:
z_0 = np.array([0.0,  0.0,  0.0,  0.0])
z_f = np.array([0.5,  0.3, -0.2,  0.1])

# Ratio sweep — directly sets ||N||/||A||:
ratios = [0.01, 0.1, 0.5, 1.0, 2.0]

# === MAIN LOOP ===
results = {}
for ratio in ratios:
    print(f"Solving ||N||/||A|| = {ratio}...")
    results[ratio] = solve_all(ratio)
    for name in ['zeroth', 'newton', 'ipopt']:
        U, z, t, _, sr = results[ratio][name]
        err    = np.linalg.norm(z[-1] - z_f)
        sr_max = np.max(sr)
        print(f"  {name:<8}  err={err:.4f}  t={t*1e3:.1f}ms  max_rho={sr_max:.4f}")

# === PLOTTING ===
solver_labels = ['Zeroth-order', 'Newton', 'IPOPT']
solver_keys   = ['zeroth', 'newton', 'ipopt']
colors        = ['darkorange', 'steelblue', 'seagreen']
steps         = np.arange(N + 1)
usteps        = np.arange(N)
theta         = np.linspace(0, 2*np.pi, 300)
cmap          = plt.get_cmap('coolwarm')
nr            = len(ratios)

# -------------------------------------------------------------------------
# Figure 1: terminal error vs ratio + stability verification
# -------------------------------------------------------------------------
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))

# Left: terminal error vs ratio:
ax = axes1[0]
for key, lab, col in zip(solver_keys, solver_labels, colors):
    errs = [np.linalg.norm(results[r][key][1][-1] - z_f) for r in ratios]
    ax.semilogy(ratios, errs, '-o', color=col, linewidth=2, label=lab)
ax.set_xlabel('$\\|N\\|/\\|A\\|$')
ax.set_ylabel('Terminal error $\\|z_N - z_f\\|$')
ax.set_title('Terminal Error vs $\\|N\\|/\\|A\\|$')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: max spectral radius across all steps and solvers — should stay flat:
ax = axes1[1]
for key, lab, col in zip(solver_keys, solver_labels, colors):
    sr_max = [np.max(results[r][key][4]) for r in ratios]
    ax.plot(ratios, sr_max, '-o', color=col, linewidth=2, label=lab)
ax.axhline(1.0, color='k', linestyle='--', linewidth=1.2, alpha=0.6, label='Unit circle')
ax.set_xlabel('$\\|N\\|/\\|A\\|$')
ax.set_ylabel('Max $\\rho(\\tilde{A})$')
ax.set_title('Max Spectral Radius')
ax.legend()
ax.grid(True, alpha=0.3)

fig1.tight_layout()

# -------------------------------------------------------------------------
# Figure 2: trajectories — cols = ratio, rows = states + controls
# -------------------------------------------------------------------------
fig2, axes2 = plt.subplots(n + m, nr, figsize=(4*nr, 11), sharex=True)
fig2.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.06,
                     wspace=0.15, hspace=0.12)

for col, ratio in enumerate(ratios):
    axes2[0, col].set_title(f'$\\|N\\|/\\|A\\|={ratio}$', fontsize=10, fontweight='bold')

for i in range(n):
    axes2[i, 0].set_ylabel(f'$z_{i+1}$')
for i in range(m):
    axes2[n+i, 0].set_ylabel(f'$u_{i+1}$')

for col, ratio in enumerate(ratios):
    for i in range(n):
        ax = axes2[i, col]
        for key, lab, c in zip(solver_keys, solver_labels, colors):
            _, z, _, _, _ = results[ratio][key]
            ax.plot(steps, z[:, i], color=c, linewidth=1.8,
                    linestyle='--' if key == 'zeroth' else '-', label=lab)
        ax.axhline(z_f[i], color='r', linestyle=':', linewidth=1.2)
        ax.grid(True, alpha=0.25)
    for i in range(m):
        ax = axes2[n+i, col]
        for key, lab, c in zip(solver_keys, solver_labels, colors):
            U, _, _, _, _ = results[ratio][key]
            ax.step(usteps, U[:, i], color=c, linewidth=1.8,
                    linestyle='--' if key == 'zeroth' else '-', label=lab)
        ax.grid(True, alpha=0.25)
    axes2[-1, col].set_xlabel('Time step')

axes2[0, 0].legend(fontsize=7, loc='upper right')
fig2.tight_layout()

# -------------------------------------------------------------------------
# Figure 3: eigenvalue clouds — rows = solver, cols = ratio
# -------------------------------------------------------------------------
all_eigs   = [e for r in ratios for key in solver_keys for e in results[r][key][3]]
global_lim = max(1.4, np.max(np.abs(np.concatenate(all_eigs))) * 1.15)

fig3, axes3 = plt.subplots(3, nr, figsize=(4*nr, 11), sharex=True, sharey=True)
fig3.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.93,
                     wspace=0.05, hspace=0.08)

for col, ratio in enumerate(ratios):
    axes3[0, col].set_title(f'$\\|N\\|/\\|A\\|={ratio}$', fontsize=10, fontweight='bold')

for row, (key, lab) in enumerate(zip(solver_keys, solver_labels)):
    axes3[row, 0].set_ylabel(f'{lab}\n\nImag', fontsize=9, fontweight='bold')
    for col, ratio in enumerate(ratios):
        ax       = axes3[row, col]
        _, z, _, eigs_all, _ = results[ratio][key]
        term_err = np.linalg.norm(z[-1] - z_f)

        ax.plot(np.cos(theta), np.sin(theta),
                'k--', linewidth=1.2, alpha=0.6, zorder=1)
        ax.axhline(0, color='k', linewidth=0.4, alpha=0.25)
        ax.axvline(0, color='k', linewidth=0.4, alpha=0.25)

        for k, eigs_k in enumerate(eigs_all):
            c = cmap(k / (N - 1))
            ax.scatter(eigs_k.real, eigs_k.imag,
                       s=35, color=c, edgecolors='k',
                       linewidths=0.3, zorder=3)

        ax.text(0.02, 0.97, f'err={term_err:.4f}',
                transform=ax.transAxes, fontsize=7, va='top', ha='left')

        if row == 2:
            ax.set_xlabel('Real', fontsize=8)

        ax.set_aspect('equal')
        ax.set_xlim([-global_lim, global_lim])
        ax.set_ylim([-global_lim, global_lim])
        ax.grid(True, alpha=0.2)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=N-1))
sm.set_array([])
cb = fig3.colorbar(sm, ax=axes3.ravel().tolist(), orientation='horizontal',
                   fraction=0.02, pad=0.06, aspect=60)
cb.set_label('Time step $k$', fontsize=9)

plt.show()