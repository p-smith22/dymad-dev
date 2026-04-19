# Import packages:
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import time

# === HELPER FUNCTIONS ===
# Build matrices:
def build_matrices(alpha):

    # Pack N matrices:
    N_mats = [alpha * N1, alpha * N2]

    # Precompute powers of A:
    A_pows = [np.eye(n)]
    for _ in range(N):
        A_pows.append(A_pows[-1] @ A)

    # Build C0_aug, which is a combination of the zero order term + the first bilinear correction:
    C0_aug = np.zeros((n, N * m))
    for k in range(N):
        C0_aug[:, k*m:(k+1)*m] = A_pows[N-1-k] @ B
        for i in range(m):
            C0_aug[:, k*m+i] += A_pows[N-1-k] @ N_mats[i] @ A_pows[k] @ z_0

    # Prepare the bilinear  columns for the monomial control term:
    # Here, j and i are the time step and control index of the first control, and k and l are the corresponding
    # values for the second control value:
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

# Calculate the residual and Jaobian for our newton's step:
def newton_step(U, C0_aug, bil_cols, delta_z):

    # Residual given current control:
    r = C0_aug @ U - delta_z
    J = C0_aug.copy()

    # Loop through each of the bilinear columns:
    for (j, k, i, l), col in bil_cols.items():

        # Add monomial term to residual:
        r += col * U[j*m+i] * U[k*m+l]

        # Jacobian w.r.t. control vector:
        J[:, j*m+i] += col * U[k*m+l]

        # Jacobian w.r.t. control scalar:
        J[:, k*m+l] += col * U[j*m+i]

    # Return residual and Jacobian:
    return r, J

# Zeroth (linear) solve:
def solve_zeroth(C0_aug, delta_z, N_mats):

    # Initialize timer:
    t0 = time.perf_counter()

    # Build the augmented gramian w/o the bilinear correction, then solve for controls sequence:
    W0 = C0_aug @ C0_aug.T
    U  = (C0_aug.T @ np.linalg.solve(W0, delta_z)).reshape(N, m)

    # Calculate computation time:
    t  = time.perf_counter() - t0

    # Return controls, states, and time to solve:
    return U, simulate(U, N_mats), t

# Newton's step to apply bilinear correction:
def solve_newton(C0_aug, bil_cols, delta_z, N_mats):

    # Start timer:
    t0 = time.perf_counter()

    # Build the augmented gramian w/o the bilinear correction, then solve for controls sequence:
    W0 = C0_aug @ C0_aug.T
    U  = C0_aug.T @ np.linalg.solve(W0, delta_z)

    # Apply the corrections:
    for _ in range(n_iter):

        # Fetch the residual and Jacobian for our given controls:
        r, J = newton_step(U, C0_aug, bil_cols, delta_z)

        # Break if residual is small enough:
        if np.linalg.norm(r) < tol:
            break

        # If not, take the newton's step to push the residual to zero:
        U = U - J.T @ np.linalg.solve(J @ J.T, r)

    # Calculate computation time:
    t = time.perf_counter() - t0

    # Return controls, states, and time to solve:
    return U.reshape(N, m), simulate(U.reshape(N, m), N_mats), t

# IPOPT to apply bilinear correction:
def solve_ipopt(C0_aug, bil_cols, delta_z, N_mats):

    # Start timer:
    t0    = time.perf_counter()

    # Set optimization variables and objective:
    opti  = ca.Opti()
    U_var = opti.variable(N * m)
    opti.minimize(ca.dot(U_var, U_var))

    # Set dynamics constraint that r must = 0:
    r = ca.DM(C0_aug) @ U_var - ca.DM(delta_z)
    for (j, k, i, l), col in bil_cols.items():
        r = r + ca.DM(col) * U_var[j*m+i] * U_var[k*m+l]
    opti.subject_to(r == 0)

    # Warm start from zeroth-order solve:
    W0     = C0_aug @ C0_aug.T
    U_init = C0_aug.T @ np.linalg.solve(W0, delta_z)
    opti.set_initial(U_var, U_init)
    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0,
                          'ipopt.tol': 1e-10, 'ipopt.constr_viol_tol': 1e-10})

    # Solve optimization problem and package solution:
    sol    = opti.solve()
    U_flat = np.array(sol.value(U_var)).reshape(N, m)

    # Calculate computation time:
    t      = time.perf_counter() - t0

    # Return controls, states, and time to solve:
    return U_flat, simulate(U_flat, N_mats), t

# Simulate the dynamics:
def simulate(U, N_mats):

    # Initialize array:
    z = np.zeros((N + 1, n))

    # Set initial condition:
    z[0] = z_0

    # Loop through each step:
    for k in range(N):

        # Build N(u_k) terms:
        N_u    = sum(N_mats[i] * U[k, i] for i in range(m))

        # Propagate:
        z[k+1] = A @ z[k] + N_u @ z[k] + B @ U[k]

    # Return states:
    return z

# Solve all the solutions:
def solve_all(alpha):

    # Build the required matrices:
    C0_aug, bil_cols, delta_z, N_mats = build_matrices(alpha)

    # Fetch the controls, states, and computation time for each method:
    U_lin,   z_lin,   t_lin   = solve_zeroth(C0_aug, delta_z, N_mats)
    U_newt,  z_newt,  t_newt  = solve_newton(C0_aug, bil_cols, delta_z, N_mats)
    U_ipopt, z_ipopt, t_ipopt = solve_ipopt(C0_aug, bil_cols, delta_z, N_mats)

    # Find the eigenvalues of A_tilde at each step:
    def eigs_from_U(U):
        return [np.linalg.eigvals(A + sum(N_mats[i]*U[k,i] for i in range(m))) for k in range(N)]

    # Package the results and return:
    return {
        'zeroth': (U_lin,   z_lin,   t_lin,   eigs_from_U(U_lin)),
        'newton': (U_newt,  z_newt,  t_newt,  eigs_from_U(U_newt)),
        'ipopt' : (U_ipopt, z_ipopt, t_ipopt, eigs_from_U(U_ipopt)),
    }

# Settings:
alpha_work = 0.1
alpha_fail = 0.8
n_iter     = 20
tol        = 1e-10

# System definition:
n, m, N = 4, 2, 20

# Define matrices:
A = np.array([
    [0.80,  0.10,  0.05,  0.00],
    [0.00,  0.75,  0.10,  0.05],
    [0.05,  0.00,  0.70,  0.10],
    [0.00,  0.05,  0.00,  0.65],
])
B = np.array([
    [1.0,  0.2],
    [0.5,  1.0],
    [0.2,  0.5],
    [0.1,  0.3],
])
N1 = np.array([
    [ 1.0,  0.0,  0.5,  0.0],
    [ 0.0,  1.0,  0.0,  0.5],
    [ 0.5,  0.0, -1.0,  0.0],
    [ 0.0,  0.5,  0.0, -1.0],
])
N2 = np.array([
    [-1.0,  0.5,  0.0,  0.0],
    [ 0.5, -1.0,  0.5,  0.0],
    [ 0.0,  0.5, -1.0,  0.5],
    [ 0.0,  0.0,  0.5, -1.0],
])

# Define initial and final states:
z_0 = np.array([0.0,  0.0,  0.0,  0.0])
z_f = np.array([0.5,  0.3, -0.2,  0.1])

# === MAIN LOOP ===
# Solve the working cases:
print(f"Solving alpha = {alpha_work} (working)...")
res_w = solve_all(alpha_work)

# Solve the failing cases:
print(f"Solving alpha = {alpha_fail} (failing)...")
res_f = solve_all(alpha_fail)

# Loop through each of the working cases and print results:
for name, (U, z, t, _) in res_w.items():
    err = np.linalg.norm(z[-1] - z_f)
    print(f"  [{alpha_work}] {name:<8}  err={err:.4f}  t={t*1e3:.1f}ms")

# Loop through each of the failing cases and print results:
for name, (U, z, t, _) in res_f.items():
    err = np.linalg.norm(z[-1] - z_f)
    print(f"  [{alpha_fail}] {name:<8}  err={err:.4f}  t={t*1e3:.1f}ms")

# === PLOTTING ===
# Styling for plotting:
solver_labels = ['Zeroth-order', 'Newton', 'IPOPT']
solver_keys   = ['zeroth', 'newton', 'ipopt']
colors        = ['darkorange', 'steelblue', 'seagreen']
steps         = np.arange(N + 1)
usteps        = np.arange(N)
theta         = np.linspace(0, 2*np.pi, 300)
cmap          = plt.get_cmap('coolwarm')

# -------------------------------------------------------------------------
# Figure 1: eigenvalue clouds (working vs failing, one solver per row)
# -------------------------------------------------------------------------
all_eigs = [e for res in [res_w, res_f] for key in solver_keys
            for e in res[key][3]]
global_lim = max(1.4, np.max(np.abs(np.concatenate(all_eigs))) * 1.15)

fig1, axes1 = plt.subplots(2, 3, figsize=(14, 10), sharex=True, sharey=True)
fig1.subplots_adjust(left=0.12, bottom=0.12, right=0.98, top=0.95,
                     wspace=0.05, hspace=0.15)

for col, (key, lab) in enumerate(zip(solver_keys, solver_labels)):
    axes1[0, col].set_title(lab, fontsize=11, fontweight='bold')

for row, (res, alpha, case) in enumerate([
        (res_w, alpha_work, 'Working'),
        (res_f, alpha_fail, 'Failing')]):

    axes1[row, 0].set_ylabel(f'{case}  ($\\alpha={alpha}$)\n\nImag', fontsize=9, fontweight='bold')

    for col, (key, lab) in enumerate(zip(solver_keys, solver_labels)):

        ax       = axes1[row, col]
        _, z, _, eigs_all = res[key]
        term_err = np.linalg.norm(z[-1] - z_f)

        ax.plot(np.cos(theta), np.sin(theta),
                'k--', linewidth=1.4, alpha=0.6, zorder=1)
        ax.axhline(0, color='k', linewidth=0.4, alpha=0.25)
        ax.axvline(0, color='k', linewidth=0.4, alpha=0.25)

        for k, eigs_k in enumerate(eigs_all):
            c = cmap(k / (N - 1))
            ax.scatter(eigs_k.real, eigs_k.imag,
                       s=45, color=c, edgecolors='k',
                       linewidths=0.4, zorder=3)

        if row == 1:
            ax.set_xlabel('Real part', fontsize=9, fontweight='bold')

        ax.text(0.02, 0.97, f'err = {term_err:.4f}',
                transform=ax.transAxes, fontsize=8,
                va='top', ha='left')

        ax.set_aspect('equal')
        ax.set_xlim([-global_lim, global_lim])
        ax.set_ylim([-global_lim, global_lim])
        ax.grid(True, alpha=0.2)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=N-1))
sm.set_array([])
cb = fig1.colorbar(sm, ax=axes1.ravel().tolist(), orientation='horizontal',
                   fraction=0.02, pad=0.08, aspect=50)
cb.set_label('Time step $k$', fontsize=9)

# -------------------------------------------------------------------------
# Figure 2: trajectories for both alphas, all three solvers
# -------------------------------------------------------------------------
fig2, axes2 = plt.subplots(n + m, 2, figsize=(13, 11), sharex='col')

for col, (res, alpha, case) in enumerate([
        (res_w, alpha_work, 'Working'),
        (res_f, alpha_fail, 'Failing')]):

    for i in range(n):
        ax = axes2[i, col]
        for key, lab, col_c in zip(solver_keys, solver_labels, colors):
            _, z, _, _ = res[key]
            err = np.linalg.norm(z[-1] - z_f)
            ax.plot(steps, z[:, i], color=col_c, linewidth=2,
                    label=f'{lab} (err={err:.3f})',
                    linestyle='-' if key != 'zeroth' else '--')
        ax.axhline(z_f[i], color='r', linestyle='--', linewidth=1.5,
                   label='Target' if i == 0 else '')
        ax.set_ylabel(f'$z_{i+1}$')
        ax.grid(True)
        if i == 0:
            ax.set_title(f'{case}:  $\\alpha = {alpha}$', fontsize=11)
            ax.legend(fontsize=7)

    for i in range(m):
        ax = axes2[n + i, col]
        for key, lab, col_c in zip(solver_keys, solver_labels, colors):
            U, _, _, _ = res[key]
            ax.step(usteps, U[:, i], color=col_c, linewidth=2,
                    label=lab,
                    linestyle='-' if key != 'zeroth' else '--')
        ax.set_ylabel(f'$u_{i+1}$')
        ax.grid(True)

    axes2[-1, col].set_xlabel('Time step')
    axes2[-1, col].set_xlim([0, N])

axes2[0, 0].legend(fontsize=7)

fig2.tight_layout()

# -------------------------------------------------------------------------
# Figure 3: accuracy and timing comparison
# -------------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))

x      = np.arange(len(solver_keys))
width  = 0.35

for ax_idx, (res, alpha, case) in enumerate([
        (res_w, alpha_work, 'Working'),
        (res_f, alpha_fail, 'Failing')]):

    errs  = [np.linalg.norm(res[k][1][-1] - z_f) for k in solver_keys]
    times = [res[k][2] * 1e3 for k in solver_keys]

    # Per-state terminal error bars:
    if ax_idx == 0:
        ax = axes3[0]
        ax.bar(x - width/2, [np.linalg.norm(res_w[k][1][-1] - z_f)
                              for k in solver_keys],
               width, label=f'$\\alpha={alpha_work}$',
               color=colors, edgecolor='k', linewidth=0.8, alpha=0.9)
        ax.bar(x + width/2, [np.linalg.norm(res_f[k][1][-1] - z_f)
                              for k in solver_keys],
               width, label=f'$\\alpha={alpha_fail}$',
               color=colors, edgecolor='k', linewidth=0.8, alpha=0.4,
               hatch='//')
        ax.set_xticks(x)
        ax.set_xticklabels(solver_labels)
        ax.set_ylabel('Terminal error $\\|z_N - z_f\\|$')
        ax.set_title('Terminal Error')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

# Timing comparison (both alphas, grouped):
ax = axes3[1]
times_w = [res_w[k][2] * 1e3 for k in solver_keys]
times_f = [res_f[k][2] * 1e3 for k in solver_keys]
bars_w = ax.bar(x - width/2, times_w, width,
                label=f'$\\alpha={alpha_work}$',
                color=colors, edgecolor='k', linewidth=0.8, alpha=0.9)
bars_f = ax.bar(x + width/2, times_f, width,
                label=f'$\\alpha={alpha_fail}$',
                color=colors, edgecolor='k', linewidth=0.8, alpha=0.4,
                hatch='//')
for bar, t in zip(list(bars_w) + list(bars_f), times_w + times_f):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(times_w + times_f) * 0.01,
            f'{t:.1f}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(solver_labels)
ax.set_ylabel('Solve time (ms)')
ax.set_title('Computational Cost')
ax.legend(fontsize=8)
ax.grid(True, axis='y', alpha=0.3)

# ||U|| norm comparison:
ax = axes3[2]
norms_w = [np.linalg.norm(res_w[k][0]) for k in solver_keys]
norms_f = [np.linalg.norm(res_f[k][0]) for k in solver_keys]
ax.bar(x - width/2, norms_w, width,
       label=f'$\\alpha={alpha_work}$',
       color=colors, edgecolor='k', linewidth=0.8, alpha=0.9)
ax.bar(x + width/2, norms_f, width,
       label=f'$\\alpha={alpha_fail}$',
       color=colors, edgecolor='k', linewidth=0.8, alpha=0.4,
       hatch='//')
ax.set_xticks(x)
ax.set_xticklabels(solver_labels)
ax.set_ylabel('$\\|U\\|$')
ax.set_title('Control Effort')
ax.legend(fontsize=8)
ax.grid(True, axis='y', alpha=0.3)

fig3.tight_layout()

plt.show()
