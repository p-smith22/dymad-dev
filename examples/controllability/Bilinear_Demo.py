from funcs import *

# Define shapes:
n, m = 4, 2
N = 20

# Define system matrices:
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
N1 = 0.3 * np.array([
    [ 1.0,  0.0,  0.5,  0.0],
    [ 0.0,  1.0,  0.0,  0.5],
    [ 0.5,  0.0, -1.0,  0.0],
    [ 0.0,  0.5,  0.0, -1.0],
])
N2 = 0.3 * np.array([
    [-1.0,  0.5,  0.0,  0.0],
    [ 0.5, -1.0,  0.5,  0.0],
    [ 0.0,  0.5, -1.0,  0.5],
    [ 0.0,  0.0,  0.5, -1.0],
])
N_mats = [N1, N2]

# Define initial condition and objective:
z_0 = np.array([0.0,  0.0,  0.0,  0.0])
z_f = np.array([0.5,  0.3, -0.2,  0.1])

# Solve for optimal controls:
n_iter = 15
U_bil, U_lin, z_traj_bil, z_traj_lin, resids, t_bil, t_lin = \
    bilinear_optimal_ctrl(A, B, N_mats, N, z_0, z_f,
                          n_iter=n_iter, verbose=False)

# Compute errors:
err_bil = np.linalg.norm(z_traj_bil[-1] - z_f)
err_lin = np.linalg.norm(z_traj_lin[-1] - z_f)

print(f"\n--- Results ---")
print(f"  Bilinear terminal error: {err_bil:.6f}")
print(f"  Zeroth-order terminal error: {err_lin:.6f}")

# =========================================================
# FIGURE 1: Trajectories
# =========================================================
steps  = np.arange(N + 1)
labels = ['Bilinear OCG', 'Zeroth-order']
colors = ['steelblue', 'darkorange']

fig1, axes1 = plt.subplots(n + m, 1, figsize=(10, 10), sharex=True)
for i in range(n):
    for traj, lab, col in zip([z_traj_bil, z_traj_lin], labels, colors):
        axes1[i].plot(steps, traj[:, i], label=lab, color=col, linewidth=2)
    axes1[i].axhline(z_f[i], color='r', linestyle='--', linewidth=1.5,
                     label='Target' if i == 0 else '')
    axes1[i].set_ylabel(f'$z_{i+1}$')
    axes1[i].grid(True)
    if i == 0:
        axes1[i].legend(fontsize=9)

for i in range(m):
    for Uplot, lab, col in zip([U_bil, U_lin], labels, colors):
        axes1[n + i].step(np.arange(N), Uplot[:, i],
                          label=lab, color=col, linewidth=2)
    axes1[n + i].set_ylabel(f'$u_{i+1}$')
    axes1[n + i].grid(True)

axes1[-1].set_xlabel('Time step')
axes1[-1].set_xlim([0, N])
fig1.suptitle('Bilinear OCG (Newton) vs Zeroth-Order', fontsize=12)
fig1.tight_layout()

# =========================================================
# FIGURE 2: Convergence, per-state error, timing
# =========================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))

axes2[0].semilogy(range(1, len(resids) + 1), resids, 'o-',
                  color='steelblue', linewidth=2, markersize=6)
axes2[0].axhline(err_bil, color='r', linestyle='--', linewidth=1.5,
                 label=f'True error = {err_bil:.4f}')
axes2[0].set_xlabel('Iteration')
axes2[0].set_ylabel('Residual')
axes2[0].set_title('Newton Convergence\n(residual vs true error floor)')
axes2[0].legend(fontsize=8)
axes2[0].grid(True, which='both', alpha=0.4)
axes2[0].set_xticks(range(1, len(resids) + 1))

errs_bil = np.abs(z_traj_bil[-1] - z_f)
errs_lin = np.abs(z_traj_lin[-1] - z_f)
x     = np.arange(n)
width = 0.35
axes2[1].bar(x - width/2, errs_lin, width, label='Zeroth-order',
             color='darkorange', edgecolor='k', linewidth=0.8)
axes2[1].bar(x + width/2, errs_bil, width, label='Bilinear OCG',
             color='steelblue', edgecolor='k', linewidth=0.8)
axes2[1].set_xticks(x)
axes2[1].set_xticklabels([f'$z_{i+1}$' for i in range(n)])
axes2[1].set_ylabel('$|z_N^{(i)} - z_f^{(i)}|$')
axes2[1].set_title('Per-State Terminal Error')
axes2[1].legend(fontsize=9)
axes2[1].grid(True, axis='y', alpha=0.4)

time_labels = ['Zeroth-order', 'Bilinear\n(Newton)']
times_ms    = [t_lin * 1000, t_bil * 1000]
bar_colors2 = ['darkorange', 'steelblue']
bars2 = axes2[2].bar(time_labels, times_ms, color=bar_colors2,
                     edgecolor='k', linewidth=0.8, width=0.4)
axes2[2].set_ylabel('Solve time (ms)')
axes2[2].set_title('Computational Cost')
axes2[2].grid(True, axis='y', alpha=0.4)
for bar, t in zip(bars2, times_ms):
    axes2[2].text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + max(times_ms) * 0.01,
                  f'{t:.2f} ms', ha='center', va='bottom', fontsize=9)

fig2.suptitle('Bilinear OCG: Newton Convergence and Cost', fontsize=13)
fig2.tight_layout()

plt.show()
