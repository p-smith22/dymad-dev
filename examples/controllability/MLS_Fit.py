# === RUN SETTINGS ===
# Import packages:
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import sys

# Hide IPOPT warnings for cleaner runs:
sys.stderr = open(os.devnull, 'w')

# === SYSTEM ===
# Define shapes:
n, m, N  = 4, 2, 20

# Define system matrices:
A = np.diag([0.90, 0.85, 0.80, 0.75])
B = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
N1_base = np.array([[ 0., 1., 0., 0.], [-1., 0., 0., 0.],
                    [ 0., 0., 0., 1.], [ 0., 0.,-1., 0.]])
N2_base = np.array([[ 0., 0., 1., 0.], [ 0., 0., 0., 1.],
                    [-1., 0., 0., 0.], [ 0.,-1., 0., 0.]])

# Definite initial and end point (*FOR TRAINING*):
z_0 = 2 * np.array([-3.0,  2.5, -2.0,  1.2])
z_A = 2 * np.array([ 3.0, -2.5,  2.0, -1.2])
z_des = z_A + 5 * np.array([1.8, -1.5, 1.2, -1.0])

# Define ratios to sweep through:
ratios = [0.01, 0.5, 1.0, 2.0, 5.0]

# MLS Settings:
n_train  = 300   # Number of training IPOPT runs
n_query  = 10    # Number of refinements (>=1)
train_r  = 5.0   # Training points
k_mls    = 50    # Number of nearest neighbors to use
h_mls    = 0.6   # How much to prioritize nearest neighbors

# === DYNAMICS ===
# Construct N matrices given the ratio:
def construct_N(r):

    # Apply ratio to the two N matrices and return:
    s = r * np.linalg.norm(A)
    return [s / np.linalg.norm(N1_base) * N1_base,
            s / np.linalg.norm(N2_base) * N2_base]

# Propagate dynamics:
def sim(z0, U, Nm):

    # Copy initial state and initialize trajectory:
    z = z0.copy()
    traj = [z.copy()]

    # Loop through each control:
    for u in U:

        # Take step and add to trajectory:
        z = A @ z + sum(Nm[i] * u[i] for i in range(m)) @ z + B @ u
        traj.append(z.copy())

    # Return full trajectory:
    return np.array(traj)

# Recover Controllability Matrix based on optimal control:
def recover_Co(z0, U_nom, Nm):

    # Initialize:
    z = z0.copy()
    traj = [z.copy()]
    At = []

    # Loop through each of the controls:
    for u in U_nom:

        # Accumulate A:
        Ak = A + sum(Nm[i] * u[i] for i in range(m))
        At.append(Ak)

        # Propagate and append to trajectory:
        z = Ak @ z + B @ u
        traj.append(z.copy())

    # Initialize Phi:
    Phi = [None] * (N + 1)

    # Last step is just identity (A^0):
    Phi[N] = np.eye(n)

    # Loop back through all the steps and accumulate A:
    for k in range(N - 1, -1, -1):
        Phi[k] = Phi[k + 1] @ At[k]

    # Initialize Controllability Matrix:
    Co = np.zeros((n, N * m))

    # Loop through each of the steps and controls:
    for tau in range(N):
        for j in range(m):

            # Fill the appropriate columns of the Controllability Matrix:
            Co[:, tau*m+j] = Phi[tau + 1] @ (B[:, j] + Nm[j] @ traj[tau])

    # Return Matrix:
    return Co

# Solve Gramian for optimal control:
def gram_solve(Co, dz):

    # Solve for optimal control:
    return (Co.T @ np.linalg.lstsq(Co @ Co.T, dz, rcond=None)[0]).reshape(N, m)

# Solve IPOPT Optimization:
def ipopt_solve(z0, U_init, Nm, z_tgt):

    # Initialize variables:
    opti = ca.Opti()
    U_var = opti.variable(N * m)
    z = ca.DM(z0)
    Nd = [ca.DM(Ni) for Ni in Nm]

    # Loop through each step in the trajectory:
    for k in range(N):

        # Dissect control and propagate:
        u_k = U_var[k*m:(k+1)*m]
        z = ca.DM(A) @ z + sum(Nd[i]*u_k[i] for i in range(m)) @ z + ca.DM(B) @ u_k

    # Set objectives and constraints:
    opti.minimize(ca.dot(U_var, U_var))
    opti.subject_to(z == ca.DM(z_tgt))
    opti.set_initial(U_var, U_init.flatten())

    # Solver settings:
    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0, 'verbose': False,
                          'ipopt.tol': 1e-10, 'ipopt.constr_viol_tol': 1e-10})

    # Solve:
    sol = opti.solve().value(U_var)

    # Return optimized control:
    return np.array(sol).reshape(N, m)

# Query the MLS solver to get the approximated Controllability Matrix:
def query_mls(z_desired, train_zf, train_Co, k=20, h_frac=0.6):

    # Find distances between training and ACTUAL desired:
    dists  = np.linalg.norm(train_zf - z_desired, axis=1)

    # Sort to find nearest neighbors, extract their distances:
    nn_idx = np.argsort(dists)[:k]
    d_loc  = dists[nn_idx]

    # Compute weights (favor nearest neighbors first):
    h      = max(h_frac * d_loc[-1], 1e-8)
    w      = np.exp(-0.5 * (d_loc / h)**2)

    # Make linear system:
    dz     = train_zf[nn_idx] - z_desired
    Phi    = np.hstack([np.ones((k, 1)), dz])

    # Get training Matrices of nearest neighbors:
    Co_loc = np.array([train_Co[i].flatten() for i in nn_idx])
    sqrtw  = np.sqrt(w)

    # Weighted least squares fit:
    coeff, _, _, _ = np.linalg.lstsq(Phi * sqrtw[:, None], Co_loc * sqrtw[:, None], rcond=None)

    # Return MLS Controllability Matrix:
    return coeff[0].reshape(n, N * m)

# Solve MLS fit of training data:
def mls_solve(z_f, train_zf, train_Co, train_U, Nm, n_passes=1):

    # Ensure n_passes is large enough to avoid errors:
    if n_passes < 1:
        n_passes = 1

    # Find nearest neighbor and extract the control there:
    nn = np.argmin(np.linalg.norm(train_zf - z_f, axis=1))
    U_nn = train_U[nn]

    # Simulate dynamics to get final position:
    z_nn_end = sim(z_0, U_nn, Nm)[-1]

    # Solve for initial control:
    Co = query_mls(z_f, train_zf, train_Co, k=k_mls, h_frac=h_mls)
    U = U_nn + gram_solve(Co, z_f - z_nn_end)

    # Solve for observation gap:
    z_end = sim(z_0, U, Nm)[-1]
    dz = z_f - z_end

    # Requery to refine:
    for n in range(n_passes-1):

        # Resolve for gramian and update control:
        Co = query_mls(z_end, train_zf, train_Co, k=k_mls, h_frac=h_mls)
        U += gram_solve(Co, dz)

        # Solve for new observation gap:
        z_end = sim(z_0, U, Nm)[-1]
        dz = z_f - z_end

    # Return optimal control:
    return U

# === SAMPLE CLOUDS FOR TRAINING ===
# Make RNG w/ set seed:
rng = np.random.default_rng(42)

# Randomize the magnitude:
dirs_t = rng.standard_normal((n_train, n))

# Normalize:
dirs_t /= np.linalg.norm(dirs_t, axis=1, keepdims=True)

# Define new training point within the defined range:
z_train = z_A + dirs_t * rng.uniform(0.3, train_r, n_train)[:, None]

# === COMPARE IPOPT AND MLS ===
# Initialize results:
results = {}

# Loop through each ratio:
for ratio in ratios:

    # Initialize results:
    print(f"\n── ratio = {ratio} ──────────────────────────────")
    Nm     = construct_N(ratio)
    z_free = sim(z_0, np.zeros((N, m)), Nm)[-1]

    # Solve nominal trajectory:
    U_nom = ipopt_solve(z_0, np.zeros(shape=(N, m)), Nm, z_A)
    z_nom  = sim(z_0, U_nom,  Nm)

    # --- Offline Training ---
    # Start timer, intialize lists:
    t0 = time.perf_counter()
    train_Co_list, train_U_list = [], []

    # Loop through each of the training objectives:
    for z_t in z_train:

        # Solve IPOPT and append results:
        U_i = ipopt_solve(z_0, np.zeros(shape=(N,m)), Nm, z_t)
        train_Co_list.append(recover_Co(z_0, U_i, Nm))
        train_U_list.append(U_i)

    # End timer, print results:
    t_train = time.perf_counter() - t0
    print(f"  [train]    {n_train} IPOPT solves  t={t_train:.1f}s")

    # --- Online Solves ---
    # MLS Solve:
    t0 = time.perf_counter()
    U_mls = mls_solve(z_des, z_train, train_Co_list, train_U_list, Nm, n_query)
    t_online = time.perf_counter() - t0

    # IPOPT Solve:
    t_gt0 = time.perf_counter()
    U_gt = ipopt_solve(z_0, np.zeros(shape=(N,m)), Nm, z_des)
    t_gt = time.perf_counter() - t_gt0

    # Propagate trajectories under solved controls:
    z_mls = sim(z_0, U_mls, Nm)
    z_gt  = sim(z_0, U_gt,  Nm)

    # Calculate errors/cost:
    mls_err  = np.linalg.norm(z_mls[-1] - z_des)
    gt_err   = np.linalg.norm(z_gt[-1]  - z_des)
    mls_cost = float(np.sum(U_mls**2))
    gt_cost  = float(np.sum(U_gt**2))

    # Print results:
    print(f"  [MLS]      err={mls_err:.2e}  cost={mls_cost:.4f}  t={t_online*1e3:.1f}ms")
    print(f"  [IPOPT]    err={gt_err:.2e}   cost={gt_cost:.4f}   t={t_gt*1e3:.1f}ms  ")

    # Package results into a dictionary:
    results[ratio] = dict(
        mls_err=mls_err,   mls_cost=mls_cost,
        gt_err=gt_err,     gt_cost=gt_cost,
        z_mls=z_mls,       z_gt=z_gt,    z_nom=z_nom,
        U_mls=U_mls,       U_gt=U_gt,    U_nom=U_nom,
        t_train=t_train, t_online=t_online, t_gt=t_gt,
    )

# === PLOTTING ===
# Settings:
steps  = np.arange(N + 1)
nr     = len(ratios)

# --- Figure 1: Error, cost, runtime across ratios ---
fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4))

# Terminal error at z_des:
ax = axes1[0]
ax.semilogy(ratios, [results[r]['gt_err']  for r in ratios],
            '-s',  lw=2, label='IPOPT')
ax.semilogy(ratios, [results[r]['mls_err'] for r in ratios],
            '-o', lw=2, label='MLS')
ax.set_xlabel(r'$\|N\|/\|A\|$'); ax.set_ylabel(r'Terminal error $\|z_N - z_f\|$')
ax.set_title(r'Terminal Error at $z_{\rm des}$')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Control cost at z_des:
ax = axes1[1]
ax.semilogy(ratios, [results[r]['gt_cost']  for r in ratios],
        '-s',  lw=2, label='IPOPT')
ax.semilogy(ratios, [results[r]['mls_cost'] for r in ratios],
        '-o', lw=2, label='MLS')
ax.set_xlabel(r'$\|N\|/\|A\|$'); ax.set_ylabel(r'Control cost $\sum_k \|u_k\|^2$')
ax.set_title(r'Control Cost at $z_{\rm des}$')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Runtime:
ax = axes1[2]
ax.plot(ratios, [results[r]['t_gt']*1e3     for r in ratios],
        '-s',  lw=2, label='IPOPT')
ax.plot(ratios, [results[r]['t_online']*1e3 for r in ratios],
        '-o', lw=2, label='MLS')
ax.set_xlabel(r'$\|N\|/\|A\|$'); ax.set_ylabel('Time (ms)')
ax.set_title(r'Runtime at $z_{\rm des}$')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
fig1.tight_layout()

# --- Figure 2: Trajectories at z_des ---
fig2, axes2 = plt.subplots(n + m, nr, figsize=(4*nr, 11), sharex=True)
fig2.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.05,
                     wspace=0.15, hspace=0.10)
for col, ratio in enumerate(ratios):
    axes2[0, col].set_title(fr'$\|N\|/\|A\|={ratio}$',
                             fontsize=10, fontweight='bold')
for i in range(n): axes2[i, 0].set_ylabel(f'$z_{i+1}$', fontsize=8)
for j in range(m): axes2[n+j, 0].set_ylabel(f'$u_{j+1}$', fontsize=8)

for col, ratio in enumerate(ratios):
    r = results[ratio]
    for i in range(n):
        ax = axes2[i, col]
        ax.plot(steps, r['z_gt'][:,  i],  lw=2,
                label='IPOPT' if (col==0 and i==0) else None)
        ax.plot(steps, r['z_mls'][:, i], lw=2,
                label='MLS' if (col==0 and i==0) else None)
        ax.plot(steps, r['z_nom'][:, i], lw=2,
                label='NOM' if (col==0 and i==0) else None)
        ax.axhline(z_des[i], color='k', ls=':', lw=2, alpha=0.4)
        ax.grid(True, alpha=0.25)
    for j in range(m):
        ax = axes2[n+j, col]
        ax.step(np.arange(N), r['U_gt'][:,  j],
                lw=2, where='post')
        ax.step(np.arange(N), r['U_mls'][:, j],
                lw=2, where='post')
        ax.step(np.arange(N), r['U_nom'][:, j],
                lw=2, where='post')
        ax.grid(True, alpha=0.25)
    axes2[-1, col].set_xlabel('Time step')

axes2[0, 0].legend(fontsize=8, loc='best')

# Show plots:
plt.show()
