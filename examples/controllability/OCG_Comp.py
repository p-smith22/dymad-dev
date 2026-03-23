"""
Full OCG Strategy Comparison
==============================
Side-by-side comparison of all strategies from the paper:

  1. LTI OCG (trim)         — single linearization at alpha=0
  2. LTV OCG (ref traj)     — linearize along free-response nominal
  3. LTV OCG (RBF maps)     — explicit A(y),B(y) via RBF interpolation
  4. Empirical OCG (open)   — single-shot empirical Gramian
  5. RH-OCG (dual-pass)    — receding horizon + Newton refinement

==============================
"""

# Import packages:
import numpy as np
import scipy as sp
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import time

# --- Define System ---
# Define time information:
dt = 0.05
T  = 200

# Define initial and final states:
z_0 = np.array([0.0, 0.0, np.deg2rad(-10.0), 0.0, 0.3, -0.2, 0.1])
y_f = np.array([0.2, 0.0, np.deg2rad(12.0), 0.0])

# Define bounds:
min_ctrl = -np.deg2rad(40.0)
max_ctrl = -min_ctrl
rate_clip = np.deg2rad(8)

# Define sizes and observation:
n, m, p = 7, 2, 4
C = np.array([
    [1, 0, 0, 0, 0, 0, 0],  # h
    [0, 1, 0, 0, 0, 0, 0],  # hdot
    [0, 0, 1, 0, 0, 0, 0],  # alpha
    [0, 0, 0, 1, 0, 0, 0],  # alphadot
])

# Define labels:
state_labs = ['h (m)', 'hdot (m/s)', 'alpha (deg)', 'alphadot (deg/s)']
ctrl_labs  = ['Elevator (deg)', 'Aileron (deg)']

# Scaling factor for error:
sc = [1, 1, np.rad2deg(1), np.rad2deg(1)]

# Define dynamics:
def ctrl_eff_fn(a):   return 1.0 - 2.0*(a/np.deg2rad(10.0))**2
def fluid_gain_fn(a): return 1.0/(1.0+(a/np.deg2rad(6.0))**2)
def dynamics(z, u):

    # Unpack states and controls:
    h, hdot, alpha, alphadot, q1, q2, q3 = z
    de, da = u

    # Compute fluid and control effects:
    fg = fluid_gain_fn(alpha)
    ce = ctrl_eff_fn(alpha)

    # Return states_dot:
    return np.array([hdot,
                     -0.5*h - 2.0*h**3 - 1.0*hdot - 0.2*alpha + fg*1.5*q1 - 2.0*ce*de + 0.3*ce*da,
                     alphadot,
                     -4.0*alpha - 1.0*alphadot + 0.1*h + fg*2.0*q2 - 0.1*q3 + 5.0*ce*de - 0.8*ce*da,
                     -0.4*q1+0.10*alpha, -0.6*q2+0.15*alpha, -0.8*q3,
    ])

# Define RK4 integrator:
def rk4(z, u):

    # Define each part:
    k1 = dynamics(z, u)
    k2 = dynamics(z + dt/2*k1, u)
    k3 = dynamics(z + dt/2*k2, u)
    k4 = dynamics(z + dt*k3, u)

    # Combine and return integral:
    return z + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# Simulate dynamics:
def simulate(z0, U):

    # Copy beginning of trajectory:
    z = z0.copy()
    traj = [z.copy()]

    # Loop through each control step:
    for u in U:

        # Compute next step:
        z = rk4(z, u)
        traj.append(z.copy())

    # Return trajectory:
    return np.array(traj)

# Get linearized continuous-time matrices via finite difference:
def linearize_ct(z_op, eps=1e-5):

    # Initialize:
    u0 = np.zeros(m)
    Ac = np.zeros((n, n))
    Bc = np.zeros((n, m))

    # Get function at current step:
    f0 = dynamics(z_op, u0)

    # Loop through each state:
    for i in range(n):

        # Linearize wrt state:
        zp = z_op.copy()
        zp[i] += eps
        Ac[:, i] = (dynamics(zp, u0) - f0) / eps

    # Loop through each control:
    for i in range(m):

        # Linearize wrt control:
        up = u0.copy()
        up[i] += eps
        Bc[:, i] = (dynamics(z_op, up) - f0) / eps

    # Return linearized matrices:
    return Ac, Bc

# Discretize continuous-time linearization via Zero-Order Hold (ZOH):
def linearize_zoh(z_op):

    # Fetch continuous-time matrices:
    Ac, Bc = linearize_ct(z_op)

    # Build augmented matrix for ZOH discretization:
    M = np.zeros((n+m, n+m))
    M[:n, :n] = Ac*dt
    M[:n, n:] = Bc*dt

    # Matrix exponential gives exact discrete equivalent:
    eM = sp.linalg.expm(M)

    # Return discrete matrices Ad, Bd:
    return eM[:n, :n], eM[:n, n:]

# Initialize results:
results = {}

# --- STRATEGY 1: LTI OCG (trim) ---
print("[1/5] LTI OCG (trim)...")
t0 = time.perf_counter()

# Get discrete matrices at trim (alpha=0, all states zero):
Ad_trim, Bd_trim = linearize_zoh(np.zeros(n))

# Initialize controllability matrix and transition matrix:
C_o = np.zeros((p, T*m))
Phi = np.eye(n)

# Construct output controllability matrix backwards in time.
# At each tau, Phi holds Phi(T, tau+1) = A^{T-1-tau}, so
# C_o[:,tau] = C @ Phi(T,tau+1) @ B, then left-multiply Phi by A.
for tau in range(T-1, -1, -1):
    C_o[:, tau*m:(tau+1)*m] = C @ Phi @ Bd_trim
    Phi = Ad_trim @ Phi

# Construct output controllability Gramian:
W  = C_o @ C_o.T

# Define observation gap (where free response lands vs. target):
dy = y_f - C @ Phi @ z_0

# Solve for minimum-energy control and apply bounds:
U  = np.clip((C_o.T @ np.linalg.lstsq(W, dy, rcond=None)[0]).reshape(T, m), min_ctrl, max_ctrl)

# Simulate and end timer:
z  = simulate(z_0, U)
t  = time.perf_counter() - t0

# Calculate error and store:
e = [abs(z[-1,i]*sc[i] - y_f[i]*sc[i]) for i in range(p)]
results['lti'] = dict(label='LTI', z=z, U=U, e=e, t=t,
                      color='tab:orange', ls='--')
print(f"   h={e[0]:.4f}m  alpha={e[2]:.3f}°  t={t:.2f}s")


# --- STRATEGY 2: LTV OCG (reference trajectory) ---
print("[2/5] LTV OCG (reference trajectory)...")
t0 = time.perf_counter()

# Simulate a zero-control nominal trajectory to linearize along:
z_nom = simulate(z_0, np.zeros((T, m)))

# Linearize at each step along the nominal:
Ad_list = []
Bd_list = []
for k in range(T):
    Ad_k, Bd_k = linearize_zoh(z_nom[k])
    Ad_list.append(Ad_k); Bd_list.append(Bd_k)

# Construct output controllability matrix backwards in time,
# using the time-varying A_k, B_k sequence along the nominal:
C_o = np.zeros((p, T*m))
Phi = np.eye(n)
for tau in range(T-1, -1, -1):
    C_o[:, tau*m:(tau+1)*m] = C @ Phi @ Bd_list[tau]
    Phi = Phi @ Ad_list[tau]

# Construct output controllability Gramian:
W  = C_o @ C_o.T

# Observation gap using the linearized free response (Phi is now Phi(T,0)):
dy = y_f - C @ Phi @ z_0

# Solve for minimum-energy control and apply bounds:
U  = np.clip((C_o.T @ np.linalg.lstsq(W, dy, rcond=None)[0]).reshape(T, m), min_ctrl, max_ctrl)

# Simulate and end timer:
z  = simulate(z_0, U)
t  = time.perf_counter() - t0

# Calculate error and store:
e = [abs(z[-1,i]*sc[i] - y_f[i]*sc[i]) for i in range(p)]
results['ref'] = dict(label='Reference Traj', z=z, U=U, e=e, t=t,
                      color='tab:purple', ls='--')
print(f"   h={e[0]:.4f}m  alpha={e[2]:.3f}°  t={t:.2f}s")


# --- STRATEGY 3: LTV OCG (RBF maps) ---
print("[3/5] LTV OCG (RBF maps)...  building maps...", end='', flush=True)
t0 = time.perf_counter()

# Define training grid over (h, alpha) operating points:
h_pts = np.linspace(-0.4, 0.4, 10)
a_pts = np.unique(np.concatenate([
    np.linspace(np.deg2rad(-15), np.deg2rad(15), 12),
    np.linspace(np.deg2rad(-9),  np.deg2rad(-5),  8),
    np.linspace(np.deg2rad(5),   np.deg2rad(9),   8),
]))

# Linearize at each training point and collect into arrays:
pts_rbf = []; Ac_all = []; Bc_all = []
for h in h_pts:
    for alpha in a_pts:
        q1 = 0.10*alpha/0.4; q2 = 0.15*alpha/0.6
        z_op = np.array([h, 0., alpha, 0., q1, q2, 0.])
        Ac, Bc = linearize_ct(z_op)
        pts_rbf.append([h, alpha]); Ac_all.append(Ac); Bc_all.append(Bc)

pts_rbf = np.array(pts_rbf); Ac_arr = np.array(Ac_all); Bc_arr = np.array(Bc_all)

# Fit one RBF interpolant per matrix entry:
Ac_rbf = [[RBFInterpolator(pts_rbf, Ac_arr[:,i,j], kernel='thin_plate_spline', smoothing=1e-6)
            for j in range(n)] for i in range(n)]
Bc_rbf = [[RBFInterpolator(pts_rbf, Bc_arr[:,i,j], kernel='thin_plate_spline', smoothing=1e-6)
            for j in range(m)] for i in range(n)]

# Evaluate RBF maps at a given output y to get discrete A, B:
def eval_AB_rbf(y):

    # Package query point (h, alpha):
    pt = np.array([[y[0], y[2]]])

    # Evaluate each matrix entry from its RBF interpolant:
    Ac = np.zeros((n, n)); Bc = np.zeros((n, m))
    for i in range(n):
        for j in range(n): Ac[i, j] = Ac_rbf[i][j](pt)[0]
        for j in range(m): Bc[i, j] = Bc_rbf[i][j](pt)[0]

    # Discretize via ZOH:
    M = np.zeros((n+m, n+m)); M[:n,:n] = Ac*dt; M[:n,n:] = Bc*dt
    eM = sp.linalg.expm(M)
    return eM[:n, :n], eM[:n, n:]

print(" solving...", end='', flush=True)

# Evaluate A, B along the nominal trajectory using the RBF maps:
Ad_rbf_list = []; Bd_rbf_list = []
for k in range(T):
    Ad_k, Bd_k = eval_AB_rbf(C @ z_nom[k])
    Ad_rbf_list.append(Ad_k); Bd_rbf_list.append(Bd_k)

# Construct output controllability matrix backwards in time:
C_o = np.zeros((p, T*m)); Phi = np.eye(n)
for tau in range(T-1, -1, -1):
    C_o[:, tau*m:(tau+1)*m] = C @ Phi @ Bd_rbf_list[tau]
    Phi = Phi @ Ad_rbf_list[tau]

# Construct output controllability Gramian:
W  = C_o @ C_o.T

# Observation gap using exact nonlinear free response:
dy = y_f - C @ z_nom[-1]

# Solve for minimum-energy control and apply bounds:
U  = np.clip((C_o.T @ np.linalg.lstsq(W, dy, rcond=None)[0]).reshape(T, m), min_ctrl, max_ctrl)

# Simulate and end timer:
z  = simulate(z_0, U)
t  = time.perf_counter() - t0

# Calculate error and store:
e = [abs(z[-1,i]*sc[i] - y_f[i]*sc[i]) for i in range(p)]
results['rbf'] = dict(label='RBF Maps', z=z, U=U, e=e, t=t,
                      color='tab:green', ls='--')
print(f"\n   h={e[0]:.4f}m  alpha={e[2]:.3f}°  t={t:.2f}s")


# --- STRATEGY 4: Empirical OCG (open-loop) ---
print("[4/5] Empirical OCG (open-loop)...")
t0 = time.perf_counter()

eps   = 0.02
z_nom = simulate(z_0, np.zeros((T, m)))

# Observation gap from the exact nonlinear free response:
dy = y_f - C @ z_nom[-1]

# Build each column of C_o empirically via central-difference probing.
# For each (tau, j): fire +/- eps impulse at input j at step tau,
# coast to end under zero input, take C @ (z+ - z-) / 2eps:
C_o = np.zeros((p, T*m))
for tau in range(T):
    for j in range(m):
        up = np.zeros(m); up[j] =  eps
        un = np.zeros(m); un[j] = -eps
        zp = rk4(z_nom[tau], up); zn = rk4(z_nom[tau], un)
        for k in range(tau+1, T):
            zp = rk4(zp, np.zeros(m)); zn = rk4(zn, np.zeros(m))
        C_o[:, tau*m+j] = C @ (zp - zn) / (2*eps)

# Construct output controllability Gramian:
W = C_o @ C_o.T

# Solve for minimum-energy control and apply bounds:
U = np.clip((C_o.T @ np.linalg.lstsq(W, dy, rcond=None)[0]).reshape(T, m), min_ctrl, max_ctrl)

# Simulate and end timer:
z = simulate(z_0, U)
t = time.perf_counter() - t0

# Calculate error and store:
e = [abs(z[-1,i]*sc[i] - y_f[i]*sc[i]) for i in range(p)]
results['emp'] = dict(label='Empirical OCG', z=z, U=U, e=e, t=t,
                      color='tab:blue', ls='-.')
print(f"   h={e[0]:.4f}m  alpha={e[2]:.3f}°  t={t:.2f}s")


# --- STRATEGY 5: RH-OCG (dual-pass) ---
print("[5/5] RH-OCG (dual-pass)...")

def ocg_solve_inner(z0, y_goal, N_rem, N_probe=60, eps=0.02):

    # Cap probe horizon at remaining steps:
    N_p    = min(N_probe, N_rem)

    # Simulate zero-control nominal from current state:
    z_nom_ = simulate(z0, np.zeros((N_p, m)))

    # Observation gap:
    dy = y_goal - C @ z_nom_[-1]

    # Build empirical C_o via central-difference probing:
    C_o = np.zeros((p, N_p*m))
    for tau in range(N_p):
        for j in range(m):
            up = np.zeros(m); up[j] =  eps
            un = np.zeros(m); un[j] = -eps
            zp = rk4(z_nom_[tau], up); zn = rk4(z_nom_[tau], un)
            for k in range(tau+1, N_p):
                zp = rk4(zp, np.zeros(m)); zn = rk4(zn, np.zeros(m))
            C_o[:, tau*m+j] = C @ (zp - zn) / (2*eps)

    # Construct Gramian and solve:
    W   = C_o @ C_o.T
    U_p = np.clip((C_o.T @ np.linalg.lstsq(W, dy, rcond=None)[0]).reshape(N_p, m), min_ctrl, max_ctrl)

    # Pad with zeros if probe horizon shorter than remaining steps:
    U = np.vstack([U_p, np.zeros((N_rem-N_p, m))]) if N_p < N_rem else U_p
    return U, dy, C_o

t0 = time.perf_counter()
n_passes  = 2
replan    = 5
n_probe   = 60
min_hor   = 15

# --- Pass 1: RH-OCG ---
z = z_0.copy(); U_hist = []; U_plan = None; u_prev = np.zeros(m)
C_o_acc = np.zeros((p, T*m)); C_o_last = None; k_replan = 0

for k in range(T):
    N_rem = T - k

    # Replan if first step, or every `replan` steps, and horizon is long enough:
    if (U_plan is None or k%replan == 0) and N_rem >= min_hor:
        U_plan, delta_y, C_o_last = ocg_solve_inner(z, y_f, N_rem, N_probe=n_probe)
        k_replan = k

    # Accumulate C_o columns from the most recent plan into a full-horizon matrix.
    # tau_loc is the local offset within the current plan window:
    if C_o_last is not None:
        tau_loc = k - k_replan
        if tau_loc*m < C_o_last.shape[1]:
            C_o_acc[:, k*m:(k+1)*m] = C_o_last[:, tau_loc*m:(tau_loc+1)*m]

    # Apply next planned control (fall back to zero if plan exhausted):
    if U_plan is None or len(U_plan) == 0: u_k = np.zeros(m)
    else: u_k = U_plan[0]; U_plan = U_plan[1:]

    # Rate-limit the control:
    u_k    = u_prev + np.clip(u_k - u_prev, -rate_clip, rate_clip)
    u_prev = u_k.copy(); z = rk4(z, u_k); U_hist.append(u_k.copy())

U_passes = [np.array(U_hist)]; z_passes = [simulate(z_0, U_passes[0])]
e_passes = [[abs(z_passes[0][-1,i]*sc[i] - y_f[i]*sc[i]) for i in range(p)]]
print(f"   Pass1: h={e_passes[0][0]:.4f}m  alpha={e_passes[0][2]:.3f}°")

# --- Pass 2: free correction using accumulated C_o ---
# W_out downweights hdot and alphadot since alpha=8° is not a trim point —
# those velocities won't be zero and we don't want to penalize them:
W_out = np.diag([1.0, 0.0, np.rad2deg(1.0), 0.0])
dy2   = y_f - C @ z_passes[0][-1]

# Solve for correction direction using the weighted accumulated Gramian:
C_o_w = W_out @ C_o_acc
W_acc = C_o_w @ C_o_w.T
du    = (C_o_w.T @ np.linalg.lstsq(W_acc, W_out @ dy2, rcond=None)[0]).reshape(T, m)

# Armijo line search: shrink step until weighted residual improves:
err0 = np.linalg.norm(W_out @ dy2); alpha_ls = 1.0; U2 = U_passes[0].copy()
for _ in range(20):
    U_try = np.clip(U_passes[0] + alpha_ls*du, min_ctrl, max_ctrl)
    z_try = simulate(z_0, U_try)
    if np.all(np.isfinite(z_try)) and np.linalg.norm(W_out @ (y_f - C @ z_try[-1])) < err0:
        U2 = U_try; break
    alpha_ls *= 0.5

z2 = simulate(z_0, U2); U_passes.append(U2); z_passes.append(z2)
e_passes.append([abs(z2[-1,i]*sc[i] - y_f[i]*sc[i]) for i in range(p)])
print(f"   Pass2: h={e_passes[1][0]:.4f}m  alpha={e_passes[1][2]:.3f}°")

# Pick best pass by combined h + alpha error:
best_idx = min(range(len(e_passes)), key=lambda i: e_passes[i][0] + np.deg2rad(e_passes[i][2]))
z_rh = z_passes[best_idx]; U_rh = U_passes[best_idx]; e_rh = e_passes[best_idx]
t_rh = time.perf_counter() - t0
results['rh'] = dict(label='RH-OCG', z=z_rh, U=U_rh, e=e_rh, t=t_rh,
                     color='tab:red', ls='-')
print(f"   Best (pass {best_idx+1}): h={e_rh[0]:.4f}m  alpha={e_rh[2]:.3f}°  t={t_rh:.1f}s")

# Zero control baseline:
z_zc = simulate(z_0, np.zeros((T, m)))


# --- SUMMARY TABLE ---
xfp = [y_f[i]*sc[i] for i in range(p)]
print(f"\n{'='*66}")
print(f"{'Strategy':<28}   {'h err (m)':>10}   {'α err (°)':>10} {'Time':>8}")
print('-'*66)
for k,r in results.items():
    print(f"  {r['label']:<26} {r['e'][0]:>10.4f}   {r['e'][2]:>10.3f}   {r['t']:>7.1f}s")
print('='*66)


# --- FIGURE 1: State trajectories ---
time_ax = np.arange(T+1)*dt; t_u = np.arange(T)*dt

fig1, axes = plt.subplots(3, 2, figsize=(12, 9), sharex='col')
fig1.subplots_adjust(hspace=0.08, wspace=0.28)

# States — only bottom row of each column gets x label:
for i in range(p):
    ax = axes[i//2, i%2]
    ax.plot(time_ax, z_zc[:,i]*sc[i], '#cccccc', lw=1.0, ls=':', label='No control', zorder=1)
    for k,r in results.items():
        ax.plot(time_ax, r['z'][:,i]*sc[i], lw=2.0 if k=='rh' else 1.3,
                color=r['color'], ls=r['ls'], label=r['label'], zorder=3 if k=='rh' else 2)
    ax.axhline(xfp[i], color='k', ls=':', lw=1.2, label='Target', zorder=4)
    ax.set_ylabel(state_labs[i], fontsize=8)
    ax.grid(True, alpha=0.2); ax.tick_params(labelsize=7)
    if i < 2:  # only top row gets legend
        ax.legend(fontsize=6.5, loc='best', ncol=2)

# Controls:
for i in range(m):
    ax = axes[2, i]
    ax.plot(t_u, np.zeros(T), '#cccccc', lw=1.0, ls=':', zorder=1)
    for k,r in results.items():
        U = r['U']
        if len(U) < T: U = np.vstack([U, np.zeros((T-len(U), m))])
        ax.plot(t_u, np.rad2deg(U[:,i]), lw=2.0 if k=='rh' else 1.3,
                color=r['color'], ls=r['ls'], zorder=3 if k=='rh' else 2)
    ax.set_ylabel(ctrl_labs[i], fontsize=8)
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.grid(True, alpha=0.2); ax.tick_params(labelsize=7)

fig1.suptitle('OCG Strategy Comparison — Trajectories & Controls', fontsize=10, fontweight='bold')


# --- FIGURE 2: Error and computation time ---
h_errs = [r['e'][0] for r in results.values()]
a_errs = [r['e'][2] for r in results.values()]
times  = [r['t']    for r in results.values()]
cols   = [r['color'] for r in results.values()]
labels = [r['label'] for r in results.values()]
x = np.arange(len(results)); w = 0.32

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))

# Final errors (h and alpha):
ax = axes2[0]
ax.bar(x-w/2, h_errs, w, color=cols, alpha=0.9,  label='h error (m)')
ax.bar(x+w/2, a_errs, w, color=cols, alpha=0.5, hatch='//', label='α error (°)')
for xi,(h,a) in enumerate(zip(h_errs, a_errs)):
    ax.text(xi-w/2, h*1.05, f'{h:.4f}',  ha='center', va='bottom', fontsize=7, rotation=90)
    ax.text(xi+w/2, a*1.05, f'{a:.3f}°', ha='center', va='bottom', fontsize=7, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('Final Error', fontsize=10)
ax.set_title('Output Errors', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis='y')
ax.tick_params(labelsize=8)

# Computation time:
ax = axes2[1]
ax.bar(x, times, color=cols, alpha=0.9, width=0.5)
for xi,t in enumerate(times):
    ax.text(xi, t*1.05, f'{t:.1f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('Computation Time (s)', fontsize=10)
ax.set_title('Computation Time', fontsize=11)
ax.grid(True, alpha=0.2, axis='y')
ax.tick_params(labelsize=8)

fig2.suptitle('OCG Strategy Comparison — Accuracy vs Computation Time',
              fontsize=11, fontweight='bold')

plt.show()
