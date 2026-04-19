# Import packages:
import numpy as np
import matplotlib.pyplot as plt
import time

# === FUNCTIONS ===
# Define nonlinear injectors:
def ctrl_eff_fn(a):   return 1.0 - 2.0*(a/np.deg2rad(10.0))**2
def fluid_gain_fn(a): return 1.0/(1.0+(a/np.deg2rad(6.0))**2)

# Dynamics of the system:
def dynamics(z, u):
    h, hdot, alpha, alphadot, q1, q2, q3 = z
    de, da = u
    fg = fluid_gain_fn(alpha)
    ce = ctrl_eff_fn(alpha)
    return np.array([
        hdot,
        -0.5*h - 2.0*h**3 - 1.0*hdot - 0.2*alpha + fg*1.5*q1 - 2.0*ce*de + 0.3*ce*da,
        alphadot,
        -4.0*alpha - 1.0*alphadot + 0.1*h + fg*2.0*q2 - 0.1*q3 + 5.0*ce*de - 0.8*ce*da,
        -0.4*q1 + 0.10*alpha,
        -0.6*q2 + 0.15*alpha,
        -0.8*q3,
    ])

# RK4 to propagate:
def rk4(z, u):
    k1 = dynamics(z, u)
    k2 = dynamics(z + dt/2*k1, u)
    k3 = dynamics(z + dt/2*k2, u)
    k4 = dynamics(z + dt*k3,   u)
    return z + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# Simulate the dynamics:
def simulate(z0, U):
    z    = z0.copy()
    traj = [z.copy()]
    for u in U:
        z = rk4(z, u)
        traj.append(z.copy())
    return np.array(traj)


# OCG solve:
def ocg_solve_inner(z0, y_goal, N_rem, R, R_rate=None, u_prev=None,
                    N_probe=60, eps=0.02):

    # Define steps, simulate, and set observation gap:
    N_p   = min(N_probe, N_rem)
    z_nom = simulate(z0, np.zeros((N_p, m)))
    dy    = y_goal - C @ z_nom[-1]

    # Build empirical output controllability matrix C_o via finite differences:
    C_o = np.zeros((p, N_p * m))
    for tau in range(N_p):
        for j in range(m):
            up = np.zeros(m); up[j] =  eps
            un = np.zeros(m); un[j] = -eps
            zp = rk4(z_nom[tau], up)
            zn = rk4(z_nom[tau], un)
            for k in range(tau+1, N_p):
                zp = rk4(zp, np.zeros(m))
                zn = rk4(zn, np.zeros(m))
            C_o[:, tau*m+j] = C @ (zp - zn) / (2*eps)

    # Build block-diagonal R_N over horizon:
    R_N = np.kron(np.eye(N_p), R)

    # If rate penalty, incorporate rate penalty term in cost:
    if R_rate is not None and u_prev is not None:

        # Build diagonal of R:
        R_rate_N = np.kron(np.eye(N_p), R_rate)

        # First-difference matrix (gives u_k - u_{k-1}):
        D = np.eye(N_p * m) - np.eye(N_p * m, k=-m)

        # Build quadratic cost matrix (regular cost + rate penalty term):
        Q_eff = R_N + D.T @ R_rate_N @ D

        # Calculate linear term (shifts previous control from zero to true control):
        d_prev       = np.zeros(N_p * m)
        d_prev[:m]   = u_prev
        rhs_shift    = D.T @ R_rate_N @ d_prev

    # If no rate penalty, build as normal:
    else:

        # Cost is just R, no shift required:
        Q_eff     = R_N
        rhs_shift = np.zeros(N_p * m)

    # Weighted pseudoinverse:
    Q_inv   = np.linalg.inv(Q_eff)
    W       = C_o @ Q_inv @ C_o.T
    U_flat  = Q_inv @ C_o.T @ np.linalg.lstsq(W, dy, rcond=None)[0]

    # Apply linear rate correction (shift due to u_prev):
    U_flat  = U_flat + Q_inv @ rhs_shift

    # Clip control within bounds:
    U = np.clip(U_flat.reshape(N_p, m), min_ctrl, max_ctrl)
    if N_p < N_rem:
        U = np.vstack([U, np.zeros((N_rem - N_p, m))])
    return U

# Run RH-OCG loop:
def run_rh_ocg(label, R, R_rate=None, replan=5, N_probe=60, min_hor=15):

    # Initialize:
    print(f"  Running {label}...")
    t0     = time.perf_counter()
    z      = z_0.copy()
    u_prev = np.zeros(m)
    U_hist = []
    U_plan = None

    # Loop through each step, call inner loop for control, extract:
    for k in range(T):
        N_rem = T - k
        if (U_plan is None or k % replan == 0) and N_rem >= min_hor:
            U_plan = ocg_solve_inner(z, y_f, N_rem,
                                     R=R, R_rate=R_rate, u_prev=u_prev,
                                     N_probe=N_probe)
        u_k = np.zeros(m) if (U_plan is None or len(U_plan) == 0) \
              else U_plan[0].copy()
        if U_plan is not None and len(U_plan) > 0:
            U_plan = U_plan[1:]

        # Rate clip and actuator limits:
        u_k    = u_prev + np.clip(u_k - u_prev, -rate_clip, rate_clip)
        u_k    = np.clip(u_k, min_ctrl, max_ctrl)
        u_prev = u_k.copy()
        z      = rk4(z, u_k)
        U_hist.append(u_k.copy())

    # Pack controls and simulate:
    U_arr = np.array(U_hist)
    z_arr = simulate(z_0, U_arr)
    t_sol = time.perf_counter() - t0

    # Calculate error:
    errs  = [abs(z_arr[-1, i]*sc[i] - y_f[i]*sc[i]) for i in range(p)]

    # Compute control rates for plotting:
    du = np.diff(U_arr, axis=0, prepend=U_arr[[0]])

    print(f"    h={errs[0]:.4f}m  alpha={errs[2]:.3f}deg  t={t_sol:.1f}s")
    return dict(label=label, z=z_arr, U=U_arr, dU=du, e=errs, t=t_sol)


# === SETUP ===
# Define system settings:
dt = 0.05
T  = 200

# Initial and final conditions:
z_0 = np.array([0.0, 0.0, np.deg2rad(-10.0), 0.0, 0.3, -0.2, 0.1])
y_f = np.array([0.2, 0.0, np.deg2rad(12.0), 0.0])

# Rate limits:
min_ctrl = -np.deg2rad(40.0)
max_ctrl =  np.deg2rad(40.0)
rate_clip = np.deg2rad(8.0)

# Unpack shapes:
n, m, p = 7, 2, 4

# Observation of the system:
C = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
])

state_labs = ['h (m)', 'hdot (m/s)', 'alpha (deg)', 'alphadot (deg/s)']
ctrl_labs  = ['Elevator (deg)', 'Aileron (deg)']
sc         = [1, 1, np.rad2deg(1), np.rad2deg(1)]


# === RUN CASES ===
# Baseline R (diagonal):
R_base = np.diag([0.05, 0.1])

# Off-diagonal R:
R_offdiag = np.array([[0.05, 0.03],
                       [0.03, 0.1]])

# Rate weighting:
R_rate = np.diag([0.5, 0.5])

# Run cases:
print("Running RH-OCG variants...")
res_R        = run_rh_ocg('R only',         R=R_base)
res_R_rate   = run_rh_ocg('R + rate',       R=R_base,    R_rate=R_rate)
res_R_offdiag= run_rh_ocg('R off-diagonal', R=R_offdiag)

# Pack results:
results = {
    'r_only'   : res_R,
    'r_rate'   : res_R_rate,
    'r_offdiag': res_R_offdiag,
}

# Print summary:
print(f"\n{'='*65}")
print(f"  {'Strategy':<22} {'h err (m)':>10} {'a err (deg)':>12} {'Time (s)':>10}")
print(f"  {'-'*62}")
for r in results.values():
    print(f"  {r['label']:<22} {r['e'][0]:>10.4f} {r['e'][2]:>12.3f} {r['t']:>10.1f}")
print(f"{'='*65}")


# === PLOTTING ===
time_ax = np.arange(T+1) * dt
t_u     = np.arange(T)   * dt
xfp     = [y_f[i]*sc[i] for i in range(p)]

colors = ['steelblue', 'seagreen', 'darkorange']
lss    = ['-', '--', '-.']

# --- Figure 1: State and control trajectories ---
fig1, axes1 = plt.subplots(p + m, 1, figsize=(10, 12), sharex=True)

for i in range(p):
    ax = axes1[i]
    for r, col, ls in zip(results.values(), colors, lss):
        ax.plot(time_ax, r['z'][:, i]*sc[i],
                color=col, linewidth=2, linestyle=ls, label=r['label'])
    ax.axhline(xfp[i], color='r', linestyle=':', linewidth=1.5,
               label='Target' if i == 0 else '')
    ax.set_ylabel(state_labs[i], fontsize=9)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

for i in range(m):
    ax = axes1[p + i]
    for r, col, ls in zip(results.values(), colors, lss):
        ax.plot(t_u, np.rad2deg(r['U'][:, i]),
                color=col, linewidth=2, linestyle=ls, label=r['label'])
    ax.set_ylabel(ctrl_labs[i], fontsize=9)
    ax.grid(True, alpha=0.3)

axes1[-1].set_xlabel('Time (s)', fontsize=9)
axes1[-1].set_xlim([0, T*dt])
fig1.suptitle('RH-OCG: R only vs R+rate vs R off-diagonal\nTrajectories and Controls',
              fontsize=12)
fig1.tight_layout()

# --- Figure 2: Control rates ---
fig2, axes2 = plt.subplots(m, 1, figsize=(10, 5), sharex=True)

for i in range(m):
    ax = axes2[i]
    for r, col, ls in zip(results.values(), colors, lss):
        ax.plot(t_u, np.rad2deg(r['dU'][:, i]),
                color=col, linewidth=1.5, linestyle=ls, label=r['label'])
    ax.set_ylabel(f'd({ctrl_labs[i]})/step  (deg)', fontsize=9)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

axes2[-1].set_xlabel('Time (s)', fontsize=9)
axes2[-1].set_xlim([0, T*dt])
fig2.suptitle('Control Rates  (du_k = u_k - u_{k-1})', fontsize=12)
fig2.tight_layout()

# --- Figure 3: Accuracy and timing comparison ---
labels = [r['label'] for r in results.values()]
x      = np.arange(len(results))
width  = 0.28

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 4))

# Terminal errors:
ax = axes3[0]
h_errs = [r['e'][0] for r in results.values()]
a_errs = [r['e'][2] for r in results.values()]
bars_h = ax.bar(x - width/2, h_errs, width, color=colors, alpha=0.9,
                edgecolor='k', linewidth=0.8, label='h error (m)')
bars_a = ax.bar(x + width/2, a_errs, width, color=colors, alpha=0.4,
                edgecolor='k', linewidth=0.8, hatch='//', label='α error (°)')
for bar, v in zip(list(bars_h) + list(bars_a), h_errs + a_errs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
            f'{v:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Terminal error')
ax.set_title('Terminal Error')
ax.legend(fontsize=8)
ax.grid(True, axis='y', alpha=0.3)

# Timing:
ax = axes3[1]
times = [r['t'] for r in results.values()]
bars  = ax.bar(x, times, color=colors, edgecolor='k', linewidth=0.8, width=0.4)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(times)*0.01,
            f'{t:.1f}s', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Computation time (s)')
ax.set_title('Computational Cost')
ax.grid(True, axis='y', alpha=0.3)

fig3.suptitle('RH-OCG Variant Comparison: Accuracy, Smoothness, and Cost',
              fontsize=12)
fig3.tight_layout()

plt.show()
