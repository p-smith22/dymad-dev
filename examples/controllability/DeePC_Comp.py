"""
OCG Strategy Comparison: DeePC vs RH-OCG (Empirical)
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import time
import cvxpy as cp

# --- Define System ---
dt = 0.05
T  = 200

z_0 = np.array([0.0, 0.0, np.deg2rad(-10.0), 0.0, 0.3, -0.2, 0.1])
y_f = np.array([0.2, 0.0, np.deg2rad(12.0), 0.0])

min_ctrl = -np.deg2rad(40.0)
max_ctrl = -min_ctrl

n, m, p = 7, 2, 4
C = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
])

state_labs = ['h (m)', 'hdot (m/s)', 'alpha (deg)', 'alphadot (deg/s)']
ctrl_labs  = ['Elevator (deg)', 'Aileron (deg)']
sc = [1, 1, np.rad2deg(1), np.rad2deg(1)]

def ctrl_eff_fn(a):   return 1.0 - 2.0*(a/np.deg2rad(10.0))**2
def fluid_gain_fn(a): return 1.0/(1.0+(a/np.deg2rad(6.0))**2)

def dynamics(z, u):
    h, hdot, alpha, alphadot, q1, q2, q3 = z
    de, da = u
    fg = fluid_gain_fn(alpha)
    ce = ctrl_eff_fn(alpha)
    return np.array([hdot,
                     -0.5*h - 2.0*h**3 - 1.0*hdot - 0.2*alpha + fg*1.5*q1 - 2.0*ce*de + 0.3*ce*da,
                     alphadot,
                     -4.0*alpha - 1.0*alphadot + 0.1*h + fg*2.0*q2 - 0.1*q3 + 5.0*ce*de - 0.8*ce*da,
                     -0.4*q1+0.10*alpha, -0.6*q2+0.15*alpha, -0.8*q3])

def rk4(z, u):
    k1 = dynamics(z, u)
    k2 = dynamics(z + dt/2*k1, u)
    k3 = dynamics(z + dt/2*k2, u)
    k4 = dynamics(z + dt*k3, u)
    return z + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def simulate(z0, U):
    z = z0.copy()
    traj = [z.copy()]
    for u in U:
        z = rk4(z, u)
        traj.append(z.copy())
    return np.array(traj)

results = {}

# --- STRATEGY 1: DeePC ---
print("[1/2] DeePC...")
t0 = time.perf_counter()

T_ini         = 5
L             = 30
lam_g         = 20.0
lam_h         = 2e3
Q_term_d      = np.diag([50.0, 1.0, 50.0, 1.0])
Q_run_d       = np.diag([ 1.0, 0.1,  1.0, 0.1])
R_deepc       = np.diag([0.05, 0.05])
replan_deepc  = 5
min_hor_deepc = T_ini + L + 2

# Offline data collection:
rng   = np.random.default_rng(42)
seg   = 400
z_mid1 = np.array([0.10, 0.0, np.deg2rad( 2.0), 0.0,  0.10, -0.10,  0.05])
z_mid2 = np.array([0.14, 0.0, np.deg2rad( 6.0), 0.0,  0.07, -0.07,  0.03])
z_tgt  = np.array([0.18, 0.0, np.deg2rad(10.0), 0.0,  0.05, -0.05,  0.02])
ICs_d  = [z_0, z_mid1, z_mid2, z_tgt]

u_d_segs, y_d_segs = [], []
for ic in ICs_d:
    t_off = np.arange(seg) * dt
    u_seg = np.zeros((seg, m))
    for f in np.linspace(0.1, 3.0, 10):
        phase = rng.uniform(0, 2*np.pi, m)
        u_seg += np.deg2rad(5.0) * np.sin(2*np.pi*f*t_off[:, None] + phase[None, :])
    u_seg = np.clip(u_seg, min_ctrl, max_ctrl)
    z_off = ic.copy(); y_seg = []; safe = True
    for k in range(seg):
        y_seg.append(C @ z_off)
        z_off = rk4(z_off, u_seg[k])
        if np.any(np.abs(z_off) > 1e3): safe = False; break
    if safe:
        u_d_segs.append(u_seg)
        y_d_segs.append(np.array(y_seg))

u_d = np.vstack(u_d_segs)
y_d = np.vstack(y_d_segs)
T_d = len(u_d)

# Build Hankel matrix:
depth = T_ini + L
N_col = T_d - depth + 1

def build_hankel(seq, depth):
    dim = seq.shape[1]
    Nc  = seq.shape[0] - depth + 1
    H   = np.zeros((depth * dim, Nc))
    for i in range(depth):
        H[i*dim:(i+1)*dim, :] = seq[i:i+Nc].T
    return H

H_u_d = build_hankel(u_d, depth)
H_y_d = build_hankel(y_d, depth)

H_full_raw = np.vstack([H_u_d[:T_ini*m, :], H_y_d[:T_ini*p, :],
                         H_u_d[T_ini*m:, :], H_y_d[T_ini*p:, :]])
col_norms_d = np.linalg.norm(H_full_raw, axis=0)
col_norms_d[col_norms_d < 1e-12] = 1.0
H_full_d = H_full_raw / col_norms_d[None, :]

nup_d = T_ini * m;  nyp_d = T_ini * p
Hup_d = H_full_d[:nup_d, :]
Hyp_d = H_full_d[nup_d:nup_d+nyp_d, :]
Huf_d = H_full_d[nup_d+nyp_d:nup_d+nyp_d+L*m, :]
Hyf_d = H_full_d[nup_d+nyp_d+L*m:, :]

def deepc_solve(u_ini_flat, y_ini_flat, y_target):
    g     = cp.Variable(N_col)
    u_seq = cp.Variable(L * m)
    y_seq = cp.Variable(L * p)
    hankel_pen = lam_h * (
        cp.sum_squares(Hup_d @ g - u_ini_flat) +
        cp.sum_squares(Hyp_d @ g - y_ini_flat) +
        cp.sum_squares(Huf_d @ g - u_seq)      +
        cp.sum_squares(Hyf_d @ g - y_seq)
    )
    cost = lam_g * cp.sum_squares(g) + hankel_pen
    for t in range(L - 1):
        cost += (cp.quad_form(y_seq[t*p:(t+1)*p] - y_target, Q_run_d) +
                 cp.quad_form(u_seq[t*m:(t+1)*m],             R_deepc))
    cost += (cp.quad_form(y_seq[(L-1)*p:] - y_target, Q_term_d) +
             cp.quad_form(u_seq[(L-1)*m:],             R_deepc))
    prob = cp.Problem(cp.Minimize(cost),
                      [u_seq >= min_ctrl, u_seq <= max_ctrl])
    prob.solve(solver=cp.OSQP, warm_starting=True,
               eps_abs=1e-5, eps_rel=1e-5, max_iter=8000, verbose=False)
    if prob.status not in ('optimal', 'optimal_inaccurate') or u_seq.value is None:
        return np.zeros((L, m))
    return u_seq.value.reshape(L, m)

# Online MPC loop:
z = z_0.copy(); U_hist_deepc = []; U_plan = None; u_prev = np.zeros(m)
u_buf = np.zeros((T_ini, m)); y_buf = np.array([C @ z_0] * T_ini)
rate_clip = np.deg2rad(8)

for k in range(T):
    N_rem = T - k
    if (U_plan is None or k % replan_deepc == 0) and N_rem >= min_hor_deepc:
        U_plan = deepc_solve(u_buf.flatten(), y_buf.flatten(), y_f)
    u_k = np.zeros(m) if (U_plan is None or len(U_plan) == 0) else U_plan[0]
    if U_plan is not None and len(U_plan) > 0: U_plan = U_plan[1:]
    u_k    = u_prev + np.clip(u_k - u_prev, -rate_clip, rate_clip)
    u_k    = np.clip(u_k, min_ctrl, max_ctrl); u_prev = u_k.copy()
    z      = rk4(z, u_k); y_meas = C @ z
    u_buf  = np.roll(u_buf, -1, axis=0); u_buf[-1]  = u_k
    y_buf  = np.roll(y_buf, -1, axis=0); y_buf[-1]  = y_meas
    U_hist_deepc.append(u_k.copy())

z_deepc = simulate(z_0, np.array(U_hist_deepc))
t       = time.perf_counter() - t0
e = [abs(z_deepc[-1,i]*sc[i] - y_f[i]*sc[i]) for i in range(p)]
results['deepc'] = dict(label='DeePC', z=z_deepc,
                        U=np.array(U_hist_deepc), e=e, t=t,
                        color='tab:brown', ls='-.')
print(f"   h={e[0]:.4f}m  alpha={e[2]:.3f}°  t={t:.1f}s")

# --- STRATEGY 2: RH-OCG (empirical) ---
print("[2/2] RH-OCG (empirical)...")
t0 = time.perf_counter()

replan  = 5
n_probe = 60
min_hor = 15

def ocg_solve_inner(z0, y_goal, N_rem, N_probe=60, eps=0.02):
    N_p    = min(N_probe, N_rem)
    z_nom_ = simulate(z0, np.zeros((N_p, m)))
    dy     = y_goal - C @ z_nom_[-1]
    C_o    = np.zeros((p, N_p*m))
    for tau in range(N_p):
        for j in range(m):
            up = np.zeros(m); up[j] =  eps
            un = np.zeros(m); un[j] = -eps
            zp = rk4(z_nom_[tau], up); zn = rk4(z_nom_[tau], un)
            for k in range(tau+1, N_p):
                zp = rk4(zp, np.zeros(m)); zn = rk4(zn, np.zeros(m))
            C_o[:, tau*m+j] = C @ (zp - zn) / (2*eps)
    W   = C_o @ C_o.T
    U_p = np.clip((C_o.T @ np.linalg.lstsq(W, dy, rcond=None)[0]).reshape(N_p, m),
                  min_ctrl, max_ctrl)
    U = np.vstack([U_p, np.zeros((N_rem-N_p, m))]) if N_p < N_rem else U_p
    return U

z = z_0.copy(); U_hist = []; U_plan = None; u_prev = np.zeros(m)

for k in range(T):
    N_rem = T - k
    if (U_plan is None or k % replan == 0) and N_rem >= min_hor:
        U_plan = ocg_solve_inner(z, y_f, N_rem, N_probe=n_probe)
    u_k = np.zeros(m) if (U_plan is None or len(U_plan) == 0) else U_plan[0]
    if U_plan is not None and len(U_plan) > 0: U_plan = U_plan[1:]
    u_k    = u_prev + np.clip(u_k - u_prev, -rate_clip, rate_clip)
    u_prev = u_k.copy(); z = rk4(z, u_k); U_hist.append(u_k.copy())

U_rh = np.array(U_hist); z_rh = simulate(z_0, U_rh)
t_rh = time.perf_counter() - t0
e_rh = [abs(z_rh[-1,i]*sc[i] - y_f[i]*sc[i]) for i in range(p)]
results['rh'] = dict(label='RH-OCG (Empirical)', z=z_rh, U=U_rh,
                     e=e_rh, t=t_rh, color='tab:red', ls='-')
print(f"   h={e_rh[0]:.4f}m  alpha={e_rh[2]:.3f}°  t={t_rh:.1f}s")

# Zero control baseline:
z_zc = simulate(z_0, np.zeros((T, m)))

# --- Summary ---
print(f"\n{'='*60}")
print(f"{'Strategy':<25} {'h err (m)':>10} {'α err (°)':>10} {'Time':>8}")
print('-'*60)
for k, r in results.items():
    print(f"  {r['label']:<23} {r['e'][0]:>10.4f} {r['e'][2]:>10.3f} {r['t']:>7.1f}s")
print('='*60)

# --- FIGURE 1: Trajectories ---
time_ax = np.arange(T+1)*dt
t_u     = np.arange(T)*dt
xfp     = [y_f[i]*sc[i] for i in range(p)]

fig1, axes = plt.subplots(3, 2, figsize=(12, 9), sharex='col')
fig1.subplots_adjust(hspace=0.08, wspace=0.28)

for i in range(p):
    ax = axes[i//2, i%2]
    ax.plot(time_ax, z_zc[:,i]*sc[i], '#cccccc', lw=1.0, ls=':', label='No control', zorder=1)
    for k, r in results.items():
        ax.plot(time_ax, r['z'][:,i]*sc[i],
                lw=2.0, color=r['color'], ls=r['ls'], label=r['label'], zorder=2)
    ax.axhline(xfp[i], color='k', ls=':', lw=1.2, label='Target', zorder=3)
    ax.set_ylabel(state_labs[i], fontsize=9)
    ax.grid(True, alpha=0.2); ax.tick_params(labelsize=8)
    if i == 0:
        ax.legend(fontsize=8, loc='best')

for i in range(m):
    ax = axes[2, i]
    for k, r in results.items():
        U = r['U']
        if len(U) < T: U = np.vstack([U, np.zeros((T-len(U), m))])
        ax.plot(t_u, np.rad2deg(U[:,i]),
                lw=2.0, color=r['color'], ls=r['ls'], label=r['label'])
    ax.set_ylabel(ctrl_labs[i], fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.grid(True, alpha=0.2); ax.tick_params(labelsize=8)

fig1.suptitle('DeePC vs RH-OCG (Empirical) — Trajectories & Controls',
              fontsize=11, fontweight='bold')

# --- FIGURE 2: Error and timing ---
labels = [r['label'] for r in results.values()]
cols   = [r['color'] for r in results.values()]
h_errs = [r['e'][0]  for r in results.values()]
a_errs = [r['e'][2]  for r in results.values()]
times  = [r['t']     for r in results.values()]
x = np.arange(len(results)); w = 0.32

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5))

ax = axes2[0]
ax.bar(x-w/2, h_errs, w, color=cols, alpha=0.9,       label='h error (m)')
ax.bar(x+w/2, a_errs, w, color=cols, alpha=0.5, hatch='//', label='α error (°)')
for xi, (h, a) in enumerate(zip(h_errs, a_errs)):
    ax.text(xi-w/2, h*1.05, f'{h:.4f}',   ha='center', va='bottom', fontsize=8, rotation=90)
    ax.text(xi+w/2, a*1.05, f'{a:.3f}°',  ha='center', va='bottom', fontsize=8, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Final Error', fontsize=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis='y')

ax = axes2[1]
bars = ax.bar(x, times, color=cols, alpha=0.9, width=0.4)
for xi, t in enumerate(times):
    ax.text(xi, t*1.05, f'{t:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Computation Time (s)', fontsize=10)
ax.grid(True, alpha=0.2, axis='y')

fig2.suptitle('DeePC vs RH-OCG — Accuracy vs Computation Time',
              fontsize=11, fontweight='bold')
fig2.tight_layout()
plt.show()