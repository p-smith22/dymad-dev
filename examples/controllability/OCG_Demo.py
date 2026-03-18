# Import packages:
from funcs import *

"""
Physical Example: Aeroelastic System with Uncontrollable Fluid States
----------------------------------------------------------------------
Consider a coupled rigid-body + fluid system where:
  - z = [h, hdot, alpha, alphadot, q1, q2, q3]  (7 states)
      h, hdot         : plunge displacement and velocity
      alpha, alphadot : pitch angle and rate  
      q1, q2, q3      : uncontrollable fluid/wake states
  - u = [delta_e, delta_a]  (2 inputs: elevator, aileron)
  - x = [h, alpha]  (2 observations: plunge and pitch only)
The fluid states evolve autonomously (no B columns) and C does not
observe them
"""

# --- DEFINE SYSTEM ---
# Define system matrices:
A = np.array([
    [ 0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # h
    [-2.0, -2.0, -1.2,  0.0,  0.5,  0.0,  0.0],  # hdot
    [ 0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0],  # alpha
    [ 0.8,  0.0, -3.0, -2.0,  0.0,  0.3, -0.2],  # alphadot
    [ 0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0],  # q1
    [ 0.0,  0.0,  0.0,  0.0,  0.0, -2.0,  0.0],  # q2
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -3.0],  # q3cd
])
B = np.array([
    [ 0.0,   0.0],  # h
    [-0.8,   0.1],  # hdot
    [ 0.0,   0.0],  # alpha
    [ 1.2,  -0.2],  # alphadot
    [ 0.0,   0.0],  # q1
    [ 0.0,   0.0],  # q2
    [ 0.0,   0.0],  # q3
])
C = np.array([
    [1, 0, 0, 0, 0, 0, 0],  # h
    [0, 1, 0, 0, 0, 0, 0],  # hdot
    [0, 0, 1, 0, 0, 0, 0],  # alpha
    [0, 0, 0, 1, 0, 0, 0],  # alphadot
])

# Unpack shapes:
n = A.shape[0]
m = B.shape[1]
p = C.shape[0]

# Define problem:
n_tsteps = 100
dt       = 0.05

# Discretize system:
A, B = discretize_system(A, B, dt)
C = C

# Define initial and final states:
z_0 = np.array([0.05, 0.0, 0.0, 0.0, 0.5, -0.3, 0.2])
x_f = np.array([0.05,  0, np.deg2rad(3), 0])

# --- CHECK LATENT SPACE CONTROLLABILITY ---
# Build Controllability Matrix and Gramian:
Ctrl, W = controllability(A, B, n_tsteps)

# Check controllability:
if np.linalg.matrix_rank(Ctrl) == n:
    print("Latent Space ... Controllable")
else:
    print("Latent Space ... NOT Controllable")

# --- CHECK OBSERVATION SPACE CONTROLLABILITY ---
# Define R:
R = np.diag([1, 1])

# Build Controllability Matrix and Gramian:
Ctrl_o, W_o = weighted_output_CG(A, B, C, R, n_tsteps)

# Check controllability:
if np.linalg.matrix_rank(Ctrl_o) == p:
    print("Observation Space ... Controllable")
else:
    print("Observation Space ... NOT Controllable")

# --- ZERO CONTROL (free response) ---
z_zc = z_0.copy()
x_traj_zc = [C @ z_zc]
for k in range(n_tsteps):
    z_zc = A @ z_zc
    x_traj_zc.append(C @ z_zc)
x_traj_zc = np.array(x_traj_zc)

# --- FUNCTION TO COMPUTE OCG TRAJECTORY ---
def compute_ocg_traj(A, B, C, R, n_tsteps, z_0, x_f):
    Ctrl_o, W_o = weighted_output_CG(A, B, C, R, n_tsteps)
    A_pow = np.linalg.matrix_power(A, n_tsteps)
    x_free = C @ (A_pow @ z_0)
    delta = x_f - x_free
    U_flat = Ctrl_o.T @ np.linalg.pinv(W_o, rcond=1e-12) @ delta
    U_star = U_flat.reshape(n_tsteps, m)
    z = z_0.copy()
    x_traj = [C @ z]
    for k in range(n_tsteps):
        z = A @ z + B @ U_star[k]
        x_traj.append(C @ z)
    return np.array(x_traj), U_star

# --- COMPUTE ALL CASES ---
R_I      = np.diag([1.0,  1.0])
R_ail    = np.diag([10.0, 1.0])
R_ele    = np.diag([1.0, 10.0])

x_traj_I,   U_I   = compute_ocg_traj(A, B, C, R_I,   n_tsteps, z_0, x_f)
x_traj_ail, U_ail = compute_ocg_traj(A, B, C, R_ail, n_tsteps, z_0, x_f)
x_traj_ele, U_ele = compute_ocg_traj(A, B, C, R_ele, n_tsteps, z_0, x_f)

# --- PLOT ---
labels    = ['Neutral', 'Favor Aileron', 'Favor Elevator', 'No control']
colors    = ['tab:blue', 'tab:orange', 'tab:green', 'gray']
linestyle = ['-', '-', '-', '-']
x_trajs   = [x_traj_I, x_traj_ail, x_traj_ele, x_traj_zc]
U_stars   = [U_I, U_ail, U_ele, None]

fig, axs = plt.subplots(2, 2, figsize=(14, 9))

for i, (xt, lab, col, ls) in enumerate(zip(x_trajs, labels, colors, linestyle)):
    axs[0, 0].plot(xt[:, 0], label=lab, color=col, linestyle=ls, linewidth=1.8)
    axs[0, 1].plot(np.rad2deg(xt[:, 2]), label=lab, color=col, linestyle=ls, linewidth=1.8)

for i, (U, lab, col, ls) in enumerate(zip(U_stars, labels, colors, linestyle)):
    if U is not None:
        axs[1, 0].plot(np.rad2deg(U[:, 0]), label=lab, color=col, linestyle=ls, linewidth=1.8)
        axs[1, 1].plot(np.rad2deg(U[:, 1]), label=lab, color=col, linestyle=ls, linewidth=1.8)

# Target lines
axs[0, 0].axhline(x_f[0],            color='k', linestyle='--', linewidth=1.2, label='target')
axs[0, 1].axhline(np.rad2deg(x_f[2]), color='k', linestyle='--', linewidth=1.2, label='target')

axs[0, 0].set_ylabel('h (m)');          axs[0, 0].set_title('Plunge Displacement')
axs[0, 1].set_ylabel('alpha (deg)');    axs[0, 1].set_title('Pitch Angle')
axs[1, 0].set_ylabel('Elevator (deg)'); axs[1, 0].set_title('Elevator Input')
axs[1, 1].set_ylabel('Aileron (deg)');  axs[1, 1].set_title('Aileron Input')

for ax in axs.flat:
    ax.grid(True)
    ax.legend(fontsize=8)
for ax in axs[1]:
    ax.set_xlabel('Time step')

plt.suptitle('OCG Optimal Control: Effect of Input Weighting R', fontsize=13)
plt.tight_layout()

# --- PRINT FINAL ERRORS ---
print(f"\n{'Case':<30} {'h error':>12}       {'alpha error (deg)':>18}")
print("-" * 68)
for xt, lab in zip([x_traj_I, x_traj_ail, x_traj_ele, x_traj_zc], labels):
    h_err = abs(xt[-1, 0] - x_f[0])
    a_err = abs(np.rad2deg(xt[-1, 2]) - np.rad2deg(x_f[2]))
    print(f"{lab:<30} {h_err:>12.6f} {a_err:>18.6f}")

plt.show()
