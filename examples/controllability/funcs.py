import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy as sp
from dymad.io import DynData
from dymad.models.prediction import _prepare_data
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
import cvxpy as cp

def single_step(model, z, u=None):

    """
    Discrete single-step dynamics in latent space for linearization

    INPUTS
    model: dymad.models
        DyMAD trained physics model
    z: torch.Tensor
        (latent_dim,)
        Current latent state vector
    u: torch.Tensor or None
        (m,)
        Current control input (optional)

    OUTPUTS
    z_next: torch.Tensor
        (latent_dim,)
        Next latent state vector
    """

    # Create container:
    w = DynData()

    # Add control:
    u = u.view(1, -1)
    w_next = w.get_step(0).set_u(u)

    # Take step in latent space:
    z_next = model.dynamics(z.view(1, -1), w_next)

    # Return next latent state:
    return z_next.view(-1)

def linearize(model, x_ref, u_ref=None):

    """
    Linearize model to get system matrices (using autograd)

    INPUTS
    model: dymad.models
        DyMAD trained physics model
    x_ref: torch.Tensor or numpy.ndarray
        (#, n)
        Reference state vector
    u_ref (Optional): torch.Tensor or numpy.ndarray
        (#, m)
        Reference control input (optional, will use zeros if model requires control)

    OUTPUTS
    A: torch.Tensor or numpy.ndarray
        (#, latent_dim, latent_dim)
        System matrix for latent states
    B (Optional): torch.Tensor or numpy.ndarray
        (#, latent_dim, m)
        System matrix for controls (None if autonomous system)
    C: torch.Tensor or numpy.ndarray
        (#, n, latent_dim)
        Observation matrix (latent to observation space)
    """

    # Transfer reference arrays to torch for autograd:
    is_numpy = False
    if isinstance(x_ref, np.ndarray):
        x_ref = torch.from_numpy(x_ref)
        is_numpy = True
    has_control = u_ref is not None
    if has_control and isinstance(u_ref, np.ndarray):
        u_ref = torch.from_numpy(u_ref)

    # Reshape vectors:
    if x_ref.ndim == 1:
        x_ref = x_ref.unsqueeze(0)
    if has_control and u_ref.ndim == 1:
        u_ref = u_ref.unsqueeze(0)

    # Extract dimensions:
    n_points, n = x_ref.shape

    # Extract data types:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Assign identical data types to avoid mismatch:
    x_ref = x_ref.to(device=device, dtype=dtype)

    # Determine control dimension (if any) and setup:
    if has_control:
        m = u_ref.shape[1]
        n_points_u = u_ref.shape[0]

        # Ensure each reference has the same number of reference points:
        if n_points_u == 1 and n_points > 1:
            u_ref = u_ref.expand(n_points, m)
        elif n_points_u < n_points:
            u_ref = u_ref.expand(n_points, m)
        elif n_points_u > n_points:
            x_ref = x_ref.expand(n_points_u, n)
            n_points = x_ref.shape[0]
        u_ref = u_ref.to(device=device, dtype=dtype)

    # If no controls were provided:
    else:

        # Fetch the number of control features from training:
        if hasattr(model, 'n_total_control_features'):
            ctrl_dim = model.n_total_control_features

        # Raise an error if none are found (probably indicative of a larger error):
        else:
            raise "No control features provided, could not find."

        # Create zero control tensor so we can continue:
        u_ref = torch.zeros(n_points, ctrl_dim, dtype=dtype, device=device)
        has_control = True

    # Initialize linearized matrices lists:
    A_list, B_list, C_list = [], [], []

    # Loop through each of the samples:
    for i in range(n_points):

        # Assign current reference points:
        x_i = x_ref[i].clone().detach()
        u_i = u_ref[i].clone().detach().requires_grad_(True) if has_control else None

        # Create container:
        w = DynData()

        # Add control if specified:
        if has_control:
            w_step = w.get_step(0).set_x(x_i.view(1, -1)).set_u(u_i.view(1, -1))

        # Do not add control if not specified:
        else:
            w_step = w.get_step(0).set_x(x_i.view(1, -1))

        # Encode to latent space:
        z_i = model.encoder(w_step).view(-1).clone().detach().requires_grad_(True)

        # Define forward step in latent space for A matrix:
        def f_z(z):

            # Call single step function:
            return single_step(model, z, u_i)

        # Define decoder mapping for C matrix:
        def f_c(z):

            # If control is needed:
            if has_control:
                w_tmp = w.get_step(0).set_u(u_i.view(1, -1))

            # If control is not needed:
            else:
                w_tmp = w.get_step(0)

            # Return decoder:
            return model.decoder(z.view(1, -1), w_tmp).view(-1)

        # Get Jacobians from autograd:
        A_i = torch.autograd.functional.jacobian(f_z, z_i)  # dz_next/dz
        C_i = torch.autograd.functional.jacobian(f_c, z_i)  # dx/dz

        # Append to reference points list:
        A_list.append(A_i)
        C_list.append(C_i)

        # Compute B matrix only if control is present:
        if has_control:

            # Helper function to call single step:
            def f_u(u):
                return single_step(model, z_i, u)

            # Use autograd to compute jacobian:
            B_i = torch.autograd.functional.jacobian(f_u, u_i)

            # Append to list:
            B_list.append(B_i)

    # Stack for correct shape:
    A = torch.stack(A_list)
    C = torch.stack(C_list)
    B = torch.stack(B_list) if has_control else None

    # Switch back to numpy if desired:
    if is_numpy:
        A = A.detach().cpu().numpy()
        C = C.detach().cpu().numpy()
        if B is not None:
            B = B.detach().cpu().numpy()

    # Return stack of linearized matrices:
    return A, B, C

def controllability(_A, _B, n_tsteps):

    """
    Compute controllability matrix using the system matrices of the model.
    Solves Lyapunov equation for optimal solution if system is stable. If not, calculate best solution over the
    number of defined time steps.

    INPUTS
    _A: np.ndarray or Torch.Tensor
        (n, n)
        System matrices
    _B: np.ndarray or Torch.Tensor
        (n, m)
        System matrices
    n_tsteps: int
        N/A
        Number of time steps in desired trajectory

    OUTPUTS
    _Cmat: np.ndarray or Torch.Tensor
        (n, m * n_tsteps)
        Controllability matrix
    _W: np.ndarray or Torch.Tensor
        (n, n)
        Controllability Gramian
    """

    # Decide whether torch or numpy:
    is_torch = torch.is_tensor(_A)

    # Torch sequence:
    if is_torch:

        # Transfer B to torch:
        device = _A.device
        dtype = _A.dtype
        _B = torch.as_tensor(_B, device=device, dtype=dtype)

        # Extract dimensions:
        n = _A.shape[0]
        m = _B.shape[1]

        # Calculate controllability matrix:
        _Ctrl = torch.zeros((n, n_tsteps * m), dtype=_A.dtype, device=_A.device)
        for i in range(n_tsteps):
            _Ctrl[:, i * m:(i + 1) * m] = torch.linalg.matrix_power(_A, n_tsteps-1-i) @ _B

        # Extract eigenvalues from A:
        eigvals = torch.linalg.eigvals(_A)

        # If A is stable, use discrete Lyapunov equation to algebraically solve infinite sum:
        if torch.all(torch.abs(eigvals) < 1):
            _A_np = _A.detach().cpu().numpy()
            _B_np = _B.detach().cpu().numpy()
            _W_np = sp.linalg.solve_discrete_lyapunov(_A_np, _B_np @ _B_np.T)
            _W = torch.tensor(_W_np, dtype=_A.dtype, device=_A.device)

        # If unstable, solve infinite sum up to the nth time step:
        elif n_tsteps is not None:
            _W = torch.zeros((n, n), dtype=_A.dtype, device=_A.device)
            for i in range(n_tsteps):
                _Ak = torch.linalg.matrix_power(_A, i)
                _W += _Ak @ _B @ _B.T @ _Ak.T

        # If no time step is specified, just return none:
        else:
            _W = None

        # Return controllability matrix and gramian:
        return _Ctrl, _W

    # Numpy sequence:
    else:

        # Transfer B to numpy:
        _B = np.asarray(_B)

        # Extract dimensions:
        n = _A.shape[0]
        m = _B.shape[1]

        # Solve for controllability matrix:
        _Ctrl = np.zeros((n, n_tsteps * m))
        for i in range(n_tsteps):
            _Ctrl[:, i*m:(i+1)*m] = np.linalg.matrix_power(_A, n_tsteps-1-i) @ _B

        # Calculate eigenvalues of A:
        eigvals = np.linalg.eigvals(_A)

        # If A is stable, solve discrete Lyapnuov equation to get algebraic solution for infinite sum:
        if np.all(np.abs(eigvals) < 1):
            _W = sp.linalg.solve_discrete_lyapunov(_A, _B @ _B.T)

        # If A is unstable, solve infinite sum up to the nth time step:
        elif n_tsteps is not None:
            n = _A.shape[0]
            _W = np.zeros((n, n))
            for i in range(n_tsteps):
                _Ak = np.linalg.matrix_power(_A, i)
                _W += _Ak @ _B @ _B.T @ _Ak.T

        # If no number of time steps are specified, just return none:
        else:
            _W = None

        # Return controllability matrix and gramian:
        return _Ctrl, _W

def output_CG(W_c, C, _Ctrl=None):

    """
    Calculate the Output Controllability Matrix/Gramian based on the Latent Space Controllability Matrix/Gramian and
    the observation matrix.

    INPUTS
    _Ctrl: np.ndarray or Torch.Tensor
        (n_z, m * n_tsteps)
        Controllability matrix
    W_c: np.ndarray or Torch.Tensor
        (n_z, n_z)
        Latent Space Controllability Gramian
    C: np.ndarray or Torch.Tensor
        (n_x, n_z)
        Observation matrix

    OUTPUTS
    _C: np.ndarray or Torch.Tensor
        (n_x, m * n_tsteps)
        Output Controllability Matrix
    W_co: np.ndarray or Torch.Tensor
        (n_x, n_x)
        Output Controllability Gramian
    """

    # Calculate Output Controllability Matrix:
    if _Ctrl is not None:
        _C = C@_Ctrl
    else:
        _C = None

    # Calculate Output Controllability Gramian:
    W_co = C @ W_c @ C.T

    # Return OCM and OCG:
    return _C, W_co

def prop_dyn(A, B, x_0, u, n_tsteps, x_ref=None, u_ref=None):

    """
    Propagate any given discrete linear system through time. Takes into consideration the reference conditions at which
    the matrices are linearized about, which defaults to zero if none are given.

    INPUTS
    A: np.ndarray or Torch.Tensor
        (n, n)
        System matrix
    B: np.ndarray or Torch.Tensor
        (n, m)
        System matrix
    x_0: np.ndarray or Torch.Tensor
        (n,)
        Initial state condition
    u: np.ndarray or Torch.Tensor
        (n_tsteps, m)
        Control sequence
    n_tsteps: int
        N/A
        Number of time steps in the trajectory
    x_ref_seq (Optional): np.ndarray or Torch.Tensor
        (n,)
        Reference states
    u_ref_seq (Optional): np.ndarray or Torch.Tensor
        (m,)
        Reference controls

    OUTPUTS
    x: np.ndarray or Torch.Tensor
        (n, n_tsteps)
        Propagated states of length n_tsteps
    """

    # Determine if user wants Torch or Numpy:
    is_torch = torch.is_tensor(A)

    # Assign everything as torch:
    if is_torch:

        # Capture data type:
        device = A.device
        dtype = A.dtype

        # Convert to tensors:
        A = torch.as_tensor(A, device=device, dtype=dtype)
        B = torch.as_tensor(B, device=device, dtype=dtype)
        u = torch.as_tensor(u, device=device, dtype=dtype)

        # Default reference:
        if x_ref is None:
            x_ref = torch.zeros(A.shape[0], device=device, dtype=dtype)
        else:
            x_ref = torch.as_tensor(x_ref, device=device, dtype=dtype)

        if u_ref is None:
            u_ref = torch.zeros(u.shape[1], device=device, dtype=dtype)
        else:
            u_ref = torch.as_tensor(u_ref, device=device, dtype=dtype)

        # Initialize states:
        x = torch.zeros((A.shape[0], n_tsteps), device=device, dtype=dtype)

    else:

        # Set as numpy arrays:
        A = np.asarray(A)
        B = np.asarray(B)
        u = np.asarray(u)

        # Default references:
        if x_ref is None:
            x_ref = np.zeros(A.shape[0])
        else:
            x_ref = np.asarray(x_ref)

        if u_ref is None:
            u_ref = np.zeros(u.shape[1])
        else:
            u_ref = np.asarray(u_ref)

        # Initialize states:
        x = np.zeros((A.shape[0], n_tsteps))

    # initial condition:
    x[:, 0] = x_0

    # Propagate dynamics:
    for k in range(n_tsteps - 1):
        delta_x = x[:, k] - x_ref
        delta_u = u[k] - u_ref
        x[:, k + 1] = x_ref + A @ delta_x + B @ delta_u

    # Return state:
    return x


def plot_traj(x, u, n_tsteps, x_f=None, lab=None, line=None):
    """
    Plot states and controls for any given number of trajectories

    If you want several trajectories at once, stack on the third dimension. For instance, to plot two trajectories
    with 3 state variables and 100 time steps, the shape should be: (3, 100, *2*)

    INPUTS
    x: np.ndarray or Torch.Tensor
        (n, n_tsteps, #) or (n, n_tsteps)
        State values throughout the trajectory
    u: np.ndarray or Torch.Tensor
        (n_tsteps, m, #) or (n_tsteps, m)
        Control sequence
    n_tsteps: int
        N/A
        Number of time steps in the trajectory
    x_f (Optional): np.ndarray or Torch.Tensor
        (n,)
        Desired final state
    lab (Optional): List of strings
        (#,) --> In the above example, (2,)
        List of labels for plotting
    line (Optional): List of strings
        (#,) --> In the above example, (2,)
        List of linestyles for plotting

    OUTPUTS
    None
    """

    # Add dummy u if none:
    u_exists = True
    if u is None:
        u_exists = False
        u = np.zeros((n_tsteps, 1))

    # Transfer everything to numpy (easier to work with for matplotlib):
    x = np.asarray(x)
    u = np.asarray(u)
    if x_f is not None:
        x_f = np.asarray(x_f)

    # Extract dimensions:
    n = x.shape[0]
    m = u.shape[1] if u.ndim >= 2 else 1

    # Reshape for plotting:
    if x.ndim == 2:
        x = x.reshape(n, n_tsteps, 1)  # Add trajectory dimension
    else:
        x = x.reshape(n, n_tsteps, -1)

    if u.ndim == 1:
        u = u.reshape(n_tsteps, 1, 1)  # Add control and trajectory dimensions
    elif u.ndim == 2:
        u = u.reshape(n_tsteps, m, 1)  # Add trajectory dimension
    else:
        u = u.reshape(n_tsteps, m, -1)

    # If there are no labels specified, just fill with placeholders:
    if lab is None:
        lab = [''] * x.shape[-1]

    # If there are no linestyles specified, fill with regular lines:
    if line is None:
        line = ['-'] * x.shape[-1]

    # Initialize plotting:
    if u_exists:
        fig, axes = plt.subplots(n + m, 1, figsize=(8, 6), sharex=True)
    else:
        fig, axes = plt.subplots(n, 1, figsize=(8, 6), sharex=True)

    # Make axes iterable if only one subplot:
    if not hasattr(axes, '__iter__'):
        axes = [axes]

    # Plot states:
    for j in range(x.shape[-1]):
        for i in range(n):
            axes[i].plot(x[i, :, j], label=lab[j], linestyle=line[j], linewidth=2)
            axes[i].set_ylabel(f'State {i + 1}')
            axes[i].grid(True)
            axes[i].set_ylim(np.min(x[i, :, :]) - 1, np.max(x[i, :, :]) + 1)

            # Plot desired final state if specified:
            if x_f is not None:
                if j == 0:
                    axes[i].axhline(x_f[i], color='r', linestyle='--', label='Desired')
                else:
                    axes[i].axhline(x_f[i], color='r', linestyle='--')

    # Plot controls:
    if u_exists:
        for i in range(m):
            for j in range(u.shape[-1]):
                axes[n + i].step(range(n_tsteps), u[:, i, j],
                                 label=lab[j] if i == 0 else "",
                                 linestyle=line[j], linewidth=2)
            axes[n + i].set_ylabel(f'Control {i + 1}')
            axes[n + i].set_xlabel('Time step')
            axes[n + i].grid(True)
            axes[n + i].set_ylim(np.min(u[:, i, :]) - 1, np.max(u[:, i, :]) + 1)

    # Set x labels for sharex axis:
    axes[-1].set_xlabel('Time step')
    axes[-1].set_xlim([0, n_tsteps])

    # Apply legend if user specified labels for trajectories:
    if lab is not None and any(lab):
        axes[0].legend()

    # Tight layout:
    plt.tight_layout()


def optimal_ctrl_ocg(_A, _B, _C, model, _Ctrl_o, _W_co, n_tsteps, x_0, x_f, plot_graph):

    """
    Calculate the optimal control using the Output Controllability Gramian (OCG).
    Operates directly in observation space without requiring full latent controllability.

    INPUTS
    _A: np.ndarray or Torch.Tensor
        (n_z, n_z)
        System matrices (latent space)
    _B: np.ndarray or Torch.Tensor
        (n_z, m)
        System matrices (latent space)
    _C: np.ndarray or Torch.Tensor
        (n_x, n_z)
        Observation matrix
    model: dymad.models
        Trained model for encoding/decoding
    _Ctrl_o: np.ndarray or Torch.Tensor
        (n_x, m * n_tsteps)
        Output Controllability Matrix (C @ Ctrl)
    _W_co: np.ndarray or Torch.Tensor
        (n_x, n_x)
        Output Controllability Gramian (C @ W_c @ C^T)
    n_tsteps: int
        Number of desired time steps
    x_0: np.ndarray or Torch.Tensor
        (n_x,)
        Initial state (observation space)
    x_f: np.ndarray or Torch.Tensor
        (n_x,)
        Final desired state (observation space)
    plot_graph: bool
        Option to automatically plot graphs

    OUTPUTS
    x: np.ndarray or Torch.Tensor
        (n_steps+1, n_x)
        Optimal trajectory in observation space
    _U: np.ndarray or Torch.Tensor
        (n_tsteps, m)
        Optimal control sequence
    """

    # Decide if user wants torch or numpy:
    is_torch = torch.is_tensor(_A)

    # Torch sequence:
    if is_torch:

        # Transfer everything to torch:
        device = _A.device
        dtype = _A.dtype
        _B = torch.as_tensor(_B, device=device, dtype=dtype)
        _C = torch.as_tensor(_C, device=device, dtype=dtype)
        _Ctrl_o = torch.as_tensor(_Ctrl_o, device=device, dtype=dtype)
        _W_co = torch.as_tensor(_W_co, device=device, dtype=dtype)
        x_0 = torch.as_tensor(x_0, device=device, dtype=dtype)
        x_f = torch.as_tensor(x_f, device=device, dtype=dtype)

        # Check output controllability (rank of W_co must equal n_x = p):
        if torch.linalg.matrix_rank(_W_co) != _W_co.shape[0]:
            raise ValueError("Cannot compute optimal control (Output Gramian singular: system not output controllable)")

        # Encode initial state to latent space, apply A^N, decode to get free response:
        w0 = DynData().get_step(0).set_x(x_0.view(1, -1))
        z_0 = model.encoder(w0).view(-1)
        _Apow = torch.linalg.matrix_power(_A, n_tsteps)
        x_free = _C @ (_Apow @ z_0)  # free response in observation space

        # Observation gap:
        _delta = x_f - x_free

        # Optimal control via OCG pseudoinverse (U* = C_o^T W_co^{-1} delta):
        _U = _Ctrl_o.T @ torch.linalg.pinv(_W_co, rcond=1e-12) @ _delta

    # Numpy sequence:
    else:

        # Transfer everything to numpy:
        _B = np.asarray(_B)
        _C = np.asarray(_C)
        _Ctrl_o = np.asarray(_Ctrl_o)
        _W_co = np.asarray(_W_co)
        x_0 = np.asarray(x_0)
        x_f = np.asarray(x_f)

        # Check output controllability (rank of W_co must equal n_x = p):
        if np.linalg.matrix_rank(_W_co) != _W_co.shape[0]:
            raise ValueError("Cannot compute optimal control (Output Gramian singular: system not output controllable)")

        # Encode initial state to latent space, apply A^N, decode to get free response:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        x_0_torch = torch.from_numpy(x_0).to(device=device, dtype=dtype)
        w0 = DynData().get_step(0).set_x(x_0_torch.view(1, -1))
        z_0 = model.encoder(w0).view(-1).detach().cpu().numpy()

        _Apow = np.linalg.matrix_power(_A, n_tsteps)
        x_free = _C @ (_Apow @ z_0)  # free response in observation space

        # Observation gap:
        _delta = x_f - x_free

        # Optimal control via OCG pseudoinverse (U* = C_o^T W_co^{-1} delta):
        _U = _Ctrl_o.T @ np.linalg.pinv(_W_co, rcond=1e-12) @ _delta

    # Reshape:
    m = _B.shape[1]
    _U = _U.reshape(n_tsteps, m)

    # Propagate dynamics in latent space and decode to observation space:
    x, _ = prop_dyn_latent(model, _A, _B, _C, x_0, _U, use_C=False)

    if plot_graph:
        plot_traj(x.T, _U, n_tsteps, x_f)

    # Return states and controls:
    return x, _U

def optimal_ctrl(_A, _B, _C, model, _Ctrl, _W, n_tsteps, x_0, x_f, plot_graph):

    """
    Calculate the optimal control given the controllability evaluation
    Now works in latent space and decodes to observation space

    INPUTS
    _A: np.ndarray or Torch.Tensor
        (n_z, n_z)
        System matrices (latent space)
    _B: np.ndarray or Torch.Tensor
        (n_z, m)
        System matrices (latent space)
    _C: np.ndarray or Torch.Tensor
        (n_x, n_z)
        Observation matrix
    model: dymad.models
        Trained model for encoding/decoding
    _Ctrl: np.ndarray or Torch.Tensor
        (n_z, m * n_tsteps)
        Controllability Matrix
    _W: np.ndarray or Torch.Tensor
        (n_z, n_z)
        Controllability Gramian
    n_tsteps: int
        Number of desired time steps
    x_0: np.ndarray or Torch.Tensor
        (n_x,)
        Initial state (observation space)
    x_f: np.ndarray or Torch.Tensor
        (n_x,)
        Final desired state (observation space)
    plot_graph: bool
        Option to automatically plot graphs

    OUTPUTS
    x: np.ndarray or Torch.Tensor
        (n_steps+1, n_x)
        Optimal trajectory in observation space
    _U: np.ndarray or Torch.Tensor
        (n_tsteps, m)
        Optimal control sequence
    """

    # Decide if user wants torch or numpy:
    is_torch = torch.is_tensor(_A)

    # Torch sequence:
    if is_torch:

        # Transfer everything to torch:
        device = _A.device
        dtype = _A.dtype
        _B = torch.as_tensor(_B, device=device, dtype=dtype)
        _C = torch.as_tensor(_C, device=device, dtype=dtype)
        _Ctrl = torch.as_tensor(_Ctrl, device=device, dtype=dtype)
        _W = torch.as_tensor(_W, device=device, dtype=dtype)
        x_0 = torch.as_tensor(x_0, device=device, dtype=dtype)
        x_f = torch.as_tensor(x_f, device=device, dtype=dtype)

        # Ensure that Gramian is not singular:
        if torch.linalg.matrix_rank(_W) != _A.shape[0]:
            raise ValueError("Cannot compute optimal control (Gramian singular)")

        # Encode initial and final states to latent space:
        w0 = DynData().get_step(0).set_x(x_0.view(1, -1))
        z_0 = model.encoder(w0).view(-1)

        wf = DynData().get_step(0).set_x(x_f.view(1, -1))
        z_f = model.encoder(wf).view(-1)

        # Solve for optimal control solution in latent space:
        _Apow = torch.linalg.matrix_power(_A, n_tsteps)
        _RHS = z_f - _Apow @ z_0
        _U = _Ctrl.T @ torch.linalg.pinv(_W, rcond=1e-12) @ _RHS

    # Numpy sequence:
    else:

        # Transfer everything to numpy:
        _B = np.asarray(_B)
        _C = np.asarray(_C)
        _Ctrl = np.asarray(_Ctrl)
        _W = np.asarray(_W)
        x_0 = np.asarray(x_0)
        x_f = np.asarray(x_f)

        # Ensure gramian is non-singular
        if np.linalg.matrix_rank(_W) != _A.shape[0]:
            raise ValueError("Cannot compute optimal control (Gramian singular)")

        # Encode initial and final states to latent space:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        x_0_torch = torch.from_numpy(x_0).to(device=device, dtype=dtype)
        w0 = DynData().get_step(0).set_x(x_0_torch.view(1, -1))
        z_0 = model.encoder(w0).view(-1).detach().cpu().numpy()

        x_f_torch = torch.from_numpy(x_f).to(device=device, dtype=dtype)
        wf = DynData().get_step(0).set_x(x_f_torch.view(1, -1))
        z_f = model.encoder(wf).view(-1).detach().cpu().numpy()

        # Solve for optimal control sequence in latent space:
        _Apow = np.linalg.matrix_power(_A, n_tsteps)
        _RHS = z_f - _Apow @ z_0
        _U = _Ctrl.T @ np.linalg.pinv(_W, rcond=1e-12) @ _RHS

    # Reshape:
    m = _B.shape[1]
    _U = _U.reshape(n_tsteps, m)

    # Propagate dynamics in latent space and decode to observation space:
    x, _ = prop_dyn_latent(model, _A, _B, _C, x_0, _U, use_C=False)

    if plot_graph:
        plot_traj(x.T, _U, n_tsteps, x_f)

    # Return states and controls:
    return x, _U

def discretize_system(_Acts, _Bcts, dt):

    """
    Discretize a linear system

    INPUTS
    _Acts: np.ndarray or Torch.Tensor
        (n, n)
        Continuous system matrix
    _Bcts: np.ndarray or Torch.Tensor
        (n, m)
        Continuous system matrix
    dt: float
        N/A
        Time step

    OUTPUTS
    _A: np.ndarray or Torch.Tensor
        (n, n)
        Discretized system matrix
    _B: np.ndarray or Torch.Tensor
        (n, m)
        Discretized system matrix
    """

    # Decide if the user wants tensor or numpy:
    is_torch = torch.is_tensor(_Acts)

    # Extract dimensions:
    n = _Acts.shape[0]
    m = _Bcts.shape[1]

    # Torch sequence:
    if is_torch:

        # Transfer B to torch:
        device = _Acts.device
        dtype = _Acts.dtype
        _Bcts = torch.as_tensor(_Bcts, device=device, dtype=dtype)

        # Construct augmented matrix:
        _M = torch.zeros((n + m, n + m), device=device, dtype=dtype)
        _M[:n, :n] = _Acts * dt
        _M[:n, n:] = _Bcts * dt

        # Solve matrix exponential:
        exp_M = torch.matrix_exp(_M)

        # Extract matrices:
        _A = exp_M[:n, :n]
        _B = exp_M[:n, n:]

    # Numpy sequence:
    else:

        # Transfer B to numpy:
        _Bcts = np.asarray(_Bcts)

        # Construct augmented matrix:
        _M = np.zeros((n + m, n + m))
        _M[:n, :n] = _Acts * dt
        _M[:n, n:] = _Bcts * dt

        # Solve matrix exponential:
        exp_M = sp.linalg.expm(_M)

        # Extract matrices:
        _A = exp_M[:n, :n]
        _B = exp_M[:n, n:]

    # Return discretized A and B matrices:
    return _A, _B

def prop_dyn_latent(model, A, B, C, x0, u_seq=None, use_C=False, n_tsteps=None, x_ref=None):

    """
    Propagate dynamics in latent space and decode to real space

    INPUTS
    model: dymad.models
        N/A
        Trained DyMAD model with encoder/decoder
    A: np.ndarray or torch.Tensor
        (n_z, n_z)
        Latent space dynamics matrix (DISCRETE-TIME)
    B (Optional): np.ndarray or torch.Tensor
        (n_z, n_u)
        Latent space control matrix
    C: np.ndarray or torch.Tensor
        (n_x, n_z)
        Observation matrix (latent to real space)
    x0: np.ndarray or torch.Tensor
        (n_x,)
        Initial condition in real/observation space
    u_seq (Optional): np.ndarray or torch.Tensor
        (n_steps, n_u)
        Control input sequence
    use_C: bool
        N/A
        If True, use C matrix in deviation form: x = x_ref + C (z - z_ref)
        If False, use model.decoder(z) (nonlinear decoding)
    n_tsteps: int (Optional)
        N/A
        Number of time steps to propagate (required if u_seq is None)
    x_ref: np.ndarray or torch.Tensor (Optional)
        (n_x,)
        Reference state for linearization (defaults to zero if None)

    OUTPUTS
    x_traj: np.ndarray or torch.Tensor
        (n_steps+1, n_x)
        State trajectory in real/observation space
    z_traj: np.ndarray or torch.Tensor
        (n_steps+1, n_z)
        State trajectory in latent space
    """

    # Decide numpy vs torch output:
    is_numpy = isinstance(A, np.ndarray) or isinstance(C, np.ndarray) or isinstance(x0, np.ndarray)

    # Extract device and dtype:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Convert inputs to Torch:
    x0_torch = torch.from_numpy(x0).to(device=device, dtype=dtype) if isinstance(x0, np.ndarray) \
        else x0.to(device=device, dtype=dtype)
    A_torch = torch.from_numpy(A).to(device=device, dtype=dtype) if isinstance(A, np.ndarray) \
        else A.to(device=device, dtype=dtype)
    C_torch = torch.from_numpy(C).to(device=device, dtype=dtype) if isinstance(C, np.ndarray) \
        else C.to(device=device, dtype=dtype)

    # Reference state in observation space:
    if x_ref is None:
        x_ref_torch = torch.zeros_like(x0_torch)
    else:
        x_ref_torch = torch.from_numpy(x_ref).to(device=device, dtype=dtype) if isinstance(x_ref, np.ndarray) \
            else x_ref.to(device=device, dtype=dtype)

    # Time and control handling:
    if u_seq is None and n_tsteps is not None:
        n_steps = n_tsteps
        u_seq = np.zeros((n_steps, 1))
        u_seq_torch = torch.from_numpy(u_seq).to(device=device, dtype=dtype)
        if B is None:
            n_z = A_torch.shape[0]
            B_torch = torch.zeros((n_z, 1), device=device, dtype=dtype)
        else:
            B_torch = torch.from_numpy(B).to(device=device, dtype=dtype) if isinstance(B, np.ndarray) \
                else B.to(device=device, dtype=dtype)
    elif n_tsteps is None and u_seq is not None:
        u_seq_torch = torch.from_numpy(u_seq).to(device=device, dtype=dtype) if isinstance(u_seq, np.ndarray) \
            else u_seq.to(device=device, dtype=dtype)
        n_steps = u_seq_torch.shape[0]
        if B is None:
            raise ValueError("B must be provided when u_seq is provided")
        B_torch = torch.from_numpy(B).to(device=device, dtype=dtype) if isinstance(B, np.ndarray) \
            else B.to(device=device, dtype=dtype)
    else:
        raise ValueError("Must provide either u_seq or n_tsteps (but not both None or both set)")

    # Encode reference to latent space:
    x_ref_torch = x_ref_torch.view(1, -1)
    w_ref = DynData().get_step(0).set_x(x_ref_torch)
    z_ref = model.encoder(w_ref).view(-1)

    # Encode initial state to latent space:
    x0_torch = x0_torch.view(1, -1)
    w0 = DynData().get_step(0).set_x(x0_torch)
    z0 = model.encoder(w0).view(-1)

    # Deviation in latent space:
    dz_current = z0 - z_ref

    # Initialize trajectories:
    z_traj = [z0]
    x_traj = []

    # Propagate dynamics:
    for k in range(n_steps + 1):

        # Decode using observation matrix::
        if use_C:

            # Simple matrix multiplication:
            x_ref_flat = x_ref_torch.view(-1)
            x_current = x_ref_flat + C_torch @ dz_current

        # Nonlinear decoder from trained model:
        else:

            # Use controls up until last timestep:
            if k < n_steps:
                u_current = u_seq_torch[k].view(1, -1)
                w_current = DynData().get_step(0).set_u(u_current)
            else:
                w_current = DynData().get_step(0)

            # Decode:
            x_current = model.decoder((z_ref + dz_current).view(1, -1), w_current).view(-1)

        # Append state:
        x_traj.append(x_current)

        # Propagate latent deviation:
        if k < n_steps:
            u_k = u_seq_torch[k]
            dz_next = A_torch @ dz_current + B_torch @ u_k
            dz_current = dz_next
            z_traj.append(z_ref + dz_current)

    # Convert back to numpy if needed:
    x_traj = torch.stack(x_traj)
    z_traj = torch.stack(z_traj)
    if is_numpy:
        x_traj = x_traj.detach().cpu().numpy()
        z_traj = z_traj.detach().cpu().numpy()

    # Return trajectories:
    return x_traj, z_traj

def riccati_opt(_A, _B, _C, model, x_0, x_f, n_tsteps, _Q, _R):

    """
    Solves Riccati equation for optimal control as a QP with HARD terminal constraint
    Now works in latent space with observation space constraints

    Formulation (in latent space):
        min     sum_{k=0}^{N-1} [z_k^T Q z_k + u_k^T R u_k] + z_N^T P_f z_N
        s.t.    z_{k+1} = A z_k + B u_k,  k=0,...,N-1
                z_0 = encoder(x_0) (given)
                z_N = encoder(x_f) (HARD CONSTRAINT)

    INPUTS
    _A: np.ndarray
        (n_z, n_z)
        Discretized system matrix (latent space)
    _B: np.ndarray
        (n_z, m)
        Discretized control matrix (latent space)
    _C: np.ndarray
        (n_x, n_z)
        Observation matrix
    model: dymad.models
        Trained model for encoding/decoding
    x_0: np.ndarray
        (n_x,)
        Initial state (observation space)
    x_f: np.ndarray
        (n_x,)
        Final desired state (observation space - HARD CONSTRAINT)
    n_tsteps: int
        Number of time steps
    _Q: np.ndarray
        (n_z, n_z)
        State weight (latent space)
    _R: np.ndarray
        (m, m)
        Control weight

    OUTPUTS
    x_opt: np.ndarray
        (n_steps+1, n_x) - Optimal trajectory in observation space
    u_opt: np.ndarray
        (m, n_tsteps) - Optimal control sequence
    """

    # Convert to numpy:
    A = np.asarray(_A)
    B = np.asarray(_B)
    C = np.asarray(_C)
    x0 = np.asarray(x_0)
    xf = np.asarray(x_f)
    Q = np.asarray(_Q)
    R = np.asarray(_R)

    # Encode initial and final states to latent space:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    x0_torch = torch.from_numpy(x0).to(device=device, dtype=dtype)
    w0 = DynData().get_step(0).set_x(x0_torch.view(1, -1))
    z0 = model.encoder(w0).view(-1).detach().cpu().numpy()

    xf_torch = torch.from_numpy(xf).to(device=device, dtype=dtype)
    wf = DynData().get_step(0).set_x(xf_torch.view(1, -1))
    zf = model.encoder(wf).view(-1).detach().cpu().numpy()

    # Extract dimensions:
    n_z = A.shape[0]
    m = B.shape[1]

    # Define optimization variables (in latent space):
    Z = cp.Variable((n_z, n_tsteps + 1))
    U = cp.Variable((m, n_tsteps))

    # Build objective function:
    cost = 0
    for k in range(n_tsteps):
        cost += cp.quad_form(Z[:, k], Q)
        cost += cp.quad_form(U[:, k], R)

    # Build constraints:
    constraints = []

    # Initial condition (latent space):
    constraints.append(Z[:, 0] == z0)

    # Dynamics (latent space):
    for k in range(n_tsteps):
        constraints.append(Z[:, k + 1] == A @ Z[:, k] + B @ U[:, k])

    # Terminal constraint (latent space):
    constraints.append(Z[:, n_tsteps] == zf)

    # Solve QP:
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    # Check if optimization failed:
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")

    # Extract solution:
    u_opt = U.value

    # Decode latent trajectory to observation space:
    x_opt, _ = prop_dyn_latent(model, A, B, C, x0, u_opt.T, use_C=False)

    # Return optimal control sequence and trajectory in observation space:
    return x_opt, u_opt

def plot_poles(A, title="Discrete System Pole Locations", threshold=1e-4):

    """
    Plot eigenvalues of matrix A on the complex plane with unit circle.

    INPUTS
    A: np.ndarray
        (n_z, n_z)
        System matrix (discrete-time)
    title (Optional): str
        N/A
        Plot title
    threshold (Optional): float
        N/A
        Threshold for eigenvalue


    OUTPUTS
    eigs: np.ndarray
        (n_z, )
        Eigenvalues of array
    """

    # Compute eigenvalues:
    eigs = np.linalg.eigvals(A)

    # Create figure:
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot unit circle:
    theta = np.linspace(0, 2 * np.pi, 1000)
    unit_circle_x = np.cos(theta)
    unit_circle_y = np.sin(theta)
    ax.plot(unit_circle_x, unit_circle_y, 'k--', linewidth=1.5, label='Unit Circle')

    # Plot eigenvalues:
    ax.plot(np.real(eigs), np.imag(eigs), 'rx', markersize=12,
            markeredgewidth=2, label='Poles (Eigenvalues)')

    # Formatting:
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Real Part', fontsize=12)
    ax.set_ylabel('Imaginary Part', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.axis('equal')

    # Set limits to show unit circle clearly:
    max_val = max(1.2, np.max(np.abs(eigs)) * 1.1)
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])

    # Check stability:
    eig_mags = np.abs(eigs)
    if np.all(eig_mags < 1 - threshold):
        stability_text = "STABLE"
        color = 'green'
    elif np.all(np.abs(eig_mags - 1) < threshold):
        stability_text = f"NEUTRALLY STABLE"
        color = 'yellow'
    elif np.any(eig_mags > 1 + threshold):
        stability_text = "UNSTABLE"
        color = 'red'
    else:
        stability_text = f"MARGINALLY STABLE"
        color = 'orange'

    # Display stability results:
    ax.text(0.02, 0.98, stability_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    # Formatting:
    plt.tight_layout()

    # Print eigenvalue information:
    print(f"\nEigenvalues and magnitudes:")
    print("-" * 50)
    for i, eig in enumerate(eigs):
        mag = np.abs(eig)
        print(f"λ_{i + 1} = {eig.real:8.5f} + {eig.imag:8.5f}j  |λ| = {mag:.5f}")
    print("-" * 50)

    # Return eigenvalues:
    return eigs

def weighted_output_CG(A, B, C, R, n_tsteps):

    """
    Compute weighted Output Controllability Gramian with control cost matrix R

    The weighted OCG is: W_R = C W_weighted C^T
                              = C (Σ A^k B R^{-1} B^T (A^T)^k) C^T

    INPUTS
    A: np.ndarray
        (n, n)
        System matrix
    B: np.ndarray
        (n, m)
        Control matrix
    C: np.ndarray
        (p, n)
        Observation matrix
    R: np.ndarray
        (m, m)
        Control cost matrix (must be positive definite)
    n_tsteps: int
        Number of time steps

    OUTPUTS
    Ctrl_o_weighted: np.ndarray
        (p, m * n_tsteps)
        Weighted Output Controllability Matrix (C @ Ctrl_weighted)
    W_R: np.ndarray
        (p, p)
        Weighted Output Controllability Gramian (C @ W_weighted @ C^T)
    """

    # Convert to numpy:
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    R = np.asarray(R)

    # Extract dimensions:
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    # Compute R^{-1}:
    R_inv = np.linalg.inv(R)

    # Compute weighted B matrix:
    B_weighted = B @ R_inv

    # Compute weighted output controllability matrix (C @ Ctrl_weighted):
    Ctrl_o_weighted = np.zeros((p, n_tsteps * m))
    for i in range(n_tsteps):
        Ctrl_o_weighted[:, i * m:(i + 1) * m] = C @ np.linalg.matrix_power(A, n_tsteps - 1 - i) @ B_weighted

    # Compute weighted latent Gramian:
    eigs = np.linalg.eigvals(A)

    if np.all(np.abs(eigs) < 1):

        # Stable system, solve using Lyapunov:
        W_weighted = sp.linalg.solve_discrete_lyapunov(A, B @ R_inv @ B.T)

    else:

        # Unstable, do a finite sum:
        W_weighted = np.zeros((n, n))
        for i in range(n_tsteps):
            Ak = np.linalg.matrix_power(A, i)
            W_weighted += Ak @ B @ R_inv @ B.T @ Ak.T

    # Project to observation space:
    W_R = C @ W_weighted @ C.T

    # Return weighted Output Controllability Matrix and Gramian:
    return Ctrl_o_weighted, W_R

def weighted_controllability(A, B, R, n_tsteps):

    """
    Compute weighted controllability gramian with control cost matrix R

    The weighted gramian is: W_weighted = Σ A^k B R^{-1} B^T (A^T)^k
    This accounts for control effort costs in the controllability analysis

    INPUTS
    A: np.ndarray
        (n, n)
        System matrix
    B: np.ndarray
        (n, m)
        Control matrix
    R: np.ndarray
        (m, m)
        Control cost matrix (must be positive definite)
    n_tsteps: int
        N/A
        Number of time steps

    OUTPUTS
    Ctrl_weighted: np.ndarray
        (n, m * n_tsteps)
        Weighted Controllability Matrix
    W_weighted: np.ndarray
        (n, n)
        Weighted Controllability Gramian
    """

    # Convert to numpy:
    A = np.asarray(A)
    B = np.asarray(B)
    R = np.asarray(R)

    # Extract dimensions:
    n = A.shape[0]
    m = B.shape[1]

    # Compute R^{-1}:
    R_inv = np.linalg.inv(R)

    # Create weighted B matrix:
    B_weighted = B @ R_inv

    # Compute weighted controllability matrix:
    Ctrl_weighted = np.zeros((n, n_tsteps * m))
    for i in range(n_tsteps):
        Ctrl_weighted[:, i * m:(i + 1) * m] = np.linalg.matrix_power(A, n_tsteps - 1 - i) @ B_weighted

    # Compute weighted Gramian:
    eigs = np.linalg.eigvals(A)

    # Check stability:
    if np.all(np.abs(eigs) < 1):

        # Stable system, solve using Lyapunov:
        W_weighted = sp.linalg.solve_discrete_lyapunov(A, B @ R_inv @ B.T)

    else:

        # Unstable, do a finite sum:
        W_weighted = np.zeros((n, n))
        for i in range(n_tsteps):
            Ak = np.linalg.matrix_power(A, i)
            W_weighted += Ak @ B @ R_inv @ B.T @ Ak.T

    # Return weighted Controllability Matrix and Gramian:
    return Ctrl_weighted, W_weighted

def riccati_opt_soft(_A, _B, _C, model, x_0, x_f, n_tsteps, _Q, _R, terminal_weight=1000.0):

    """
    Solves Riccati equation for optimal control as a QP with SOFT terminal constraint
    Now works in latent space with observation space constraints

    Formulation (in latent space):
        min     sum_{k=0}^{N-1} [z_k^T Q z_k + u_k^T R u_k] + (z_N - z_f)^T P_f (z_N - z_f)
        s.t.    z_{k+1} = A z_k + B u_k,  k=0,...,N-1
                z_0 = encoder(x_0) (given)
                (z_N ≈ z_f via large penalty weight)

    INPUTS
    _A: np.ndarray
        (n_z, n_z)
        Discretized system matrix (latent space)
    _B: np.ndarray
        (n_z, m)
        Discretized control matrix (latent space)
    _C: np.ndarray
        (n_x, n_z)
        Observation matrix
    model: dymad.models
        N/A
        Trained model for encoding/decoding
    x_0: np.ndarray
        (n_x,)
        Initial state (observation space)
    x_f: np.ndarray
        (n_x,)
        Final desired state (observation space - SOFT CONSTRAINT)
    n_tsteps: int
        Number of time steps
    _Q: np.ndarray
        (n_z, n_z)
        State weight (latent space)
    _R: np.ndarray
        (m, m)
        Control weight
    terminal_weight: float
        N/A
        Weight on terminal cost (larger = closer to hard constraint)

    OUTPUTS
    x_opt: np.ndarray
        (n_steps+1, n_x)
        Optimal trajectory in observation space
    u_opt: np.ndarray
        (n_tsteps, m)
        Optimal control sequence
    """

    # Convert to numpy:
    A = np.asarray(_A)
    B = np.asarray(_B)
    C = np.asarray(_C)
    x0 = np.asarray(x_0)
    xf = np.asarray(x_f)
    Q = np.asarray(_Q)
    R = np.asarray(_R)

    # Encode initial and final states to latent space:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    x0_torch = torch.from_numpy(x0).to(device=device, dtype=dtype)
    w0 = DynData().get_step(0).set_x(x0_torch.view(1, -1))
    z0 = model.encoder(w0).view(-1).detach().cpu().numpy()
    xf_torch = torch.from_numpy(xf).to(device=device, dtype=dtype)
    wf = DynData().get_step(0).set_x(xf_torch.view(1, -1))
    zf = model.encoder(wf).view(-1).detach().cpu().numpy()

    # Extract dimensions:
    n_z = A.shape[0]
    m = B.shape[1]

    # Define optimization variables (in latent space):
    Z = cp.Variable((n_z, n_tsteps + 1))
    U = cp.Variable((m, n_tsteps))

    # Build objective function:
    cost = 0
    for k in range(n_tsteps):
        cost += cp.quad_form(Z[:, k], Q)
        cost += cp.quad_form(U[:, k], R)

    # Add large penalty for terminal state deviation:
    P_f = terminal_weight * np.eye(n_z)
    cost += cp.quad_form(Z[:, n_tsteps] - zf, P_f)

    # Build constraints:
    constraints = []

    # Initial condition (latent space):
    constraints.append(Z[:, 0] == z0)

    # Dynamics:
    for k in range(n_tsteps):
        constraints.append(Z[:, k + 1] == A @ Z[:, k] + B @ U[:, k])

    # Solve QP:
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    # Check if optimization failed:
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")

    # Extract solution:
    u_opt = U.value.T

    # Decode latent trajectory to observation space:
    x_opt, _ = prop_dyn_latent(model, A, B, C, x0, u_opt, use_C=False)

    # Return optimal control sequence and trajectory in observation space:
    return x_opt, u_opt
