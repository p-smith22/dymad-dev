import logging
import numpy as np
import torch
from torchdiffeq import odeint
from typing import Union

from dymad.io import DynData
from dymad.numerics import expm_low_rank, expm_full_rank
from dymad.utils import ControlInterpolator

logger = logging.getLogger(__name__)

def _process(data, Nb, Nt, base_dim, device):
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if data.ndim == base_dim:
        # Expand to batches and time steps
        # data: (...) -> (_Nb, _Nt, ...)
        _arr = data.clone().detach().to(device)
        _arr = _arr.unsqueeze(0).unsqueeze(0)  # (1, 1, ...)
        _shape = (Nb, Nt) + (1,) * base_dim
        _arr = _arr.repeat(*_shape)  # (_Nb, _Nt, ...)
    elif data.ndim == base_dim + 1:
        # Expand to batches
        assert data.shape[0] == Nt, f"3D data first dimension must match time steps. Got {data.shape[0]} vs {Nt}"
        _arr = data.clone().detach().to(device)
        _shape = (Nb, 1) + (1,) * base_dim
        _arr = _arr.unsqueeze(0).repeat(*_shape)  # (_Nb, _Nt, ...)
    elif data.ndim == base_dim + 2:
        assert data.shape[0] == Nb and data.shape[1] == Nt, \
            f"4D data first two dimensions must match batch size and time steps. Got {data.shape[:2]} vs ({Nb}, {Nt})"
        _arr = data.clone().detach().to(device)
    else:
        raise ValueError(f"data must be 2D or 3D or 4D tensor. Got {data.shape}")
    return _arr

def _prepare_data(x0, ts, us, device, edge_index=None, edge_weights=None, edge_attr=None):
    is_batch = x0.ndim == 2
    _Nb = x0.shape[0] if is_batch else 1

    # Initial conditions
    if is_batch:
        if x0.ndim != 2:
            raise ValueError(f"Batch mode: x0 must be 2D. Got x0: {x0.shape}")
        _x0 = x0.clone().detach().to(device)
    else:
        if x0.ndim != 1:
            raise ValueError(f"Single mode: x0 must be 1D. Got x0: {x0.shape}")
        _x0 = x0.clone().detach().to(device).unsqueeze(0)

    # Time stations
    _ts, _Nt = None, None
    if ts is not None:
        if isinstance(ts, np.ndarray):
            _ts = torch.from_numpy(ts).float().to(device)
        else:
            _ts = ts.float().to(device)
        _Nt = _ts.shape[0]

    # Inputs
    _us, _Nu = None, None
    if us is not None:
        if is_batch:
            if us.ndim != 3:
                raise ValueError(f"Batch mode: us must be 3D. Got us: {us.shape}")
            _us = us.clone().detach().to(device)
        else:
            if us.ndim != 2:
                raise ValueError(f"Single mode: us must be 2D. Got us: {us.shape}")
            _us = us.clone().detach().to(device).unsqueeze(0)
        _Nu = _us.shape[1]  # Time steps from us

    # Check step consistency
    if _Nt is None:
        if _Nu is None:
            raise ValueError("Either ts or us must be provided to determine time steps.")
        n_steps = _Nu
    else:
        if _Nu is not None:
            if _Nt != _Nu:
                raise ValueError(f"ts and us must have the same number of time steps. Got ts: {_Nt}, us: {_Nu}")
        n_steps = _Nt

    # Edge data
    _ei = _process(edge_index, _Nb, n_steps, 2, device)    # (_Nb, _Nt, 2, n_edges)
    if _ei is not None:
        _ei = _ei.long()
    _ew = _process(edge_weights, _Nb, n_steps, 1, device)  # (_Nb, _Nt, n_edges)
    _ea = _process(edge_attr, _Nb, n_steps, 2, device)     # (_Nb, _Nt, n_edges, n_edge_features)

    return _x0, _ts, _us, n_steps, is_batch, _ei, _ew, _ea

# ------------------
# Continuous-time case
# ------------------

def predict_continuous(
    model,
    x0: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    us: torch.Tensor = None,
    edge_index: torch.Tensor = None,
    edge_weights: torch.Tensor = None,
    edge_attr: torch.Tensor = None,
    method: str = 'dopri5',
    order: str = 'cubic',
    **kwargs
) -> torch.Tensor:
    """
    Predict trajectory(ies) for regular (non-graph) models with batch support.

    Args:
        model: Model with encoder, decoder, and dynamics methods.
        x0 (torch.Tensor): Initial state(s).

            - Single: shape (n_features,)
            - Batch: shape (batch_size, n_features)

        ts (Union[np.ndarray, torch.Tensor]): Time points (n_steps,).
        us (torch.Tensor, optional): Control trajectory(ies).

            - Single: shape (n_steps, n_controls)
            - Batch: shape (batch_size, n_steps, n_controls)

        edge_index (torch.Tensor, optional): Edge indices for the graph.
        method (str): ODE solver method (default: 'dopri5').
        order (str): Interpolation method for control inputs ('zoh', 'linear', or 'cubic').

    Returns:
        torch.Tensor: Predicted trajectory(ies).

            - Single: shape (n_steps, n_features)
            - Batch: shape (n_steps, batch_size, n_features)

    Raises:
        ValueError: If input dimensions do not match requirements.
    """
    device = x0.device
    _x0, ts, _us, n_steps, is_batch, _ei, _ew, _ea = _prepare_data(
        x0, ts, us, device,
        edge_index=edge_index, edge_weights=edge_weights, edge_attr=edge_attr)

    _has_u = _us is not None
    if _has_u:
        logger.debug(f"predict_continuous: {'Batch' if is_batch else 'Single'} mode (controlled)")
        u0 = _us[:, 0, :]
        u_intp = ControlInterpolator(ts, _us, order=order)
    else:
        logger.debug(f"predict_continuous: {'Batch' if is_batch else 'Single'} mode (autonomous)")
        u0 = None

    _has_ei = _ei is not None
    if _has_ei:
        ei0 = _ei[:, 0, :, :]
        ei_intp = ControlInterpolator(ts, _ei, axis=-3, order='zoh')
    else:
        ei0 = None

    _has_ew = _ew is not None
    if _has_ew:
        ew0 = _ew[:, 0, :]
        ew_intp = ControlInterpolator(ts, _ew, axis=-2, order=order)
    else:
        ew0 = None

    _has_ea = _ea is not None
    if _has_ea:
        ea0 = _ea[:, 0, :, :]
        ea_intp = ControlInterpolator(ts, _ea, axis=-3, order=order)
    else:
        ea0 = None

    z0 = model.encoder(DynData(x=_x0, u=u0, ei=ei0, ew=ew0, ea=ea0))
    def ode_func(t, z):
        u  = u_intp(t)  if _has_u else None
        ei = ei_intp(t) if _has_ei else None
        ew = ew_intp(t) if _has_ew else None
        ea = ea_intp(t) if _has_ea else None
        x  = model.decoder(z, DynData(ei=ei, ew=ew, ea=ea))
        _, z_dot, _ = model(DynData(x=x, u=u, ei=ei, ew=ew, ea=ea))
        return z_dot

    logger.debug(f"predict_continuous: Starting ODE integration with shape {z0.shape}, method {method}, and interpolation order {order if _has_u else 'N/A'}")
    z_traj = odeint(ode_func, z0, ts, method=method, **kwargs)
    logger.debug(f"predict_continuous: Completed integration, trajectory shape: {z_traj.shape}")

    if _ei is None:
        x_traj = model.decoder(z_traj.view(-1, z_traj.shape[-1]), None).view(n_steps, z_traj.shape[1], -1)
    else:
        tmp = z_traj.permute(1, 0, 2, 3)  # (batch_size, n_steps, node, z_dim)
        x_traj = model.decoder(tmp, DynData(ei=_ei, ew=_ew, ea=_ea)).permute(1, 0, 2)
    if not is_batch:
        x_traj = x_traj.squeeze(1)

    logger.debug(f"predict_continuous: Final trajectory shape {x_traj.shape}")
    return x_traj

def predict_continuous_exp(
    model,
    x0: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    **kwargs
) -> torch.Tensor:
    """
    Predict trajectory(ies) for regular (non-graph) models with batch support.

    Autonomous case using matrix exponential.  In continuous-time, we compute exp(A*dt).

    Currently only for KBF-type models with linear dynamics.

    Raises:
        ValueError: If input dimensions do not match requirements.
    """
    device = x0.device
    _x0, ts, _, n_steps, is_batch, _, _, _ = _prepare_data(x0, ts, None, device)

    # Get the system matrix
    if model.dynamics_net.mode == "full":
        W = (model.dynamics_net.weight, )
    else:
        U = model.dynamics_net.U
        V = model.dynamics_net.V
        W = (U, V)

    logger.debug(f"predict_continuous_exp: {'Batch' if is_batch else 'Single'} mode (autonomous)")
    z0 = model.encoder(DynData(x=_x0))

    logger.debug(f"predict_continuous_exp: Starting ODE integration with shape {z0.shape}")
    dt = ts - ts[0]  # (n_steps,)
    if len(W) == 1:
        z_traj = expm_full_rank(W[0].T, dt, z0)
    elif len(W) == 2:
        # Low-rank case: use a specialized function to exponentiate in reduced space
        z_traj = expm_low_rank(W[1], W[0], dt, z0)
    logger.debug(f"predict_continuous_exp: Completed integration, trajectory shape: {z_traj.shape}")

    x_traj = model.decoder(z_traj.view(-1, z_traj.shape[-1]), None).view(n_steps, z_traj.shape[1], -1)
    if not is_batch:
        x_traj = x_traj.squeeze(1)

    logger.debug(f"predict_continuous_exp: Final trajectory shape {x_traj.shape}")
    return x_traj

def predict_continuous_fenc(
    model,
    x0: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    us: torch.Tensor = None,
    **kwargs
) -> torch.Tensor:
    device = x0.device
    _x0, ts, _us, n_steps, is_batch, _, _, _ = _prepare_data(x0, ts, us, device)

    logger.debug(f"predict_continuous_fenc: {'Batch' if is_batch else 'Single'} mode")

    if _us is not None:
        # Initial state preparation
        u0 = _us[:, 0, :]
    else:
        u0 = None
    z0 = model.encoder(DynData(x=_x0, u=u0))

    # Discrete-time forward pass
    logger.debug(f"predict_continuous_fenc: Starting forward iterations with shape {z0.shape}")
    z_traj = [z0]
    for k in range(n_steps - 1):
        u_k = None if _us is None else _us[:, k, :]
        z_next = model.fenc_step(z_traj[-1], DynData(u=u_k), ts[k+1]-ts[k])
        z_traj.append(z_next)

    z_traj = torch.stack(z_traj, dim=0)  # (n_steps, batch_size, z_dim)
    logger.debug(f"predict_continuous_fenc: Completed integration, trajectory shape: {z_traj.shape}")

    x_traj = model.decoder(z_traj.view(-1, z_traj.shape[-1]), None).view(n_steps, z_traj.shape[1], -1)
    if not is_batch:
        x_traj = x_traj.squeeze(1)

    logger.debug(f"predict_continuous_fenc: Final trajectory shape {x_traj.shape}")
    return x_traj

# ------------------
# Discrete-time case
# ------------------

def predict_discrete(
    model,
    x0: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    us: torch.Tensor = None,
    edge_index: torch.Tensor = None,
    edge_weights: torch.Tensor = None,
    edge_attr: torch.Tensor = None,
    **kwargs
) -> torch.Tensor:
    """
    Predict trajectory(ies) for regular (non-graph) models with batch support.

    Args:
        model: Model with encoder, decoder, and dynamics methods
        x0: Initial state(s):

            - Single: (n_features,)
            - Batch: (batch_size, n_features)

        ts (Union[np.ndarray, torch.Tensor]): Time points (n_steps,).
        us: Control trajectory(ies):

            - Single: (n_steps, n_controls)
            - Batch: (batch_size, n_steps, n_controls)

        edge_index (torch.Tensor): Edge indices for the graph.

    Returns:
        torch.Tensor:
            Predicted trajectory(ies)

            - Single: (n_steps, n_features)
            - Batch: (n_steps, batch_size, n_features)

    Raises:
        ValueError: If input dimensions don't match requirements
    """
    device = x0.device
    _x0, _, _us, n_steps, is_batch, _ei, _ew, _ea = _prepare_data(
        x0, ts, us, device,
        edge_index=edge_index, edge_weights=edge_weights, edge_attr=edge_attr)

    logger.debug(f"predict_discrete: {'Batch' if is_batch else 'Single'} mode")

    _has_u  = _us is not None
    _has_ei = _ei is not None
    _has_ew = _ew is not None
    _has_ea = _ea is not None

    u0  = _us[:, 0, :] if _has_u else None
    ei0 = _ei[:, 0, :, :] if _has_ei else None
    ew0 = _ew[:, 0, :] if _has_ew else None
    ea0 = _ea[:, 0, :, :] if _has_ea else None
    z0  = model.encoder(DynData(x=_x0, u=u0, ei=ei0, ew=ew0, ea=ea0))

    # Discrete-time forward pass
    logger.debug(f"predict_discrete: Starting forward iterations with shape {z0.shape}")
    z_traj = [z0]
    for k in range(n_steps - 1):
        u_k = _us[:, k, :] if _has_u else None
        eik = _ei[:, k, :, :] if _has_ei else None
        ewk = _ew[:, k, :] if _has_ew else None
        eak = _ea[:, k, :, :] if _has_ea else None
        x_k = model.decoder(z_traj[-1], DynData(ei=eik, ew=ewk, ea=eak))
        _, z_next, _ = model(DynData(x=x_k, u=u_k, ei=eik, ew=ewk, ea=eak))
        z_traj.append(z_next)

    z_traj = torch.stack(z_traj, dim=0)  # (n_steps, batch_size, z_dim)
    logger.debug(f"predict_discrete: Completed integration, trajectory shape: {z_traj.shape}")

    if _ei is None:
        x_traj = model.decoder(z_traj.view(-1, z_traj.shape[-1]), None).view(n_steps, z_traj.shape[1], -1)
    else:
        # after stack: z_traj (n_steps, batch_size, node, z_dim)
        tmp = z_traj.permute(1, 0, 2, 3)  # (batch_size, n_steps, node, z_dim)
        x_traj = model.decoder(tmp, DynData(ei=_ei, ew=_ew, ea=_ea)).permute(1, 0, 2)

    if not is_batch:
        x_traj = x_traj.squeeze(1)

    logger.debug(f"predict_discrete: Final trajectory shape {x_traj.shape}")
    return x_traj

def predict_discrete_exp(
    model,
    x0: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    **kwargs
) -> torch.Tensor:
    """
    Predict trajectory(ies) for regular (non-graph) models with batch support.

    Autonomous case using matrix exponential.  In discrete-time, this is equivalent to
    repeated application of the dynamics.

    Currently only for KBF-type models with linear dynamics.

    Raises:
        ValueError: If input dimensions don't match requirements
    """
    device = x0.device
    # Use _prepare_data for consistency
    _x0, _, _, n_steps, is_batch, _, _, _ = _prepare_data(x0, ts, None, device)

    logger.debug(f"predict_discrete: {'Batch' if is_batch else 'Single'} mode")

    # Initial state preparation
    z0 = model.encoder(DynData(x=_x0))

    # Discrete-time forward pass
    logger.debug(f"predict_discrete_exp: Starting forward iterations with shape {z0.shape}")
    z_traj = [z0]
    for k in range(n_steps - 1):
        z_next = model.dynamics(z_traj[-1], None)
        z_traj.append(z_next)
    z_traj = torch.stack(z_traj, dim=0)  # (n_steps, batch_size, z_dim)
    logger.debug(f"predict_discrete_exp: Completed integration, trajectory shape: {z_traj.shape}")

    x_traj = model.decoder(z_traj.view(-1, z_traj.shape[-1]), None).view(n_steps, z_traj.shape[1], -1)
    if not is_batch:
        x_traj = x_traj.squeeze(1)

    logger.debug(f"predict_discrete_exp: Final trajectory shape {x_traj.shape}")
    return x_traj
