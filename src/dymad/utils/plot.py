import imageio
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

PALETTE = ["#000000", "#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e"]
LINESTY = ["-", "--", "-.", ":"]

# Disable logging for matplotlib to avoid clutter in DEBUG mode
plt_logger = logging.getLogger('matplotlib')
plt_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

def plot_trajectory(
        traj, ts, model_name, us=None, labels=None, ifclose=True, prefix='.',
        xidx=None, uidx=None, grid=None, xscl=None, uscl=None):
    """
    Plot trajectories with optional control inputs and save the figure.

    Args:
        traj (np.ndarray): Trajectory data, shape (n_steps, n_features) or (n_traj, n_steps, n_features)
        ts (np.ndarray): Time points corresponding to the trajectory data, shape (n_steps,)
        model_name (str): Name of the model for the plot title and filename
        us (np.ndarray, optional): Control inputs, shape (n_steps, n_controls)
        labels (list, optional): Labels for each trajectory, length must match number of trajectories
        ifclose (bool): Whether to close the plot after saving
        prefix (str): Directory prefix for saving the plot
        xidx (list, optional): Indices of state features to plot
        uidx (list, optional): Indices of control inputs to plot
        grid (tuple, optional): Tuple (n_rows, n_cols) for subplot layout
        xscl (str, optional): Scaling mode for state features ('01', '-11', 'std', or 'none')
        uscl (str, optional): Scaling mode for control inputs ('01', '-11', 'std', or 'none')
    """
    if traj.ndim == 2:
        traj = np.array([traj])

    Ntrj = len(traj)
    assert Ntrj == len(labels), \
        "Number of trajectories must match number of labels"

    # Plot the first trajectory and create the axes
    _, ax = plot_one_trajectory(traj[0], ts, idx=0, us=us, axes=None, label=labels[0],
                                xidx=xidx, uidx=uidx, grid=grid, xscl=xscl, uscl=uscl)

    if Ntrj > 1:
        # Add additional trajectories to the same axes
        for i in range(1, Ntrj):
            rmse = np.linalg.norm(traj[0] - traj[i]) / (traj[0].shape[0] - 1)**0.5
            plot_one_trajectory(
                traj[i], ts, idx=i, us=None, axes=ax,
                label=labels[i]+f" rmse: {rmse:4.3e}",
                xidx=xidx, uidx=uidx, grid=grid, xscl=xscl, uscl=uscl)

    # Adjust layout and save
    plt.tight_layout()
    if prefix != '.':
        os.makedirs(prefix, exist_ok=True)
    plt.savefig(f'{prefix}/{model_name}_prediction.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    if ifclose:
        plt.close()

def _scale_axes(ax, data, scl):
    if scl is None:
        return
    if scl == "01":
        ax.set_ylim([-0.1, 1.1])  # [0,1] range with small buffer
    elif scl == "-11":
        ax.set_ylim([-1.2, 1.2])  # [-1,1] range with buffer
    elif scl == "std":
        ax.set_ylim([-3, 3])      # ±3 std devs for standardized data
    else:  # mode=none
        ymx = np.max(data)
        ymn = np.min(data)
        ax.set_ylim([ymn-0.1*abs(ymn), ymx+0.1*abs(ymx)])  # Use data range with buffer

def plot_one_trajectory(
        traj, ts, idx=0, us=None, axes=None, label=None,
        xidx=None, uidx=None, grid=None, xscl=None, uscl=None):
    """
    Used by plot_trajectory to plot a single trajectory.
    """
    if xidx is None:
        dim_x = traj.shape[1]
        idx_x = np.arange(dim_x)
    else:
        idx_x = np.array(xidx)
        dim_x = len(idx_x)

    if us is None:
        dim_u = 0
    else:
        assert traj.shape[0] == us.shape[0], \
            "Trajectory and control input arrays must have the same time dimension"
        if uidx is None:
            dim_u = us.shape[1]
            idx_u = np.arange(dim_u)
        else:
            idx_u = np.array(uidx)
            dim_u = len(idx_u)

    # Trim time array if needed
    if len(ts) > traj.shape[0]:
        ts = ts[:traj.shape[0]]

    # Set up subplot layout from metadata or use default
    if grid is None:
        # Default: one column with a row per state
        n_rows, n_cols = dim_x + dim_u, 1
        fig_size = (6, n_rows * 2)
    else:
        n_rows, n_cols = grid
        fig_size = (3 * n_cols, 2.5 * n_rows)

    if axes is None:
        # Create subplots
        fig, ax = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex=True)
        if n_rows * n_cols == 1:
            ax = [ax]  # Make it iterable for single subplot
        else:
            ax = ax.flatten()
    else:
        # Use provided axes
        fig = axes[0].figure
        ax = axes

    # Plot each state
    for i in range(dim_x):
        ax[i].plot(ts, traj[:, idx_x[i]], LINESTY[idx%4], color=PALETTE[idx%6], linewidth=2, label=label)
        ax[i].set_xlim([0, ts[-1]])
        ax[i].grid(True, alpha=0.3)

        ax[i].set_ylabel(f'State {idx_x[i]+1}', fontsize=10)
        if i == 0:  # Only show legend on first subplot
            ax[i].legend(loc='best', fontsize=9)

        if axes is None:
            _scale_axes(ax[i], traj[:, idx_x[i]], xscl)

    if axes is None:
        if dim_u > 0:
            # Plot only once as this is from data
            offset = dim_x
            for i in range(dim_u):
                ax[offset + i].plot(ts, us[:, idx_u[i]], '-', color='#3498db', linewidth=2)
                ax[offset + i].set_xlim([0, ts[-1]])
                ax[offset + i].grid(True, alpha=0.3)
                ax[offset + i].set_ylabel(f'Control {i+1}', fontsize=10)

                _scale_axes(ax[offset + i], us[:, idx_u[i]], uscl)

    for i in range(n_cols):
        ax[-i-1].set_xlabel('Time', fontsize=10)
        ax[-i-1].set_xlim([2*ts[0]-ts[1], 2*ts[-1]-ts[-2]])

    return fig, ax

def plot_hist(hist, epoch, model_name, ifclose=True, prefix='.'):
    """
    Plot training history with loss curves for train/validation/test sets.

    Args:
        hist (list): History of losses, where hist[0] is epoch numbers and
                     hist[1:] contains losses for train, validation, and test sets.
        epoch (int): Number of epochs to plot.
        model_name (str): Name of the model for the plot title and filename.
        ifclose (bool): Whether to close the plot after saving.
        prefix (str): Directory prefix for saving the plot.
    """
    tmp = np.array(hist).T
    _e, _h = tmp[0][:epoch], tmp[1:,:epoch]

    # Create figure
    plt.figure(figsize=(8, 6))

    # Plot loss curves with modern styling
    plt.semilogy(_e, _h[0], '--', color='#3498db', linewidth=2,
                 label='Training', alpha=0.8)
    plt.semilogy(_e, _h[1], '-', color='#e74c3c', linewidth=2,
                 label='Validation', alpha=0.9)
    plt.semilogy(_e, _h[2], '-', color='#2ecc71', linewidth=2,
                 label='Test', alpha=0.9)

    # Styling
    plt.xlim([_e[0], _e[-1]+1])
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title(f'{model_name} - Training History', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)

    # Improve tick formatting
    plt.tick_params(axis='both', which='major', labelsize=10)

    # Save with clean formatting
    plt.tight_layout()
    plt.savefig(f'{prefix}/{model_name}_history.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    if ifclose:
        plt.close()

def plot_summary(npz_files, labels=None, ifscl=True, ifclose=True, prefix='.'):
    """
    Plot training loss and trajectory RMSE for multiple summary files on the same figure.

    Args:
        npz_files (list): List of NPZ file paths containing summary data.
        labels (list): List of labels for each run (optional).
        ifscl (bool): If True, scale the loss by the first epoch loss.
        ifclose (bool): Whether to close the plot after saving.
        prefix (str): Directory prefix for saving the plot.
    """
    npzs = [np.load(_f) for _f in npz_files]
    ax = None
    for idx, npz in enumerate(npzs):
        label = labels[idx] if labels is not None else f'Run {idx+1}'
        fig, ax = plot_one_summary(npz, label=label, index=idx, ifscl=ifscl, axes=ax)

    plt.tight_layout()
    if prefix != '.':
        os.makedirs(prefix, exist_ok=True)
    plt.savefig(f'{prefix}/node_summary.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    if ifclose:
        plt.close()

    return npzs

def plot_one_summary(npz, label='', index=0, ifscl=True, axes=None):
    """
    Plot training loss and trajectory RMSE from a summary file.

    Args:
        npz (dict): Dictionary from the NPZ file.
        label (str): Label for the plot legend.
        index (int): Index to select color from the PALETTE.
        ifscl (bool): If True, scale the loss by the first epoch loss.
        axes (list, optional): List of axes to plot on. If None, creates new subplots.
    """
    clr = PALETTE[index % len(PALETTE)]

    e_loss, h_loss = npz['epoch_loss'], npz['losses']
    e_rmse, h_rmse = npz['epoch_rmse'], npz['rmses']

    if axes is None:
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 6))
    else:
        fig = axes[0].figure
        ax = axes
    scl = h_loss[0, 0] if ifscl else 1.0
    ax[0].semilogy(e_loss, h_loss[0]/scl, '-', color=clr, label=label)
    if ifscl:
        ax[0].set_title('Training Loss (relative drop)')
        ax[0].set_ylabel('Relative Loss')
    else:
        ax[0].set_title('Training Loss *magnitude not to compare*')
        ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].semilogy(e_rmse, h_rmse[0], '-',  color=clr, label=f'{label}, Train')
    ax[1].semilogy(e_rmse, h_rmse[2], '--', color=clr, label=f'{label}, Test')
    ax[1].set_title('Trajectory RMSE')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('RMSE')
    ax[1].legend()

    return fig, ax

def _get_contour_func(ax, mode):
    if mode == 'contour':
        contour_func = ax.contour
    elif mode == 'contourf':
        contour_func = ax.contourf
    elif mode == 'tricontourf':
        contour_func = ax.tricontourf
    return contour_func

def plot_contour(
        arrays, x=None, y=None, vmin=None, vmax=None, levels=20,
        figsize=(12, 4), colorbar=True, axes=None, label=None, grid=None,
        mode='contourf', **kwargs):
    """
    Plot a grid of contour plots for a list of 2D arrays.

    Parameters:
        arrays (list of np.ndarray): List of 2D arrays to plot.
        titles (list of str): Titles for each subplot.
        x, y (np.ndarray): Optional meshgrid for axes.
        vmin, vmax (float): Color limits.
        levels (int or array): Contour levels.
        figsize (tuple): Figure size.
        colorbar (bool): Whether to add a colorbar.
        axes: Existing figure/axes tuple.
        label (list): Titles for each subplot.
        grid (tuple): Grid layout (n_rows, n_cols).
        mode (str): 'contour', 'contourf', or 'tricontourf'.
        **kwargs: Additional arguments for contourf.

    Returns:
        fig, axes: Matplotlib figure and axes.
    """
    # Validate inputs
    n = len(arrays)
    if grid is None:
        grid = (1, n)
    assert grid[0]*grid[1] >= n, "Grid size too small for number of arrays"

    if label is not None:
        assert len(label) == n, "Number of labels must match number of arrays, if provided"

    if axes is None:
        fig, ax = plt.subplots(grid[0], grid[1], figsize=figsize, sharex=True, sharey=True)
    else:
        fig, ax = axes

    assert mode in ['contour', 'contourf', 'tricontourf'], "Mode must be 'contour', 'contourf', or 'tricontourf'"
    if mode == 'tricontourf':
        assert x is not None and y is not None, "x and y must be provided for tricontourf"

    # Prepare contour arguments
    contour_args = {}
    contour_args.update(**kwargs)
    if isinstance(levels, int):
        vmin = arrays.min() if vmin is None else vmin
        vmax = arrays.max() if vmax is None else vmax
        _lvls = np.linspace(vmin, vmax, levels)
    elif isinstance(levels, (list, np.ndarray)):
        _lvls = levels
    else:
        raise ValueError("Levels must be an integer or a list/array of levels")
    contour_args = {'levels': _lvls}

    # Plotting
    ims = []
    _ax = ax.flatten() if grid[0]*grid[1] > 1 else [ax]
    for i, arr in enumerate(arrays):
        _func = _get_contour_func(_ax[i], mode)
        if x is not None and y is not None:
            im = _func(x, y, arr, **contour_args)
        else:
            im = _func(arr, **contour_args)
        ims.append(im)
        if label:
            _ax[i].set_title(label[i])
    if colorbar:
        fig.colorbar(ims[0], ax=ax, orientation='vertical', fraction=0.02, pad=0.04)

    return fig, ax

def compare_contour(
        x_true, x_pred, x=None, y=None, vmin=None, vmax=None, levels=20,
        figsize=(12, 4), colorbar=True, axes=None, label=None,
        mode='contourf', **kwargs):
    """
    Compare two contours with error contours.
    """
    vmin = x_true.min() if vmin is None else vmin
    vmax = x_true.max() if vmax is None else vmax
    x_diff = x_true - x_pred
    err    = np.linalg.norm(x_diff) / np.linalg.norm(x_true)
    label  = ['Truth', 'Reconstructed', f'Error: {err*100:4.2f}%']
    arrays = [x_true, x_pred, x_diff]
    return plot_contour(
        arrays, x=x, y=y, vmin=vmin, vmax=vmax, levels=levels,
        figsize=figsize, colorbar=colorbar, axes=axes, label=label, grid=(1,3),
        mode=mode, **kwargs)

def animate(fig_func, filename, fps=10, n_frames=None, writer_args={}, fig_args={}):
    """
    Create an animation by calling a figure-generating function for each frame.
    Args:
        fig_func (function): Function that generates a figure for a given frame index.
                             It should accept the frame index as its first argument,
                             and return a matplotlib figure and axes.
        filename (str): Output filename for the output file.
        fps (int): Frames per second for the animation.
        n_frames (int): Total number of frames in the animation.
        writer_args (dict): Additional keyword arguments to pass to imageio.get_writer.
        fig_args (dict): Additional keyword arguments to pass to fig_func.
    """
    writer = imageio.get_writer(filename, fps=fps, **writer_args)
    for j in range(n_frames):
        logger.info(f'Generating frame {j+1}/{n_frames} for {filename}')

        fig, ax = fig_func(j, **fig_args)

        # Render the figure to a numpy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(2*h, 2*w, 4)
        writer.append_data(frame[:,:,:3])

        # Removing existing objects otherwise they accumulate in canvas.draw
        for _a in ax.flatten():
            for _c in _a.collections:
                _c.remove()
        plt.close(fig)
    writer.close()
    logger.info(f'Animation saved to {filename}')
