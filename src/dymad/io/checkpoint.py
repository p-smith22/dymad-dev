import logging
import numpy as np
import os
import torch
from typing import Callable, Dict, List, Optional, Union, Tuple, Type

from dymad.io.data import DynData
from dymad.io.trajectory_manager import TrajectoryManager
from dymad.transform import Autoencoder, make_transform
from dymad.utils.misc import load_config

logger = logging.getLogger(__name__)

def load_checkpoint(model, optimizer, schedulers, ref_checkpoint_path, load_from_checkpoint=False, inference_mode=False):
    """
    Load a checkpoint from the specified path.

    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        schedulers (list[torch.optim.lr_scheduler._LRScheduler]): The schedulers to load the state into.
        ref_checkpoint_path (str): Reference path to the checkpoint file - Same as the current case.
        load_from_checkpoint (bool or str): If True, load from ref_checkpoint_path; if str, use it as the path; otherwise, skip loading.
        inference_mode (bool, optional): If True, skip loading optimizer and schedulers.

    Returns:
        tuple: A tuple containing:

        - int: The epoch number from which to continue training.
        - float: The best loss recorded in the checkpoint.
        - list: History of losses.
        - list: History of RMSE of trajectories - can be different from loss.
        - dict: Metadata about the data.
    """
    mode = "Inference" if inference_mode else "Training"
    logger.info(f"{mode} mode is enabled.")

    checkpoint_path = None
    if isinstance(load_from_checkpoint, str):
        checkpoint_path = load_from_checkpoint
    elif load_from_checkpoint:
        checkpoint_path = ref_checkpoint_path

    if checkpoint_path is None:
        logger.info(f"Got load_from_checkpoint={load_from_checkpoint}, resulting in checkpoint_path=None. Starting from scratch.")
        return 0, float("inf"), [], [], None

    if not os.path.exists(checkpoint_path):
        logger.info(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0, float("inf"), [], [], None

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if not inference_mode:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        assert len(schedulers) == len(checkpoint["scheduler_state_dict"]), \
            f"Expected {len(schedulers)} schedulers, but got {len(checkpoint['scheduler_state_dict'])} in checkpoint."
        for i in range(len(schedulers)):
            schedulers[i].load_state_dict(checkpoint["scheduler_state_dict"][i])

        # In this case, we do a new training, so we reset the best loss
        return checkpoint["epoch"], float("inf"), checkpoint["hist"], checkpoint["rmse"], checkpoint["metadata"]

    return checkpoint["epoch"], checkpoint["best_loss"], checkpoint["hist"], checkpoint["rmse"], checkpoint["metadata"]

def save_checkpoint(model, optimizer, schedulers, epoch, best_loss, hist, rmse, metadata, checkpoint_path):
    """
    Save the model, optimizer, and scheduler states to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        schedulers (list[torch.optim.lr_scheduler._LRScheduler]): The schedulers to save.
        epoch (int): The current epoch number.
        best_loss (float): The best loss recorded so far.
        hist (list): The history of losses.
        rmse (list): The history of RMSE of trajectories - can be different from loss.
        metadata (dict): Metadata about the data.
        checkpoint_path (str): Path to save the checkpoint file.
    """
    # logger.info(f"Saving checkpoint to {checkpoint_path}")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": [scheduler.state_dict() for scheduler in schedulers],
        "best_loss": best_loss,
        "hist": hist,
        "rmse": rmse,
        "metadata": metadata,
    }, checkpoint_path)

def _atleast_3d(x):
    if x.ndim == 2:
        return np.expand_dims(x, axis=0)
    return x

def load_model(model_class, checkpoint_path, config_path=None, config_mod=None):
    """
    Load a model from a checkpoint file.

    Args:
        model_class (torch.nn.Module): The class of the model to load.
        checkpoint_path (str): Path to the checkpoint file.
        config_path (str, optional): Path to the configuration file, used as backup.  Deprecated.
        config_mod (dict, optional): Dictionary to merge into the config.  Deprecated.

    Returns:
        tuple: A tuple containing the model and a prediction function.

        - nn.Module: The loaded model.
        - callable: A function to predict trajectories in data space.
    """
    chkpt = torch.load(checkpoint_path, weights_only=False)
    md = chkpt['metadata']
    dtype = torch.double if md['config']['data'].get('double_precision', False) else torch.float
    torch.set_default_dtype(dtype)   # GNNs use the default dtype, so we need to set it here

    # Model
    if config_path is not None:
        config = load_config(config_path, config_mod)
        model_config = md['config'].get('model', config['model'])
    else:
        model_config = md['config'].get('model', None)
    model = model_class(model_config, md, dtype=dtype)
    model.load_state_dict(chkpt['model_state_dict'])

    # Check if autonomous
    _is_autonomous = md.get('transform_u_state', None) is None

    # Data transformations
    _data_transform_x = make_transform(md['config'].get('transform_x', None))
    _data_transform_x.load_state_dict(md["transform_x_state"])

    if not _is_autonomous:
        _data_transform_u = make_transform(md['config'].get('transform_u', None))
        _data_transform_u.load_state_dict(md["transform_u_state"])

    def _proc_x0(x0, device):
        _x0 = np.array(_data_transform_x.transform(_atleast_3d(x0)))[:,0,:]
        _x0 = torch.tensor(_x0, dtype=dtype, device=device)
        return _x0

    def _proc_u(us, device):
        _u  = np.array(_data_transform_u.transform(_atleast_3d(us)))
        if isinstance(_u, np.ndarray):
            _u = torch.tensor(_u, dtype=dtype, device=device)
        else:
            _u = _u.clone().detach().to(device)
        return _u

    def _proc_prd(pred):
        _prd = np.array(_data_transform_x.inverse_transform(_atleast_3d(pred))).squeeze()
        if _prd.ndim == 2:
            return _prd
        return np.transpose(_prd, (1, 0, 2))

    # Prediction in data space
    if model.GRAPH:
        if _is_autonomous:
            def predict_fn(x0, t, ei=None, device="cpu"):
                """Predict trajectory in data space."""
                _x0 = _proc_x0(x0, device)
                with torch.no_grad():
                    pred = model.predict(_x0, DynData(ei=ei), t).cpu().numpy()
                return _proc_prd(pred)
        else:
            def predict_fn(x0, us, t, ei=None, device="cpu"):
                """Predict trajectory in data space."""
                _x0 = _proc_x0(x0, device)
                _u  = _proc_u(us, device)
                with torch.no_grad():
                    pred = model.predict(_x0, DynData(u=_u, ei=ei), t).cpu().numpy()
                return _proc_prd(pred)
    else:
        if _is_autonomous:
            def predict_fn(x0, t, device="cpu"):
                """Predict trajectory in data space."""
                _x0 = _proc_x0(x0, device)
                with torch.no_grad():
                    pred = model.predict(_x0, DynData(), t).cpu().numpy()
                return _proc_prd(pred)
        else:
            def predict_fn(x0, us, t, device="cpu"):
                """Predict trajectory in data space."""
                _x0 = _proc_x0(x0, device)
                _u  = _proc_u(us, device)
                with torch.no_grad():
                    pred = model.predict(_x0, DynData(u=_u), t).cpu().numpy()
                return _proc_prd(pred)

    return model, predict_fn


class DataInterface:
    """
    Interface for data transforms, possibly with learned autoencoders.

    It loads the model (if available) and data, sets up the necessary transformations,
    and provides methods to encode, decode, and apply observables.

    Cases:

        - [Priority] checkpoint_path is given: Load the data transforms and model from the checkpoint.
          May contain autoencoders.
        - [Secondary] config_path and/or config_mod is given: Instantiate the data transforms from the config.
          No model (i.e., autoencoders) in this case.
    """
    def __init__(self,
                 model_class: Union[Type[torch.nn.Module], None] = None,
                 checkpoint_path: Union[str, None] = None,
                 config_path: Union[str, None] = None,
                 config_mod: Optional[dict] = None,
                 device: Optional[torch.device] = None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        metadata, self.has_model = self._init_metadata(checkpoint_path, config_path, config_mod)
        self._setup_data(metadata)

        if self.has_model:
            self.model, _ = load_model(model_class, checkpoint_path)
            encoder = lambda x: self.model.encoder(DynData(x=x))
            decoder = lambda z: self.model.decoder(z, None)
            enc = Autoencoder(self.model, encoder, decoder)
            self._trans_x.append(enc)

        self.NT = self._trans_x.NT

    def _init_metadata(self, checkpoint_path, config_path, config_mod) -> Tuple[Dict, bool]:
        """Initialize metadata from config or checkpoint."""
        if checkpoint_path is not None:
            assert os.path.exists(checkpoint_path), "Checkpoint path does not exist."
            return torch.load(checkpoint_path, weights_only=False)['metadata'], True
        _config = load_config(config_path, config_mod)
        return {'config': _config}, False

    def _setup_data(self, metadata) -> None:
        """Setup data loaders and datasets.

        Striped from TrainerBase.
        """
        tm = TrajectoryManager(metadata, device=self.device)
        _dataloaders, _, _ = tm.process_all()

        # Turn off shuffling to ensure fixed order of samples
        self.train_loader = torch.utils.data.DataLoader(
            _dataloaders[0].dataset, batch_size=_dataloaders[0].batch_size, shuffle=False, collate_fn=DynData.collate)
        self.validation_loader = torch.utils.data.DataLoader(
            _dataloaders[1].dataset, batch_size=_dataloaders[1].batch_size, shuffle=False, collate_fn=DynData.collate)
        self.test_loader = torch.utils.data.DataLoader(
            _dataloaders[2].dataset, batch_size=_dataloaders[2].batch_size, shuffle=False, collate_fn=DynData.collate)

        self.dtype = tm.dtype
        self.t = torch.tensor(tm.t[0])

        self._trans_x = tm._data_transform_x
        self._trans_u = tm._data_transform_u

    def encode(self, X: np.ndarray, rng: Optional[List | None] = None) -> np.ndarray:
        """
        Encode new trajectory data to the observer space.
        """
        _Z = self._trans_x.transform([np.atleast_2d(X)], rng)[0]
        return _Z.squeeze()

    def decode(self, X: np.ndarray, rng: Optional[List | None] = None) -> np.ndarray:
        """
        Decode trajectory data from the observer space.
        """
        _Z = self._trans_x.inverse_transform([np.atleast_2d(X)], rng)[0]
        return _Z.squeeze()

    def apply_obs(self, fobs: Callable) -> np.ndarray:
        """
        Apply a generic observable to the raw data.

        Args:
            fobs (Callable): Observable function. It should accept a 2D array input with each row as one step.
                             The output should be a 1D array, whose ith entry corresponds to the ith step.
        """
        F = []
        for batch in self.train_loader:
            B = batch.x.cpu().numpy()[..., :-1, :]        # This is already transformed
            B = B.reshape(-1, B.shape[-1])
            end = self.NT-1 if self.has_model else self.NT
            B = self._trans_x.inverse_transform([B], [0, end])[0]   # A hack to get back to the original space
            F.append(fobs(B))
        return np.hstack(F)

    def get_forward_modes(self, ref=None, rng: Union[List, None] = None, **kwargs) -> np.ndarray:
        return self._trans_x.get_forward_modes(ref, rng, **kwargs)

    def get_backward_modes(self, ref=None, rng: Union[List, None] = None, **kwargs) -> np.ndarray:
        return self._trans_x.get_backward_modes(ref, rng, **kwargs)
