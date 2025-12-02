import copy
import logging
import numpy as np
import os
import random
import time
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict, Optional, Tuple, Type, Union

from dymad.io import DynData
from dymad.training.helper import RunState
from dymad.training.ls_update import LSUpdater
from dymad.utils import make_scheduler, plot_hist, plot_trajectory

logger = logging.getLogger(__name__)

class OptBase:
    """
    Base Opt class: owns data, model, optimizer, schedulers, training
    loop, checkpointing, history, LSUpdater, and RunState.

    Inderited Opt classes (NODE, WF, LR, ...) should:
      - implement `_process_batch(batch)` (may return Tensor or dict of losses),
      - optionally customize model / optimizer / schedulers via config.

    Args:
        config_path (str): Path to the YAML configuration file
        model_class (Type[torch.nn.Module]): Class of the model to train
        config_mod (Optional[Dict]): Modifications to the config
        run_state (Optional[RunState]): Existing RunState to attach (data-only or full)
    """
    def __init__(
        self,
        config: Dict[str, Any],
        config_phase: Dict[str, Any],
        model_class: Type[torch.nn.Module],
        run_state: RunState,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.config = copy.deepcopy(config)
        self.config_phase = copy.deepcopy(config_phase)
        self.model_class = model_class
        self.device = device
        self.dtype = dtype

        self.convergence_tolerance_reached = False

        # Loss weights (for multi-loss aggregation)
        # Example YAML:
        # training:
        #   loss_weights:
        #     dynamics: 1.0
        #     recon: 0.5
        #     physics: 10.0
        self.loss_weights: Dict[str, float] = self.config_phase.get(
            "loss_weights", {}
        )

        # Setup paths
        self.model_name = self.config["model"]["name"]
        self.checkpoint_path = self.config["path"]["checkpoint_prefix"] + "_checkpoint.pt"
        self.best_model_path = self.config["path"]["checkpoint_prefix"] + ".pt"
        os.makedirs(self.config["path"]["results_prefix"], exist_ok=True)
        self.results_prefix  = self.config["path"]["results_prefix"]

        ifloaded = self.load_checkpoint()
        if not ifloaded:
            if run_state is None:
                raise ValueError("No checkpoint found and no run_state provided.")
            # Attach an existing run state (may be data-only or full)
            self._attach_run_state(run_state)

        # ---- logging summary ----
        logger.info("Opt Initialized:")
        logger.info(f"Model name: {self.model_name}")
        logger.info(self.model)
        logger.info(self.model.diagnostic_info())
        logger.info("Optimization settings:")
        logger.info(self.optimizer)
        logger.info(self.criterion)
        logger.info("Scheduler info:")
        for _s in self.schedulers:
            logger.info(_s.diagnostic_info())
        logger.info(f"Using device: {self.device}")
        logger.info(f"Double precision: {self.config['data'].get('double_precision', False)}")
        logger.info(
            f"Epochs: {self.config_phase['n_epochs']}, "
            f"Save interval: {self.config_phase['save_interval']}"
        )

    def _setup_model(self) -> None:
        """Setup model, optimizer, schedulers, and criterion."""
        self.model = self.model_class(
            self.config["model"], self.train_md, dtype=self.dtype, device=self.device
        ).to(self.device)

        if self.config["data"].get("double_precision", False):
            self.model = self.model.double()

        # By default there is only one scheduler.
        # There might be more, e.g., in OptNODE with sweep scheduler.
        lr = float(self.config_phase.get("learning_rate", 1e-3))
        gamma = float(self.config_phase.get("decay_rate", 0.999))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.schedulers = [make_scheduler(
            torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        )]
        self.criterion = torch.nn.MSELoss(reduction="mean")

    def _setup_ls(self) -> None:
        """Setup LSUpdater if requested."""
        if self.config_phase.get("ls_update", False):
            cfg = self.config_phase["ls_update"]
            self._ls = LSUpdater(
                method=cfg.get("method", "full"),
                model=self.model,
                dt=self.train_md["dt_and_n_steps"][0][0],
                params=cfg.get("params", None),
                **cfg.get("kwargs", {}),
            )
            self._ls_update_interval = cfg.get("update_interval", 1)
            self._ls_update_times    = cfg.get("update_times", 1)
            self._ls_reset           = cfg.get("reset_optimizer", True)
            self._start_w_ls         = cfg.get("start_with_ls", True)
            self._param_to_name      = {
                param: name for name, param in self.model.named_parameters()
            }
        else:
            self._ls = None
            self._ls_update_interval = 0
            self._ls_update_times    = 0
            self._ls_reset           = False
            self._param_to_name      = {}
            self._start_w_ls         = False

    # ------------------------------------------------------------------
    # RunState attach / export
    # ------------------------------------------------------------------

    def _attach_run_state(self, state: RunState) -> None:
        """
        Attach an existing RunState.

        Cases:
          - data-only: model/optimizer/schedulers None -> we set them up fresh.
          - full: everything present -> we reuse all.
        """
        # Data
        self.train_set = state.train_set
        self.valid_set = state.valid_set
        self.train_loader = state.train_loader
        self.valid_loader = state.valid_loader
        self.train_md = state.train_md     # Only need dimension info
        self.valid_md = state.valid_md

        # If model is None, this is "data-only" state: build model from scratch
        if state.model is None or state.optimizer is None or not state.schedulers:
            self._setup_model()

            self.start_epoch = 0
            self.best_loss = float("inf")
            self.hist = []
            self.rmse = []
            self.epoch_times = []
        else:
            self.model = state.model.to(self.device)
            self.optimizer = state.optimizer
            self.schedulers = state.schedulers
            self.criterion = state.criterion

            self.start_epoch = state.epoch
            self.best_loss = state.best_loss
            self.hist = list(state.hist)
            self.rmse = list(state.rmse)
            self.epoch_times = []
        self._setup_ls()

    def export_run_state(self, epoch: int) -> RunState:
        """Package the current trainer state into a RunState."""
        return RunState(
            config=self.config,
            device=self.device,
            epoch=epoch,
            best_loss=self.best_loss,
            hist=list(self.hist),
            rmse=list(self.rmse),
            epoch_times=list(self.epoch_times),
            converged=self.convergence_tolerance_reached,
            model=self.model,
            optimizer=self.optimizer,
            schedulers=self.schedulers,
            criterion=self.criterion,
            train_set=self.train_set,
            valid_set=self.valid_set,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            train_md=self.train_md,
            valid_md=self.valid_md,
        )

    # ------------------------------------------------------------------
    # Checkpoint I/O (via RunState)
    # ------------------------------------------------------------------

    def load_checkpoint(self) -> None:
        """If checkpoint exists and config requires, load it into the current model/optimizer/schedulers."""
        checkpoint_path = None
        load_from_checkpoint = self.config_phase.get('load_checkpoint', False)
        if isinstance(load_from_checkpoint, str):
            checkpoint_path = load_from_checkpoint
        elif load_from_checkpoint:
            checkpoint_path = self.checkpoint_path

        flag = True
        if checkpoint_path is None:
            logger.info(f"Got load_from_checkpoint={load_from_checkpoint}, resulting in checkpoint_path=None. Starting from scratch.")
            flag = False
        elif not os.path.exists(checkpoint_path):
            logger.info(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
            flag = False

        if not flag:
            self.start_epoch = 0
            self.best_loss = float("inf")
            self.hist = []
            self.rmse = []
            self.epoch_times = []
            return flag

        ckpt = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)
        state = RunState.from_checkpoint(ckpt, self.model, self.optimizer, self.schedulers, self.criterion)

        # Attach the persistent parts
        self.start_epoch = state.epoch + 1  # resume from next epoch
        self.best_loss = state.best_loss
        self.hist = state.hist
        self.rmse = state.rmse
        self.convergence_tolerance_reached = False  # reset on load
        self.epoch_times = state.epoch_times

        logger.info(f"Loaded checkpoint from {self.checkpoint_path}, starting at epoch {self.start_epoch}")
        return flag

    def save_checkpoint(self, epoch: int, path: Optional[str] = None) -> None:
        """Save checkpoint using current RunState."""
        state = self.export_run_state(epoch)
        ckpt = state.to_checkpoint()
        torch.save(ckpt, self.checkpoint_path if path is None else path)

    def save_if_best(self, val_loss: float, epoch: int) -> None:
        """Save best model if validation loss improves."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.convergence_epoch = epoch+1
            self.save_checkpoint(epoch, self.best_model_path)
            logger.info(f"New best model at epoch {epoch}, val_loss={val_loss:.4e}")
            return True
        return False

    # ------------------------------------------------------------------
    # Multi-loss: `_process_batch` + aggregation
    # ------------------------------------------------------------------

    def _process_batch(self, batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss terms for a batch.

        Return either:
          - a single Tensor (total loss), or
          - a dict name -> Tensor; we aggregate using `loss_weights`.
            If the dict contains key 'total', we treat that as the final loss
            and ignore weights.
        """
        raise NotImplementedError

    def _aggregate_losses(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combine named loss terms using config-defined weights.

        - If 'total' in loss_dict: we directly return it.
        - Else: sum w_i * loss_i, where w_i = loss_weights.get(name, 1.0).
        """
        if "total" in loss_dict:
            return loss_dict["total"]

        total = 0.0
        for name, term in loss_dict.items():
            if not torch.is_tensor(term):
                continue
            w = self.loss_weights.get(name, 1.0)
            total = total + w * term
        return total

    def _compute_total_loss(
        self, out: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Given the output of `_process_batch`, compute:
          - total loss Tensor
          - dict of scalar loss terms (for logging if desired)
        """
        if isinstance(out, torch.Tensor):
            loss = out
            terms = {"total": float(loss.detach().cpu())}
            return loss, terms

        # dict case
        loss = self._aggregate_losses(out)
        terms: Dict[str, float] = {}
        for name, term in out.items():
            if torch.is_tensor(term):
                terms[name] = float(term.detach().cpu())
        # also store total
        terms.setdefault("total", float(loss.detach().cpu()))
        return loss, terms

    # ------------------------------------------------------------------
    # Generic training engine
    # ------------------------------------------------------------------

    def train_epoch(self) -> float:
        """
        Generic training loop for one epoch.

        Uses `_process_batch` + multi-loss aggregation.
        Handles:
          - optimizer step,
          - scheduler step(eploss),
          - convergence flag,
          - minimum learning rate.
        """
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            out = self._process_batch(batch)
            loss, _ = self._compute_total_loss(out)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_epoch_loss = total_loss / len(self.train_loader)

        # Step schedulers
        for scheduler in self.schedulers:
            flag, changed = scheduler.step(eploss=avg_epoch_loss)
            self.convergence_tolerance_reached = (
                self.convergence_tolerance_reached or flag
            )
            if changed:
                self.best_loss = float("inf")
                logger.info("Resetting best loss due to scheduler change.")

        # Enforce minimum LR
        min_lr = float(self.config_phase.get("min_learning_rate", 1e-6))
        if min_lr > 0.0:
            for param_group in self.optimizer.param_groups:
                if param_group["lr"] < min_lr:
                    param_group["lr"] = min_lr

        return avg_epoch_loss

    def evaluate(self, dataloader: DataLoader) -> float:
        """Generic evaluation loop using `_process_batch`."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                out = self._process_batch(batch)
                loss, _ = self._compute_total_loss(out)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    # ------------------------------------------------------------------
    # Criteria and main training loop
    # ------------------------------------------------------------------
    def evaluate_criteria_single_traj(self,
                    truth: DynData,
                    method: str = 'dopri5',
                    plot: bool = False) -> float:
        """
        Calculate RMSE between model predictions and ground truth for regular models

        Args:
            truth (DynData): Ground truth trajectory data
            method (str): ODE solver method (for models that use ODE solvers)
            plot (bool): Whether to plot the predicted vs ground truth trajectories

        Returns:
            float: Root mean squared error between predictions and ground truth
        """
        with torch.no_grad():
            # Extract states and controls
            x_truth = truth.x
            x0 = truth.x[:, 0, :]
            ts = truth.t

            # Make prediction
            x_pred = self.model.predict(x0, truth, ts, method=method)

            x_truth = x_truth.detach().cpu().numpy().squeeze(0)
            x_pred = x_pred.detach().cpu().numpy().squeeze(0)
            # Calculate RMSE
            rmse = np.sqrt(np.mean((x_pred - x_truth)**2))

            if plot:
                _us = None if truth.u is None else truth.u.detach().cpu().numpy().squeeze(0)
                plotting_config = self.config.get('plotting', {})
                plot_trajectory(np.array([x_truth, x_pred]), ts.squeeze(0), self.model_name,
                                us=_us, labels=['Truth', 'Prediction'], prefix=self.results_prefix,
                                **plotting_config)

            return rmse

    def evaluate_criteria(self, split: str = 'test', plot: bool = False, evaluate_all: bool = False) -> float:
        """
        Calculate criteria on trajectory(ies) from the specified split.

        Args:
            split (str): Dataset split to use ('train', 'valid')
            plot (bool): Whether to plot the results (only works when evaluate_all=False)
            evaluate_all (bool):

                - If True, evaluate all trajectories and return mean RMSE.
                - If False, evaluate a single random trajectory.

        Returns:
            float: Criteria value (mean criteria if evaluate_all=True)
        """
        if split == "train":
            dataset = self.train_set
        elif split == "valid":
            dataset = self.valid_set
        else:
            raise ValueError(f"Unknown split: {split}")

        if evaluate_all:
            plot = False
            dataset = dataset
        else:
            dataset = [random.choice(dataset)]

        _method = self.config.get("training", {}).get("ode_method", "dopri5")
        criteria_values = [
            self.evaluate_criteria_single_traj(trajectory, method=_method, plot=plot)
            for trajectory in dataset
        ]

        return sum(criteria_values) / len(criteria_values)

    def train(self) -> int:
        """Run full training loop."""

        n_epochs = self.config_phase['n_epochs']
        save_interval = self.config_phase['save_interval']

        self.convergence_epoch = None
        self.epoch_times = []

        _ls_count = 0

        overall_start_time = time.time()
        for epoch in range(self.start_epoch, self.start_epoch + n_epochs):
            # Training and evaluation
            # Only timing the train and validation phases
            # since test loss is only for reference
            epoch_start_time = time.time()

            # Partial LS update if specified
            if self._ls is not None:
                if epoch % self._ls_update_interval == 0:
                    if epoch == self.start_epoch and not self._start_w_ls:
                        logger.info("Skipping LS update at the start epoch as per configuration.")
                    else:
                        _ls_count += 1
                        if _ls_count == self._ls_update_times+1:
                            logger.info(f"LS update performed {self._ls_update_times} times, stopping further LS updates.")
                        elif _ls_count <= self._ls_update_times:
                            logger.info(f"Performing LS update {_ls_count} of {self._ls_update_times} times at epoch {epoch+1}")
                            _, params = self._ls.update(self.model, self.train_loader)

                            # Remove optimizer state for parameters updated by LS
                            if self._ls_reset:
                                target_names = [self._param_to_name.get(p, "<unnamed>") for p in params]
                                for _p, _n in zip(params, target_names):
                                    for param_group in self.optimizer.param_groups:
                                        param_names = [self._param_to_name.get(_q, "<unnamed>") for _q in param_group['params']]
                                        if _n in param_names:
                                            self.optimizer.state.pop(_p, None)
                                            logger.info(f"Removed optimizer state for {_n} after LS update.")

            train_loss = self.train_epoch()
            val_loss = self.evaluate(self.valid_loader)

            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)

            # Record history
            self.hist.append([epoch, train_loss, val_loss])

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.start_epoch + n_epochs}, "
                f"Train Loss: {train_loss:.4e}, "
                f"Validation Loss: {val_loss:.4e}"
            )

            # Save best model
            self.save_if_best(val_loss, epoch)

            # Periodic checkpoint and evaluation
            if (epoch + 1) % save_interval == 0 or self.convergence_tolerance_reached:
                self.save_checkpoint(epoch)

                # Plot loss curves
                plot_hist(self.hist, epoch+1, self.model_name, prefix=self.results_prefix)

                # Evaluate RMSE on random trajectories
                train_rmse = self.evaluate_criteria('train', plot=False)
                valid_rmse = self.evaluate_criteria('valid', plot=True)
                self.rmse.append([epoch, train_rmse, valid_rmse])

                logger.info(
                    f"Prediction RMSE - "
                    f"Train: {train_rmse:.4e}, "
                    f"Valid: {valid_rmse:.4e}"
                )

                if self.convergence_tolerance_reached:
                    self.convergence_epoch = epoch+1
                    logger.info(f"Convergence reached at epoch {epoch+1} "
                                f"with validation loss {val_loss:.4e}")
                    break

        if self.rmse == []:
            train_rmse = self.evaluate_criteria('train', plot=False)
            valid_rmse = self.evaluate_criteria('valid', plot=True)
            self.rmse.append([epoch, train_rmse, valid_rmse])

        plot_hist(self.hist, epoch+1, self.model_name, prefix=self.results_prefix)
        total_training_time = time.time() - overall_start_time
        avg_epoch_time = np.mean(self.epoch_times)
        final_train_loss = self.evaluate(self.train_loader)
        final_valid_loss = self.evaluate(self.valid_loader)
        _ = self.evaluate_criteria('valid', plot=True)

        # Process histories of loss and RMSE
        # These are saved in the checkpoint too, but here we process them for easier post-processing
        tmp = np.array(self.hist).T
        epoch_loss, losses = tmp[0], tmp[1:]
        tmp = np.array(self.rmse).T
        epoch_rmse, rmses = tmp[0], tmp[1:]

        # Save summary of training
        # Here we also save the model itself - a lazy approach but more "out-of-the-box" for deployment
        results = {
            'model_name': self.model_name,
            'total_training_time': total_training_time,
            'avg_epoch_time': avg_epoch_time,
            'final_train_loss': final_train_loss,
            'final_valid_loss': final_valid_loss,
            'best_valid_loss': self.best_loss,
            'convergence_epoch': self.convergence_epoch,
            'epoch_loss': epoch_loss,
            'losses': losses,
            'epoch_rmse': epoch_rmse,
            'rmses': rmses,
        }

        file_name = f"{self.results_prefix}/{self.model_name}_summary.npz"
        np.savez_compressed(file_name, **results)
        logger.info("Training complete. Summary of training:")
        for key in [
            "model_name",
            "total_training_time",
            "avg_epoch_time",
            "final_train_loss",
            "final_valid_loss",
            "best_valid_loss",
            "convergence_epoch",
        ]:
            info = f"{results[key]:.4e}" if isinstance(results[key], float) else str(results[key])
            logger.info(f"{key}: {info}")
        logger.info(f"Summary and loss/rmse histories saved to {file_name}")

        return epoch