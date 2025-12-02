from dataclasses import dataclass, field
from itertools import product
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, Iterable, List, Optional

@dataclass
class RunState:
    # Persistent
    config: Optional[Dict[str, Any]]
    device: Optional[torch.device] = None
    epoch: int = 0
    best_loss: float = float("inf")
    hist: List[Any] = field(default_factory=list)
    rmse: List[Any] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    converged: bool = False

    # Model & optim
    model: Optional[torch.nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    schedulers: List[Any] = field(default_factory=list)
    criterion: Optional[torch.nn.Module] = None

    # Data: live objects only (not serialized)
    train_set: Optional[Dataset] = None
    valid_set: Optional[Dataset] = None
    train_loader: Optional[DataLoader] = None
    valid_loader: Optional[DataLoader] = None
    train_md: Optional[Dict[str, Any]] = None
    valid_md: Optional[Dict[str, Any]] = None

    def to_checkpoint(self) -> Dict[str, Any]:
        """Serialize the persistent subset of state."""
        return {
            "config": self.config,
            "device": self.device,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "hist": self.hist,
            "rmse": self.rmse,
            "epoch_times": self.epoch_times,
            "converged": self.converged,
            "model_state_dict": None if self.model is None else self.model.state_dict(),
            "optimizer_state_dict": None if self.optimizer is None else self.optimizer.state_dict(),
            "scheduler_state_dicts": [
                scheduler.state_dict() for scheduler in self.schedulers if hasattr(scheduler, "state_dict")
            ],
            "criterion_state_dict": None if self.criterion is None else self.criterion.state_dict(),
            "train_md": self.train_md,
            "valid_md": self.valid_md,
        }

    @classmethod
    def from_checkpoint(
        cls,
        ckpt: Dict[str, Any],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        schedulers: List[Any],
        criterion: Optional[torch.nn.Module],
    ) -> "RunState":
        """Rebuild RunState from a checkpoint, plus new live model/optimizer/schedulers."""
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Restore schedulers if you stored them
        if "scheduler_state_dicts" in ckpt:
            for s, s_state in zip(schedulers, ckpt["scheduler_state_dicts"]):
                if hasattr(s, "load_state_dict"):
                    s.load_state_dict(s_state)
        criterion.load_state_dict(ckpt["criterion_state_dict"])

        return cls(
            config=ckpt.get("config", {}),
            epoch=ckpt.get("epoch", 0),
            best_loss=ckpt.get("best_loss", float("inf")),
            hist=ckpt.get("hist", []),
            rmse=ckpt.get("rmse", []),
            epoch_times=ckpt.get("epoch_times", []),
            converged=ckpt.get("converged", False),
            model=model,
            optimizer=optimizer,
            schedulers=schedulers,
            criterion=criterion,
            train_md=ckpt.get("train_md", {}),
            valid_md=ckpt.get("valid_md", {}),
        )


@dataclass
class CVResult:
    params: Dict[str, Any]
    fold_metrics: List[float]
    mean_metric: float
    std_metric: float


def iter_param_grid(param_grid: Dict[str, Iterable[Any]]):
    """
    param_grid: dict mapping dotted keys to iterables.
    Yields dicts mapping dotted keys -> single value.
    """
    keys = list(param_grid.keys())
    values_lists = [param_grid[k] for k in keys]
    for values in product(*values_lists):
        yield dict(zip(keys, values))


def set_by_dotted_key(d: Dict[str, Any], dotted_key: str, value: Any):
    """
    Set d['a']['b']['c'] when dotted_key is 'a.b.c'.
    Creates intermediate dicts as needed.
    """
    parts = dotted_key.split(".")
    curr = d
    for p in parts[:-1]:
        if p not in curr or not isinstance(curr[p], dict):
            curr[p] = {}
        curr = curr[p]
    curr[parts[-1]] = value
