from typing import Dict, Any, Type, List, Tuple, Iterable
import copy
import logging
import numpy as np
import os
import torch

from dymad.io import TrajectoryManager, TrajectoryManagerGraph
from dymad.training.driver import SingleSplitDriver
from dymad.training.helper import CVResult, iter_param_grid, RunState, set_by_dotted_key
from dymad.training.stacked_opt import StackedOpt
from dymad.utils import load_config

logger = logging.getLogger(__name__)

class NODETrainer(SingleSplitDriver):
    """
    Simple interface for single-split single-stage training by NODE.
    """
    def __init__(
        self,
        config_path: str,
        model_class: Type[torch.nn.Module],
        config_mod: Dict[str, Any] | None = None,
        param_grid: Dict[str, Iterable[Any]] | None = None,
        metric: str = "val_loss",
        device: torch.device | None = None
        ):
        super().__init__(
            config_path=config_path,
            model_class=model_class,
            config_mod=config_mod,
            param_grid=param_grid,
            metric=metric,
            device=device,
        )

        # By default, we don't specify the phases
        # So we manually create a single NODE phase here
        cfg = copy.deepcopy(self.base_config["training"])
        del self.base_config["training"]
        cfg.update({"name": "NODE", "trainer": "NODE"})
        self.base_config.update({"phases": [cfg]})


class WeakFormTrainer(SingleSplitDriver):
    """
    Simple interface for single-split single-stage training by Weak Form.
    """
    def __init__(
        self,
        config_path: str,
        model_class: Type[torch.nn.Module],
        config_mod: Dict[str, Any] | None = None,
        param_grid: Dict[str, Iterable[Any]] | None = None,
        metric: str = "val_loss",
        device: torch.device | None = None
        ):
        super().__init__(
            config_path=config_path,
            model_class=model_class,
            config_mod=config_mod,
            param_grid=param_grid,
            metric=metric,
            device=device,
        )

        # By default, we don't specify the phases
        # So we manually create a single NODE phase here
        cfg = copy.deepcopy(self.base_config["training"])
        del self.base_config["training"]
        cfg.update({"name": "WeakForm", "trainer": "Weak"})
        self.base_config.update({"phases": [cfg]})


class LinearTrainer(SingleSplitDriver):
    """
    Simple interface for single-split single-stage training by Linear regression.
    """
    def __init__(
        self,
        config_path: str,
        model_class: Type[torch.nn.Module],
        config_mod: Dict[str, Any] | None = None,
        param_grid: Dict[str, Iterable[Any]] | None = None,
        metric: str = "val_loss",
        device: torch.device | None = None
        ):
        super().__init__(
            config_path=config_path,
            model_class=model_class,
            config_mod=config_mod,
            param_grid=param_grid,
            metric=metric,
            device=device,
        )

        # By default, we don't specify the phases
        # So we manually create a single NODE phase here
        cfg = copy.deepcopy(self.base_config["training"])
        del self.base_config["training"]
        cfg.update({"name": "Linear", "trainer": "Linear"})
        self.base_config.update({"phases": [cfg]})


class StackedTrainer(SingleSplitDriver):
    """
    Simple interface for single-split single-stage training by a sequence of trainers.

    A wrapper of StackedOpt.
    """
    def __init__(
        self,
        config_path: str,
        model_class: Type[torch.nn.Module],
        config_mod: Dict[str, Any] | None = None,
        param_grid: Dict[str, Iterable[Any]] | None = None,
        metric: str = "val_loss",
        device: torch.device | None = None
        ):
        super().__init__(
            config_path=config_path,
            model_class=model_class,
            config_mod=config_mod,
            param_grid=param_grid,
            metric=metric,
            device=device,
        )
