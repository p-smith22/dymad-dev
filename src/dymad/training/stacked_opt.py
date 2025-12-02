from typing import Dict, Any, Type
import copy
import logging
import torch

from dymad.training.helper import RunState
from dymad.training.opt_base import OptBase
from dymad.training.opt_node import OptNODE
from dymad.training.opt_weak_form import OptWeakForm
# from dymad.training.lr_trainer import LRTrainer, etc.

logger = logging.getLogger(__name__)

OPT_REGISTRY: Dict[str, Type[OptBase]] = {
    "NODE": OptNODE,
    "Weak": OptWeakForm,
    # "LinearRegression": LRTrainer,
}

class PhaseResult:
    def __init__(self, name: str, run_state: RunState, hist):
        self.name = name
        self.run_state = run_state
        self.hist = hist

class StackedOpt:
    """
    Stack multiple optimization phases (e.g., WF -> NODE -> LR)
    on a (potentially precomputed) RunState.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_class: Type,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.config = copy.deepcopy(config)
        self.model_class = model_class
        self.device = device
        self.dtype = dtype

        self.phases = copy.deepcopy(self.config.get("phases", []))
        if not self.phases:
            raise ValueError("Experiment config must contain a non-empty 'phases' list.")

    def run(self, initial_state: RunState) -> Dict[str, PhaseResult]:
        """
        Run all prescribed phases in order.

        initial_state:
          - None          => first trainer will build data/model itself (legacy/simple usage).
          - data-only     => first trainer reuses data, builds model.
          - full RunState => continuation: reuse data + model + optimizer, etc.
        """
        results = {}
        current_state = initial_state

        for i, phase_cfg in enumerate(self.phases):
            phase_name  = phase_cfg.get("name", f"phase_{i}")
            trainer_key = phase_cfg["trainer"]
            trainer_cls = OPT_REGISTRY[trainer_key]

            logger.info(f"=== Starting phase '{phase_name}' with trainer '{trainer_key}' ===")

            # Instantiate trainer; it will attach to provided RunState (data-only or full).
            trainer = trainer_cls(
                config=self.config,
                config_phase=phase_cfg,
                model_class=self.model_class,
                run_state=current_state,
                device=self.device,
                dtype=self.dtype,
            )

            # Run this phase
            epoch = trainer.train()

            # Export state for the next phase
            current_state = trainer.export_run_state(epoch)
            results[phase_name] = PhaseResult(
                name=phase_name,
                run_state=current_state,
                hist=trainer.hist,
            )

        return results
