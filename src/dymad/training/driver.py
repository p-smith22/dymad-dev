from typing import Any, Dict, Iterable, List, Tuple, Type, Union
import copy
import logging
import numpy as np
import os
import torch

from dymad.io import TrajectoryManager, TrajectoryManagerGraph
from dymad.training.helper import CVResult, iter_param_grid, RunState, set_by_dotted_key
from dymad.training.stacked_opt import StackedOpt
from dymad.utils import load_config

logger = logging.getLogger(__name__)

class DriverBase:
    """
    Base driver: loops over (parameter combos x folds) and calls the optimizer.
    """

    def __init__(
        self,
        config_path: str,
        model_class: Type[torch.nn.Module],
        config_mod: Dict[str, Any] | None = None,
        device: torch.device | None = None,
    ):
        self.base_config = load_config(config_path, config_mod)
        self.model_class = model_class
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cv_config = self.base_config.get("cv", {})
        self.param_grid = self.cv_config.get("param_grid", None)   # None = single combo
        self.metric = self.cv_config.get("metric", "total")

        # Setup paths
        self.base_name = self.base_config['model']['name']
        _dir = os.path.dirname(config_path)
        _dir = '.' if _dir == '' else _dir
        os.makedirs(f'{_dir}/checkpoints', exist_ok=True)
        self.checkpoint_prefix = f'{_dir}/checkpoints/{self.base_name}'
        os.makedirs(f'{_dir}/results', exist_ok=True)
        self.results_prefix = f'{_dir}/results/{self.base_name}'

        # Initialize data sets
        self._init_trajectory_managers()
        self._init_fold_split()

    # -------- Abstract API to be implemented by subclasses --------

    def _init_trajectory_managers(self):
        """
        Depending on how folds are defined, create TrajectoryManager(s) for data loading.
        """
        raise NotImplementedError

    def _init_fold_split(self):
        """
        Determine how to split data into folds.
        """
        raise NotImplementedError

    def iter_folds(self) -> Iterable[Tuple[int, Dict[str, Any]]]:
        """
        Yield (fold_id, fold_config) pairs.

        fold_config is a *full* config dict (deep copy of base_config with
        fold-specific overrides, e.g. split_seed).
        """
        raise NotImplementedError

    # -------- Public API --------

    def train(self) -> Tuple[int, CVResult, List[CVResult]]:
        """
        Core loop over hyperparameter and folds combinations.

        Returns:
          best_result, all_results
        """
        all_results: List[CVResult] = []

        # empty grid => treat as single combo with no overrides
        if self.param_grid is None:
            combos = [ {} ]
        else:
            combos = list(iter_param_grid(self.param_grid))

        for combo_idx, combo in enumerate(combos):
            logger.info(f"=== Hyperparam combo {combo_idx+1}/{len(combos)}: {combo} ===")
            fold_metrics: List[float] = []

            for fold_id, fold_cfg in self.iter_folds():
                logger.info(f"--- Fold {fold_id} ---")

                # Apply hyperparameter overrides to this fold's config
                cfg = self._apply_combo_to_config(combo_idx, fold_id, fold_cfg, combo)

                # Build data-only RunState per fold+combo
                data_state = self._build_data_state(fold_id, cfg)

                # Run the optimizer with this config and data state
                opt = StackedOpt(
                    config=cfg,
                    model_class=self.model_class,
                    device=self.device,
                    dtype=self.dtype,
                )
                results = opt.run(initial_state=data_state)

                metric_value = results[-1].run_state.get_metric(self.metric)
                fold_metrics.append(metric_value)

                logger.info(f"Combo {combo_idx+1}, fold {fold_id}: {self.metric} = {metric_value:.6f}")

            mean_metric = float(np.mean(fold_metrics))
            std_metric = float(np.std(fold_metrics))
            logger.info(f"Combo {combo_idx+1}: mean {self.metric} = {mean_metric:.6f} ± {std_metric:.6f}")

            all_results.append(CVResult(combo, fold_metrics, mean_metric, std_metric))

        # Select best (assume lower is better; you can generalize if needed)
        best_idx = int(np.argmin([r.mean_metric for r in all_results]))
        best_result = all_results[best_idx]
        logger.info(f"Best combo: {best_result.params} with {self.metric} = {best_result.mean_metric:.6f}")

        return best_idx, best_result, all_results

    # -------- helpers --------

    def _create_trajectory_manager(self, data_key: str) -> Union[TrajectoryManager, TrajectoryManagerGraph]:
        md = {'config': copy.deepcopy(self.base_config)}
        if self.model_class.GRAPH:
            tm = TrajectoryManagerGraph(md, data_key=data_key, device=self.device)
        else:
            tm = TrajectoryManager(md, data_key=data_key, device=self.device)
        tm.prepare_data()
        return tm

    def _build_data_state(self, fold_id: int, cfg: Dict[str, Any]) -> RunState:
        """Setup data loaders and datasets."""
        # Config can contain different data transforms
        # So we need to update the datasets accordingly
        # The data transforms in valid set should be determined by train set
        trainset: TrajectoryManager | TrajectoryManagerGraph = self.train_sets[fold_id]
        trainset.update_config(cfg)
        train_loader, train_set, train_md = trainset.process_data()

        validset: TrajectoryManager | TrajectoryManagerGraph = self.valid_sets[fold_id]
        validset.update_config(cfg)
        validset.set_transforms(trajmgr=trainset)
        valid_loader, valid_set, valid_md = validset.process_data()

        self.dtype = trainset.dtype
        return RunState(
            config=cfg,
            device=self.device,
            train_loader=train_loader,
            valid_loader=valid_loader,
            train_set=train_set,
            valid_set=valid_set,
            train_md=train_md,
            valid_md=valid_md,
        )

    def _apply_combo_to_config(self, combo_idx, fold_id: int, cfg: Dict[str, Any], combo: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply dotted-key hyperparameters in combo onto a deep-copied config.
        """
        cfg = copy.deepcopy(cfg)
        for dotted_key, value in combo.items():
            set_by_dotted_key(cfg, dotted_key, value)
        _suffix = f"_c{combo_idx}_f{fold_id}"
        cfg["model"]["name"] = f"{self.base_name}{_suffix}"
        cfg.update({
            "path" : {
                "checkpoint_prefix": f"{self.checkpoint_prefix}{_suffix}",
                "results_prefix": f"{self.results_prefix}{_suffix}"
            }})
        return cfg


class KFoldDriver(DriverBase):
    def __init__(
        self,
        config_path: str,
        model_class: Type[torch.nn.Module],
        k_folds: int = 5,
        base_seed: int = 123,
        config_mod: Dict[str, Any] | None = None,
        device: torch.device | None = None,
    ):
        super().__init__(
            config_path=config_path,
            model_class=model_class,
            config_mod=config_mod,
            device=device,
        )
        self.k_folds = k_folds
        self.base_seed = base_seed

    def iter_folds(self):
        """
        For fold i, set data.split_seed = base_seed + i and yield the config.
        """
        for i in range(self.k_folds):
            fold_cfg = copy.deepcopy(self.base_config)
            split_seed = self.base_seed + i
            set_by_dotted_key(fold_cfg, "data.split_seed", split_seed)
            yield i, fold_cfg


class SingleSplitDriver(DriverBase):
    """
    Single fixed split; can still scan param_grid.

    Extreme case:
      - schedule has only one phase,
      - param_grid empty or singleton,
    => this is just “one trainer of one phase.”
    """

    def __init__(
        self,
        config_path: str,
        model_class: Type[torch.nn.Module],
        config_mod: Dict[str, Any] | None = None,
        device: torch.device | None = None,
    ):
        super().__init__(
            config_path=config_path,
            model_class=model_class,
            config_mod=config_mod,
            device=device,
        )

    def iter_folds(self):
        # Just one “fold 0” with the base config (or enforce a specific split_seed)
        fold_cfg = copy.deepcopy(self.base_config)
        if "split_seed" not in fold_cfg.get("data", {}):
            set_by_dotted_key(fold_cfg, "data.split_seed", 0)
        yield 0, fold_cfg

    def _init_trajectory_managers(self):
        assert 'data' in self.base_config, "Config must contain 'data' section."
        if 'data_valid' in self.base_config:
            # A separate validation dataset is specified
            # This is necessary esp when valid set format is different from train set
            self.train_sets = [self._create_trajectory_manager(data_key='train')]
            self.valid_sets = [self._create_trajectory_manager(data_key='valid')]
        else:
            # The same dataset is used for training and validation
            # We will adjust later
            self.train_sets = [self._create_trajectory_manager(data_key='train')]
            self.valid_sets = [self._create_trajectory_manager(data_key='train')]

    def _init_fold_split(self):
        """
        Split the dataset into training and validation sets, if not done.

        The training fraction is specified in the YAML config (default 0.75).
        The split is performed by shuffling whole trajectories.
        """
        if 'data_valid' in self.base_config:
            # A separate validation dataset is specified
            # No need to split
            return
        
        # Otherwise, split the training dataset into train/valid
        split_cfg = self.base_config.get("split", {})
        train_frac = split_cfg.get("train_frac", 0.75)
        assert 0.0 < train_frac < 1.0, f"train_frac must be in (0, 1). Got: {train_frac}"

        n_samples = self.train_sets[0].metadata['n_samples']
        n_train = int(n_samples * train_frac)
        n_val = n_samples - n_train
        assert n_train > 0, f"Training set must have at least one sample. Got {n_train}."
        assert n_val > 0, f"Validation set must have at least one sample. Got {n_val}."

        perm = torch.randperm(n_samples)
        self.train_set_index = perm[:n_train]
        self.valid_set_index = perm[n_train:]

        self.train_sets[0].set_data_index(self.train_set_index)
        self.valid_sets[0].set_data_index(self.valid_set_index)
