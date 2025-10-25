import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional, Union, Tuple, Dict, List

from dymad.io.data import DynData
from dymad.transform import make_transform
from dymad.utils.graph import adj_to_edge

logger = logging.getLogger(__name__)

def _process_data(data, x, label, base_dim=1, offset=0):
    """
    x as reference data, list of arrays.

    When offset = 1, effectively the n_steps dimension is removed, so the method processes time-invariant data.

    Expecting to return a list of `n_traj` arrays:
    _data = [... d_i ...]
    where d_i has shape (n_steps, ...) and ndim(d_i) = base_dim + 1

    d_i itself can be an array or a list, depending on the input data type.
    """
    _dim = base_dim - offset
    if data is None:
        logging.info(f"No {label} detected. Setting to None.")
        if offset == 0:
            _data = [np.empty((_x.shape[0], 0)) for _x in x]
        else:
            _data = [np.empty((0,)) for _ in x]

    elif isinstance(data, np.ndarray):
        # t,y,u,p should always go through here, by converting to np.ndarray beforehand.
        # ei/ew/ea could go through this branch too, if they are in np.ndarray form;
        # this would mean their shapes are uniform throughout the dataset.
        #
        # The possibilities here:
        # - np.ndarray of shape (n_traj, n_steps, ...)  - full data for multiple trajs
        # - np.ndarray of shape (1, n_steps, ...) - Broadcast to all trajs if needed
        # - np.ndarray of shape (n_traj, 1, ...)  - Broadcast to all steps
        # - np.ndarray of shape (n_steps, ...)    - single traj data, broadcast to all trajs if needed
        # - np.ndarray of shape (...,)            - Broadcast to all steps and trajs
        if data.ndim == _dim + 2:  # (n_traj, n_steps, ...)
            if data.shape[0] == 1 and len(x) > 1:
                logging.info(f"Detected {label} as np.ndarray (1, n_steps, ...): {data.shape} for multiple x. Broadcasting to all trajectories.")
                _data = [np.array(data[0]) for _ in x]
            elif data.shape[1] == 1:
                logging.info(f"Detected {label} as np.ndarray (n_traj, 1, ...): {data.shape}. Expanding to trajectory for each x and broadcasting to all time steps.")
                _data = [np.tile(data[_i], (_x.shape[0],) + (1,) * base_dim) for _i, _x in enumerate(x)]
            else:
                logging.info(f"Detected {label} as np.ndarray (n_traj, n_steps, ...): {data.shape}. Splitting into list of arrays.")
                _data = [np.array(_u) for _u in data]
        elif data.ndim == _dim + 1:  # (n_steps, ...)
            if len(x) > 1:
                logging.info(f"Detected {label} as np.ndarray (n_steps, ...): {data.shape} but x is multi-traj ({len(x)}). Broadcasting {label} to all trajectories.")
                _data = [np.array(data) for _ in x]
            else:
                logging.info(f"Detected {label} as np.ndarray (n_steps, ...): {data.shape}. Wrapping as single-element list.")
                _data = [np.array(data)]
        elif data.ndim == _dim and _dim > 0:  # (...,)
            logging.info(f"Detected {label} as np.ndarray (...,): {data.shape}. Expanding to trajectory for each x and broadcasting to all trajectories.")
            _data = [np.tile(data, (x.shape[0],) + (1,) * base_dim) for x in x]
        else:
            msg = f"Unsupported {label} shape: {data.shape}"
            logging.error(msg)
            raise ValueError(msg)

    elif isinstance(data, list):
        # This branch should be ei/ew/ea that are in lists.
        # an element of np.ndarray is considered one sample at one step in one trajectory,
        # so its ndim should be base_dim.
        #
        # The possibilities here:
        # - list of lists of arrays    - (n_traj, n_steps, ...) - already full data
        # - list of one list of arrays - (1, n_steps, ...) - broadcast to all trajs
        # - list of lists of one array - (n_traj, 1, ...)  - broadcast to all steps
        # - list of arrays             - (n_steps, ...)    - single traj data, broadcast to all trajs
        if isinstance(data[0], np.ndarray):
            if data[0].ndim == _dim:   # (n_steps, ...)
                if len(x) > 1:
                    logging.info(f"Detected {label} as lists (n_steps, ...): {data[0].shape} but x is multi-traj ({len(x)})." \
                                 f"Broadcasting {label} to all trajectories.")
                    _data = [data for _ in x]
                else:
                    logging.info(f"Detected {label} as lists (n_steps, ...): {data[0].shape}. Wrapping as single-element list.")
                    _data = [np.array(data[0])]
            else:
                msg = f"Unsupported {label} array shape in list: {data[0].shape}"
                logging.error(msg)
                raise ValueError(msg)
        elif isinstance(data[0], list):
            if len(data) == 1:         # (1, n_steps, ...)
                if len(x) > 1:
                    logging.info(f"Detected {label} as lists (1, n_steps, ...): {data[0][0].shape} for multiple x. Broadcasting to all trajectories.")
                    _data = [data[0] for _ in x]
                else:
                    logging.info(f"Detected {label} as lists (n_traj, n_steps, ...): {data[0][0].shape}. Return as is.")
                    _data = data
            elif len(data[0]) == 1:    # (n_traj, 1, ...)
                logging.info(f"Detected {label} as lists (n_traj, 1, ...): {data[0][0].shape}. Expanding to trajectory for each x and broadcasting to all time steps.")
                _data = [[data[_i][0] for _ in range(x[_i].shape[0])] for _i in range(len(x))]
            else:                      # (n_traj, n_steps, ...)
                logging.info(f"Detected {label} as lists (n_traj, n_steps, ...): {data[0][0].shape}. Return as is.")
                _data = data

    else:
        logging.error(f"{label} must be a np.ndarray or list of np.ndarrays")
        raise TypeError(f"{label} must be a np.ndarray or list of np.ndarrays")

    # Data validation
    assert len(_data) == len(x), f"{label} list length ({len(_data)}) must match x list length ({len(x)})"
    if _data[0].size > 0 and offset == 0:
        for xi, ui in zip(x, _data):
            if xi.shape[0] != ui.shape[0]:
                msg = f"Each trajectory in x ({xi.shape[0]}) and {label} ({ui.shape[0]}) must have the same number of time steps"
                logging.error(msg)
                raise ValueError(msg)
    return _data

class TrajectoryManager:
    """
    A class to manage trajectory data loading, preprocessing, and
    dataloader creation.

    The workflow includes:

      - Loading raw data from a binary file.
      - Preprocessing (trimming trajectories, subsetting, etc.).
      - Creating a dataset and splitting into train/validation/test sets.
      - Normalizing and transforming the data using specified transformations.
      - Creating dataloaders tailored for NN, LSTM, or GNN models.

    The class is configured via a YAML configuration file.

    Args:
        metadata (dict): Configuration dictionary.
        device (torch.device): Torch device to use.
    """

    def __init__(self, metadata: Dict, device: torch.device = torch.device("cpu")):
        self.metadata = metadata
        self.dtype = torch.double if self.metadata['config']['data'].get('double_precision', False) else torch.float
        self.device = device
        self.data_path = self.metadata['config']['data']['path']

        self._data_transform_x = make_transform(self.metadata['config'].get('transform_x', None))
        self._data_transform_y = make_transform(self.metadata['config'].get('transform_y', None))
        self._data_transform_p = make_transform(self.metadata['config'].get('transform_p', None))
        cfg_transform_u = self.metadata['config'].get('transform_u', None)
        self._data_transform_u = make_transform(cfg_transform_u)
        if cfg_transform_u is None:
            self.metadata["delay"] = self._data_transform_x.delay
        else:
            self.metadata["delay"] = max(self._data_transform_x.delay, self._data_transform_u.delay)

        if "train_set_index" in metadata:
            # If train_set_index is already in metadata, we assume the dataset has been split before.
            assert metadata["n_train"] == len(metadata["train_set_index"])
            assert metadata["n_val"]   == len(metadata["valid_set_index"])
            assert metadata["n_test"]  == len(metadata["test_set_index"])
            logging.info(f"Reusing data split from provided metadata.")

            self.train_set_index = torch.tensor(metadata["train_set_index"], dtype=torch.long)
            self.valid_set_index = torch.tensor(metadata["valid_set_index"], dtype=torch.long)
            self.test_set_index  = torch.tensor(metadata["test_set_index"], dtype=torch.long)

            self.metadata["n_train"] = metadata["n_train"]
            self.metadata["n_val"]   = metadata["n_val"]
            self.metadata["n_test"]  = metadata["n_test"]
            self.metadata["train_set_index"] = self.train_set_index.tolist()
            self.metadata["valid_set_index"] = self.valid_set_index.tolist()
            self.metadata["test_set_index"]  = self.test_set_index.tolist()
            self._data_is_split = True

            self._data_transform_x.load_state_dict(metadata["transform_x_state"])
            if "transform_y_state" in metadata:
                self._data_transform_y.load_state_dict(metadata["transform_y_state"])
            if "transform_u_state" in metadata:
                self._data_transform_u.load_state_dict(metadata["transform_u_state"])
            if "transform_p_state" in metadata:
                self._data_transform_p.load_state_dict(metadata["transform_p_state"])
            self._transform_fitted = True
        else:
            self.train_set_index = None
            self.valid_set_index = None
            self.test_set_index  = None

            self._data_is_split  = False
            self._transform_fitted = False

    def process_all(self) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]:
        """
        Returns:
            A tuple containing:
              - A tuple of (train_loader, valid_loader, test_loader)
              - A tuple of (train_set, valid_set, test_set) tensors
              - A metadata dictionary
        """
        self.load_data(self.data_path)
        self.data_truncation()

        if not self._data_is_split:        # Train-Valid-Test split before data transformations
            self.split_dataset_index()
        self.apply_data_transformations()  # Assembles the datasets
        self.create_dataloaders()          # Creates the dataloaders, depending on the model type

        logging.info(f"Data processing complete. Train/Validation/Test sizes: {len(self.train_set)}, {len(self.valid_set)}, {len(self.test_set)}.")
        return \
            (self.train_loader, self.valid_loader, self.test_loader), \
            (self.train_set, self.valid_set, self.test_set), \
            self.metadata

    def load_data(self, path: str) -> None:
        """
        Load raw data from a binary file.

        The file is assumed to store (in order):
            x: array-like or list of array-like, shape (n_samples, n_state_features)
            training data. If training data contains multiple trajectories,
            x should be a list containing data for each trajectory. Individual
            trajectories may contain different numbers of samples.

            t: float, numpy array of shape (n_samples,), or list of numpy arrays
            If t is a float, it specifies the timestep between each sample.
            If array-like, it specifies the time (seconds in physical time) at which each sample was
            collected.
            In this case the values in t must be strictly increasing.
            In the case of multi-trajectory training data, t may also be a list
            of arrays containing the collection times for each individual
            trajectory.

            u: array-like or list of array-like, shape (n_samples, n_control_features), optional (default None)
            Control variables/inputs.
            If training data contains multiple trajectories (i.e. if x is a list of
            array-like), then u should be a list containing control variable data
            for each trajectory. Individual trajectories may contain different
            numbers of samples.

        Args:
            path (str): Path to the data file.
        """
        # Load the binary data from the file.
        data = np.load(path, allow_pickle=True)

        # Extract entries from the loaded data.
        logging.info("Loading raw data...")
        keys = ['t', 'x', 'y', 'u', 'p']
        vals = []
        for k in keys:
            _tmp = data.get(k, None)
            if k == 'x' and _tmp is None:
                msg = "x must be provided in the data file."
                logging.error(msg)
                raise ValueError(msg)
            if _tmp is not None:
                logging.info(f"{k} shape: {_tmp.shape if isinstance(_tmp, np.ndarray) else f'{len(_tmp)} list of arrays'}")
            vals.append(_tmp)
        logging.info("Raw data loaded.")

        # Process x
        x = vals[1]
        if isinstance(x, np.ndarray):
            if x.ndim == 3:  # multiple trajectories as (n_traj, n_steps, n_features)
                logging.info(f"Detected x as 3D np.ndarray (n_traj, n_steps, n_features): {x.shape}. Splitting into list of arrays.")
                self.x = [np.array(_x) for _x in x]
            elif x.ndim == 2:  # single trajectory (n_steps, n_features)
                logging.info(f"Detected x as 2D np.ndarray, treating it as a single trajectory (n_steps, n_features): {x.shape}. Wrapping as single-element list.")
                self.x = [np.array(x)]
            else:
                msg = f"Unsupported x shape: {x.shape}"
                logging.error(msg)
                raise ValueError(msg)
        elif isinstance(x, list):
            logging.info(f"Detected x as list of arrays.")
            self.x = [np.array(_x) for _x in x]
        else:
            logging.error("x must be a np.ndarray or list of np.ndarrays")
            raise TypeError("x must be a np.ndarray or list of np.ndarrays")

        # In the processing below, the raw data is converted to arrays, as they are supposed to be regular.
        # Process t
        self.t = _process_data(
            None if vals[0] is None else np.array(vals[0]),
            self.x, "t", base_dim=0, offset=0)
        if self.t[0].size == 0:
            self.t = [np.arange(_x.shape[0]) for _x in self.x]
        self.dt = [ti[1] - ti[0] for ti in self.t]

        # Process y
        self.y = _process_data(
            None if vals[2] is None else np.array(vals[2]),
            self.x, "y", base_dim=1, offset=0)

        # Process u
        self.u = _process_data(
            None if vals[3] is None else np.array(vals[3]),
            self.x, "u", base_dim=1, offset=0)
        self._is_autonomous = self.u[0].size == 0

        # Process p
        self.p = _process_data(
            None if vals[4] is None else np.array(vals[4]),
            self.x, "p", base_dim=1, offset=1)

    def data_truncation(self) -> None:
        """
        Truncate the loaded data according to the configuration.

        This includes:
          - Subsetting the number of trajectories and horizon (n_steps).
          - Populating basic metadata (dt, tf, shapes, etc.).
        """
        cfg = self.metadata['config'].get("data", {})
        n_samples: Optional[int] = cfg.get("n_samples", None)
        n_steps: Optional[int] = cfg.get("n_steps", None)

        if self.x is None or self.u is None or self.t is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Subset trajectories if n_samples is provided.
        if n_samples is not None:
            if n_samples > 1:
                self.t = self.t[:n_samples]
                self.dt = self.dt[:n_samples]
                self.x = self.x[:n_samples]
                self.y = self.y[:n_samples]
                self.u = self.u[:n_samples]
                self.p = self.p[:n_samples]

        # Truncate each trajectory's length if n_steps is provided.
        if n_steps is not None:
            self.t = [_t[:n_steps] for _t in self.t]
            self.x = [_x[:n_steps] for _x in self.x]
            self.y = [_y[:n_steps] for _y in self.y]
            self.u = [_u[:n_steps] for _u in self.u]
            # self.p is time-invariant

        # Populate metadata.
        self.metadata["n_samples"] = len(self.x)
        self.metadata["n_state_features"] = int(self.x[0].shape[-1])
        self.metadata["n_aux_features"] = int(self.y[0].shape[-1])
        self.metadata["n_control_features"] = int(self.u[0].shape[-1])
        self.metadata["n_parameters"] = int(self.p[0].shape[-1])
        logging.info("Data loaded and processed.")
        logging.info(f"Number of samples: {self.metadata['n_samples']}")
        logging.info(f"Number of state features: {self.metadata['n_state_features']}")
        logging.info(f"Number of auxiliary features: {self.metadata['n_aux_features']}")
        logging.info(f"Number of control features: {self.metadata['n_control_features']}")
        logging.info(f"Number of parameters: {self.metadata["n_parameters"]}")
        logging.info(f"Delay embedding size: {self.metadata['delay']}")

    def split_dataset_index(self):
        """
        Split the dataset into training, validation, and test sets.

        The training fraction is specified in the YAML config (default 0.75).
        The split is performed by shuffling whole trajectories.
        """
        split_cfg = self.metadata['config'].get("split", {})
        train_frac: float = split_cfg.get("train_frac", 0.75)

        if train_frac < 1.0:
            n_train = int(self.metadata["n_samples"] * train_frac)
            remaining = self.metadata["n_samples"] - n_train
            n_val = remaining // 2
            n_test = remaining - n_val

            assert n_train > 0, f"Training set must have at least one sample. Got {n_train}."
            assert n_val > 0, f"Validation set must have at least one sample. Got {n_val}."
            assert n_test > 0, f"Test set must have at least one sample. Got {n_test}."

            perm = torch.randperm(self.metadata["n_samples"])

            self.train_set_index = perm[:n_train]
            self.valid_set_index = perm[n_train:n_train+n_val]
            self.test_set_index  = perm[n_train+n_val:]
        else:
            logging.info("Using the entire dataset as train/valid/test since train_frac is 1.0.")
            logging.info("This should be done only for testing/debugging purposes.")

            n_train = self.metadata["n_samples"]
            n_val = n_test = n_train

            idcs = torch.arange(self.metadata["n_samples"])
            self.train_set_index = idcs
            self.valid_set_index = idcs
            self.test_set_index  = idcs

        self.metadata["n_train"] = n_train
        self.metadata["n_val"]   = n_val
        self.metadata["n_test"]  = n_test
        logging.info(f"Dataset size: Train: {n_train}, Validation: {n_val}, Test: {n_test}.")

        self.metadata["train_set_index"] = self.train_set_index.tolist()
        self.metadata["valid_set_index"] = self.valid_set_index.tolist()
        self.metadata["test_set_index"]  = self.test_set_index.tolist()

    def apply_data_transformations(self) -> None:
        """
        Apply data transformations to the loaded trajectories and control inputs.
        This creates the train-valid-test datasets.

        This method applies transformations defined in the configuration for x, y, u, p
        """
        assert self.train_set_index is not None, "Dataset must be split before applying transformations."

        if not self._transform_fitted:
            logging.info("Fitting transformation for state features.")
            X = [self.x[i] for i in self.train_set_index]
            self._data_transform_x.fit(X)
            self.metadata["transform_x_state"] = self._data_transform_x.state_dict()

            if self.metadata["n_aux_features"] > 0:
                logging.info("Fitting transformation for auxiliary features.")
                Y = [self.y[i] for i in self.train_set_index]
                self._data_transform_y.fit(Y)
                self.metadata["transform_y_state"] = self._data_transform_y.state_dict()

            if self.metadata["n_control_features"] > 0:
                logging.info("Fitting transformation for control inputs.")
                U = [self.u[i] for i in self.train_set_index]
                self._data_transform_u.fit(U)
                self.metadata["transform_u_state"] = self._data_transform_u.state_dict()

            if self.metadata["n_parameters"] > 0:
                logging.info("Fitting transformation for parameters.")
                P = [self.p[i] for i in self.train_set_index]
                self._data_transform_p.fit(P)
                self.metadata["transform_p_state"] = self._data_transform_p.state_dict()
        else:
            logging.info("Transformations already fitted. Skipping fitting step.")

        logging.info("Applying transformations to state features and control inputs.")
        logging.info("Training...")
        self.train_set = self._transform_by_index(self.train_set_index)

        logging.info("Validation...")
        self.valid_set = self._transform_by_index(self.valid_set_index)

        logging.info("Test...")
        self.test_set = self._transform_by_index(self.test_set_index)

        if self.metadata["delay"] > 0:
            logging.info("Conforming the time data due to delay.")
            # For time, we remove the last "delay" time steps.
            self.t = [ti[:-self.metadata["delay"]] for ti in self.t]

        self._update_dataset_metadata()

    def _transform_by_index(self, indices: torch.Tensor) -> List[DynData]:
        # Process X first
        # If the common delay is larger (larger delay in u), we trim out the first few steps.
        # This way the latest x and u are aligned.
        _X = self._data_transform_x.transform([self.x[i] for i in indices])
        _d = self.metadata["delay"] - self._data_transform_x.delay
        if _d > 0:
            _X = [x[_d:] for x in _X]

        _T = [self.t[i] for i in indices]
        if self.metadata["delay"] > 0:
            _T = [t[self.metadata["delay"]:] for t in _T]

        if self.metadata["n_aux_features"] > 0:
            # Process Y only if there are auxiliary features.
            # Same idea as for X, we trim out the first few steps if the delay in x is larger.
            _Y = self._data_transform_y.transform([self.y[i] for i in indices])
            _d = self.metadata["delay"] - self._data_transform_y.delay
            if _d > 0:
                _Y = [y[_d:] for y in _Y]
        else:
            _Y = [None for _ in _X]

        if self.metadata["n_control_features"] > 0:
            # Process U only if there are control features.
            # Same idea as for X, we trim out the first few steps if the delay in x is larger.
            _U = self._data_transform_u.transform([self.u[i] for i in indices])
            _d = self.metadata["delay"] - self._data_transform_u.delay
            if _d > 0:
                _U = [u[_d:] for u in _U]
        else:
            _U = [None for _ in _X]

        if self.metadata["n_parameters"] > 0:
            # Nothing to delay for parameters, they are time-invariant.
            _P = self._data_transform_p.transform([self.p[i] for i in indices])
        else:
            _P = [None for _ in _X]

        # Lastly assemble the dataset.
        dataset = []
        for _t, _x, _y, _u, _p in zip(_T, _X, _Y, _U, _P):
            dataset.append(DynData(
                t=torch.tensor(_t, dtype=self.dtype, device=self.device),
                x=torch.tensor(_x, dtype=self.dtype, device=self.device),
                y=torch.tensor(_y, dtype=self.dtype, device=self.device) if _y is not None else None,
                u=torch.tensor(_u, dtype=self.dtype, device=self.device) if _u is not None else None,
                p=torch.tensor(_p, dtype=self.dtype, device=self.device) if _p is not None else None
        ))
        return dataset

    def _update_dataset_metadata(self):
        # Bookkeeping metadata for the dataset.
        self.metadata['n_total_state_features'] = self._data_transform_x._out_dim
        if self.metadata["n_aux_features"] == 0:
            self.metadata['n_total_aux_features'] = 0
        else:
            self.metadata['n_total_aux_features'] = self._data_transform_y._out_dim
        if self.metadata["n_control_features"] == 0:
            self.metadata['n_total_control_features'] = 0
        else:
            self.metadata['n_total_control_features'] = self._data_transform_u._out_dim
        if self.metadata["n_parameters"] == 0:
            self.metadata['n_total_parameters'] = 0
        else:
            self.metadata['n_total_parameters'] = self._data_transform_p._out_dim
        self.metadata['n_total_features'] = self.metadata['n_total_state_features'] + self.metadata['n_total_control_features']
        self.metadata["dt_and_n_steps"] = self._create_dt_n_steps_metadata()

        logging.info(f"Number of total state features: {self.metadata['n_total_state_features']}")
        logging.info(f"Number of total auxiliary features: {self.metadata['n_total_aux_features']}")
        logging.info(f"Number of total control features: {self.metadata['n_total_control_features']}")
        logging.info(f"Number of total parameters: {self.metadata['n_total_parameters']}")

    def _create_dt_n_steps_metadata(self) -> List[List[float]]:
        """
        Create metadata for dt and n_steps, optimizing storage if values are uniform.

        Returns:
            List of [dt, n_steps] pairs. If all trajectories have the same dt and n_steps,
            returns only one entry for optimization.
        """
        # Store dt and n_steps for metadata, but don't modify self.t and self.dt
        metadata_dt_and_n_steps = []
        for dt, t in zip(self.dt, self.t):
            # Use the actual length after any truncation for metadata
            actual_n_steps = len(t)
            metadata_dt_and_n_steps.append([dt, actual_n_steps])

        # Check if uniform dt and n_steps for metadata optimization
        if len(metadata_dt_and_n_steps) > 0:
            dts = [item[0] for item in metadata_dt_and_n_steps]
            nsteps = [item[1] for item in metadata_dt_and_n_steps]
            if len(set(dts)) == 1 and len(set(nsteps)) == 1:
                # Only store one entry if both dt and n_steps are uniform
                logging.info("Uniform dt and n_steps detected across all trajectories. Only saving one entry in metadata.")
                return [metadata_dt_and_n_steps[0]]
            else:
                return metadata_dt_and_n_steps
        else:
            return []

    def create_dataloaders(self) -> None:
        """
        Create dataloaders for train, validation, and test sets based on the model type.

        This method creates and stores three dataloaders as class attributes:
        - self.train_loader
        - self.valid_loader
        - self.test_loader
        """
        dl_cfg = self.metadata['config'].get("dataloader", {})
        batch_size: int = dl_cfg.get("batch_size", 1)

        logging.info(f"Creating dataloaders for model with batch size {batch_size}.")
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,  collate_fn=DynData.collate)
        self.valid_loader = DataLoader(self.valid_set, batch_size=batch_size, shuffle=False, collate_fn=DynData.collate)
        self.test_loader  = DataLoader(self.test_set,  batch_size=batch_size, shuffle=False, collate_fn=DynData.collate)

        # Shelved until LSTM is verified.

        # if self.model_type == "NN":
        #     logging.info(f"Creating dataloaders for NN model with batch size {batch_size}.")
        #     self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, collate_fn=DynData.collate)
        #     self.valid_loader = DataLoader(self.valid_set, batch_size=batch_size, shuffle=False, collate_fn=DynData.collate)
        #     self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False, collate_fn=DynData.collate)

        # elif self.model_type == "LSTM":
        #     logging.info(f"Creating dataloaders for LSTM model with batch size {batch_size}.")
        #     # Create sequences for each set
        #     train_X, train_y = self._create_lstm_sequences(self.train_set)
        #     valid_X, valid_y = self._create_lstm_sequences(self.valid_set)
        #     test_X, test_y = self._create_lstm_sequences(self.test_set)

        #     # self.train_set = TensorDataset(train_X, train_y)
        #     # self.valid_set = TensorDataset(valid_X, valid_y)
        #     # self.test_set = TensorDataset(test_X, test_y)

        #     # Create dataloaders
        #     self.train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
        #     self.valid_loader = DataLoader(TensorDataset(valid_X, valid_y), batch_size=batch_size, shuffle=False)
        #     self.test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=batch_size, shuffle=False)

        # else:
        #     raise ValueError(f"Unknown model type: {self.model_type}")

    def _create_lstm_sequences(
        self, dataset: list[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sliding window sequences for LSTM models.
        For each trajectory (assumed to be of shape (T, n_state_features + n_control_features))
        where T can vary, this method creates sequences of length `seq_length` (X) with the subsequent
        time step as the target (y).

        Args:
          dataset (list[torch.Tensor]): List of tensors, each of shape (T, n_state_features + n_control_features)
          seq_length (int): Length of the input sequence.

        Returns:
          A tuple (X, y) where:
            - X is of shape (N, seq_length, n_state_features + n_control_features)
            - y is of shape (N, n_state_features)
        """
        seq_length = self.metadata['delay'] + 1
        X_list = []
        y_list = []
        # Loop over each trajectory.
        for traj in dataset:
            T = traj.shape[0]
            if T < seq_length + 1:
                continue
            for i in range(T - seq_length):
                X_list.append(traj[i:i + seq_length])
                y_list.append(traj[i + seq_length, :self.metadata['n_state_features']])
        X_tensor = torch.stack(X_list)
        y_tensor = torch.stack(y_list)
        return X_tensor, y_tensor

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Retrieve the dataset item at the specified index.

        Args:
            index (int): Index of the dataset item.

        Returns:
            torch.Tensor: The dataset entry corresponding to the index.
        """
        raise NotImplementedError("This method is temporarily disabled.")
        # if self.dataset is None:
        #     raise ValueError("Dataset not created. Please call create_dataset() first.")
        # return self.dataset[index]

    def __len__(self) -> int:
        """
        Return the total number of dataset entries.

        Returns:
            int: The number of dataset entries.
        """
        raise NotImplementedError("This method is temporarily disabled.")
        # return 0 if self.dataset is None else len(self.dataset)

class TrajectoryManagerGraph(TrajectoryManager):
    r"""
    A class to manage trajectory data loading, preprocessing, and
    dataloader creation - graph version.

    The graph data is assumed to be homogeneous, that each node has the same number of features.
    Hence the normalization, if done, is applied globally to all nodes.

    However, the number of edges can vary over time, and hence other quantities defined on edges.

    In the raw data, the nodal state features are expected to be concatenated sequentially.
    For example, for N nodes with M features each, the raw data for states at a time step is
    
    .. math::
        x = [x_1, x_2, ..., x_N], \text{where } x_i \in R^M,

    Same applies to other data members, if present.

    Args:
        metadata (dict): Configuration dictionary.
        device (torch.device): Torch device to use.
        adj (torch.Tensor or np.ndarray, optional): Adjacency matrix for GNN models.
            If not provided, will try to get from config.
    """

    def __init__(self, metadata: Dict, device: torch.device = torch.device("cpu"), adj: Optional[Union[torch.Tensor, np.ndarray]] = None):
        super().__init__(metadata, device)
        self.adj = adj  # Store the adjacency matrix if provided externally

        self._data_transform_ew = make_transform(self.metadata['config'].get('transform_ew', None))
        if self._data_transform_ew.delay > 0:
            msg = "Edge weight transformations with delay embedding are not supported."
            logging.error(msg)
            raise ValueError(msg)
        self._data_transform_ea = make_transform(self.metadata['config'].get('transform_ea', None))
        if self._data_transform_ea.delay > 0:
            msg = "Edge attribute transformations with delay embedding are not supported."
            logging.error(msg)
            raise ValueError(msg)

        if "train_set_index" in metadata:
            if "transform_ew_state" in metadata:
                self._data_transform_ew.load_state_dict(metadata["transform_ew_state"])
            if "transform_ea_state" in metadata:
                self._data_transform_ea.load_state_dict(metadata["transform_ea_state"])

    def load_data(self, path: str) -> None:
        super().load_data(path)

        # Load the binary data from the file again.
        # by now t/x/y/u/p should have been loaded.
        data = np.load(path, allow_pickle=True)
        ei = data.get('ei', None)
        ew = data.get('ew', None)
        ea = data.get('ea', None)

        adj = data.get('adj', None)
        if adj is not None:
            if self.adj is not None:
                logging.warning("Adjacency matrix provided both externally and in data file. Using the one from data.")
            self.adj = adj
            logging.info("Loaded adjacency matrix from data file")

        # Process ei and ew
        if ei is not None:
            if self.adj is not None:
                logging.warning("Edge index provided both externally and in data file. Using the one from data.")
        else:
            logging.info("Edge index is not in data, generating from adjacency matrix")
            ei, ew = adj_to_edge(self.adj)
        self.ei = _process_data(ei, self.x, "ei", base_dim=2, offset=0)
        self.ew = _process_data(ew, self.x, "ew", base_dim=1, offset=0)

        # Process ea
        self.ea = _process_data(ea, self.x, "ea", base_dim=2, offset=0)

        # Count nodes
        _n = []
        for _e in self.ei:
            for _ee in _e:
                _n.append(np.max(_ee) + 1)
        self.n_nodes = int(np.max(_n))
        self.metadata["n_nodes"] = self.n_nodes
        logging.info(f"Number of nodes detected: {self.n_nodes}")

    def data_truncation(self) -> None:
        super().data_truncation()

        cfg = self.metadata['config'].get("data", {})
        n_samples: Optional[int] = cfg.get("n_samples", None)
        n_steps: Optional[int] = cfg.get("n_steps", None)
        # Subset trajectories if n_samples is provided.
        if n_samples is not None:
            if n_samples > 1:
                self.ei = self.ei[:n_samples]
                self.ew = self.ew[:n_samples]
                self.ea = self.ea[:n_samples]

        # Truncate each trajectory's length if n_steps is provided.
        if n_steps is not None:
            self.ei = [_ei[:n_steps] for _ei in self.ei]
            self.ew = [_ew[:n_steps] for _ew in self.ew]
            self.ea = [_ea[:n_steps] for _ea in self.ea]

        # Complete metadata
        self.metadata["n_edge_weights"] = 1 if self.ew[0].size > 0 else 0
        self.metadata["n_edge_features"] = int(self.ea[0].shape[-1])
        logging.info(f"Number of edge features: {self.metadata['n_edge_features']}")
        logging.info(f"Number of edge weights: {self.metadata['n_edge_weights']}")

    def apply_data_transformations(self) -> None:
        """
        Apply data transformations to the loaded trajectories and control inputs.
        This creates the train-valid-test datasets.

        The raw data is expected to be [T, n_nodes * n_features], but the transformation
        assumes [T * n_nodes, n_features].  So extra reshaping is needed.
        """
        assert self.train_set_index is not None, "Dataset must be split before applying transformations."

        if not self._transform_fitted:
            logging.info("Fitting transformation for state features.")
            X = [self._graph_data_reshape(self.x[i], forward=True) for i in self.train_set_index]
            self._data_transform_x.fit(np.vstack(X))  # Make sure the input is 3D
            self.metadata["transform_x_state"] = self._data_transform_x.state_dict()

            if self.metadata["n_aux_features"] > 0:
                logging.info("Fitting transformation for auxiliary features.")
                Y = [self._graph_data_reshape(self.y[i], forward=True) for i in self.train_set_index]
                self._data_transform_y.fit(np.vstack(Y))
                self.metadata["transform_y_state"] = self._data_transform_y.state_dict()

            if self.metadata["n_control_features"] > 0:
                logging.info("Fitting transformation for control inputs.")
                U = [self._graph_data_reshape(self.u[i], forward=True) for i in self.train_set_index]
                self._data_transform_u.fit(np.vstack(U))
                self.metadata["transform_u_state"] = self._data_transform_u.state_dict()

            if self.metadata["n_parameters"] > 0:
                logging.info("Fitting transformation for parameters.")
                P = [self._graph_data_reshape(self.p[i][...,None,:], forward=True) for i in self.train_set_index]
                self._data_transform_p.fit(np.vstack(P))
                self.metadata["transform_p_state"] = self._data_transform_p.state_dict()

            # For edges, stack all edges and fit the transformations.
            if self.metadata["n_edge_weights"] > 0:
                logging.info("Fitting transformation for edge weights.")
                E = [np.hstack(self.ew[i]).reshape(-1,1) for i in self.train_set_index]
                self._data_transform_ew.fit(E)
                self.metadata["transform_ew_state"] = self._data_transform_ew.state_dict()

            if self.metadata["n_edge_features"] > 0:
                logging.info("Fitting transformation for edge features.")
                E = [np.vstack(self.ea[i]) for i in self.train_set_index]
                self._data_transform_ea.fit(E)
                self.metadata["transform_ea_state"] = self._data_transform_ea.state_dict()
        else:
            logging.info("Transformations already fitted. Skipping fitting step.")

        logging.info("Applying transformations to state features and control inputs.")
        logging.info("Training...")
        self.train_set = self._transform_by_index(self.train_set_index)

        logging.info("Validation...")
        self.valid_set = self._transform_by_index(self.valid_set_index)

        logging.info("Test...")
        self.test_set = self._transform_by_index(self.test_set_index)

        if self.metadata["delay"] > 0:
            logging.info("Conforming the time data due to delay.")
            # For time, we remove the last "delay" time steps.
            self.t = [ti[:-self.metadata["delay"]] for ti in self.t]

        self._update_dataset_metadata()

    def _transform_by_index(self, indices: torch.Tensor) -> List[DynData]:
        # Process X first
        # If the common delay is larger (larger delay in u), we trim out the first few steps.
        # This way the latest x and u are aligned.
        tmp = [self._graph_data_reshape(self.x[i], forward=True) for i in indices]
        _X = [np.array(self._data_transform_x.transform(_t)) for _t in tmp]
        _d = self.metadata["delay"] - self._data_transform_x.delay
        if _d > 0:
            _X = [x[:, _d:] for x in _X]

        _T = [self.t[i] for i in indices]
        if self.metadata["delay"] > 0:
            _T = [t[self.metadata["delay"]:] for t in _T]

        if self.metadata["n_aux_features"] > 0:
            tmp = [self._graph_data_reshape(self.y[i], forward=True) for i in indices]
            _Y = [np.array(self._data_transform_y.transform(_t)) for _t in tmp]
            _d = self.metadata["delay"]
            if _d > 0:
                _Y = [y[:, _d:] for y in _Y]
        else:
            _Y = [None for _ in _X]

        if self.metadata["n_control_features"] > 0:
            tmp = [self._graph_data_reshape(self.u[i], forward=True) for i in indices]
            _U = [np.array(self._data_transform_u.transform(_t)) for _t in tmp]
            _d = self.metadata["delay"] - self._data_transform_u.delay
            if _d > 0:
                _U = [u[:, _d:] for u in _U]
        else:
            _U = [None for _ in _X]

        if self.metadata["n_parameters"] > 0:
            tmp = [self._graph_data_reshape(self.p[i][..., None, :], forward=True) for i in indices]
            _P = [self._data_transform_p.transform(_t) for _t in tmp]
        else:
            _P = [None for _ in _X]

        _Ei = [self.ei[i] for i in indices]

        if self.metadata["n_edge_weights"] > 0:
            _Ew = []
            for i in indices:
                _tmp = self._data_transform_ew.transform([e.reshape(-1,1) for e in self.ew[i][_d:]])
                _Ew.append([_t.reshape(-1) for _t in _tmp])
        else:
            _Ew = [None for _ in _X]

        if self.metadata["n_edge_features"] > 0:
            _Ea = [self._data_transform_ea.transform(self.ea[i][_d:]) for i in indices]
        else:
            _Ea = [None for _ in _X]

        # Lastly assemble the dataset.
        dataset = []
        for _t, _x, _y, _u, _p, _ei, _ew, _ea in zip(_T, _X, _Y, _U, _P, _Ei, _Ew, _Ea):
            dataset.append(DynData(
                t=torch.tensor(_t, dtype=self.dtype, device=self.device),
                x=torch.tensor(self._graph_data_reshape(_x, forward=False), dtype=self.dtype, device=self.device),
                y=torch.tensor(self._graph_data_reshape(_y, forward=False), dtype=self.dtype, device=self.device) if _y is not None else None,
                u=torch.tensor(self._graph_data_reshape(_u, forward=False), dtype=self.dtype, device=self.device) if _u is not None else None,
                p=torch.tensor(self._graph_data_reshape(_p, forward=False).squeeze(-2), dtype=self.dtype, device=self.device) if _p is not None else None,
                ei=[torch.tensor(_e) for _e in _ei],
                ew=[torch.tensor(_e) for _e in _ew] if _ew is not None else None,
                ea=[torch.tensor(_a) for _a in _ea] if _ea is not None else None,
            ))
        return dataset

    def _graph_data_reshape(self, data: np.ndarray, forward: bool) -> np.ndarray:
        """
        Reshape the raw data between [T, n_nodes * n_features] and [n_nodes, T, n_features].

        The 0th axis is as if batch.
        """
        if forward:
            # Reshape from [T, n_nodes * n_features] to [n_nodes, T, n_features]
            tmp = data.reshape(data.shape[0], self.n_nodes, -1)  # [T, n_nodes, n_features_per_node]
            return np.swapaxes(tmp, 0, 1)  # [n_nodes, T, n_features_per_node]

        # Reshape from [n_nodes, T, n_features] to [T, n_nodes * n_features]
        tmp = np.swapaxes(data, 0, 1)  # [T, n_nodes, n_features_per_node]
        return tmp.reshape(tmp.shape[0], -1)

    def create_dataloaders(self) -> None:
        """
        For graph data, we aggregate the trajectories into batches of graphs.
        """
        logging.info(f"Creating dataloaders for model with batch size 1 for graph data.")
        self.train_loader = DataLoader([DynData.collate(self.train_set)], batch_size=1, shuffle=True,  collate_fn=DynData.collate)
        self.valid_loader = DataLoader([DynData.collate(self.valid_set)], batch_size=1, shuffle=False, collate_fn=DynData.collate)
        self.test_loader  = DataLoader([DynData.collate(self.test_set)],  batch_size=1, shuffle=False, collate_fn=DynData.collate)
