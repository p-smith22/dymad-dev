from dataclasses import dataclass, field
import torch
from typing import List, Optional, Union

def _ensure_graph_format(lst: Union[List[torch.Tensor], torch.Tensor, None], base_dim) -> torch.NestedTensor:
    if lst is None:
        return None
    if isinstance(lst, list):
        _need_nested = False
        for item in lst:
            if item.shape != lst[0].shape:
                _need_nested = True
                break
        if _need_nested:
            nt = torch.nested.nested_tensor(lst, dtype=lst[0].dtype, layout=torch.jagged)
        else:
            nt = torch.tensor(lst)
    elif isinstance(lst, torch.Tensor):
        if lst.ndim == base_dim:
            nt = lst.unsqueeze(0)
        elif lst.ndim > base_dim + 1:
            raise ValueError(f"Invalid tensor shape for graph data: {lst.shape}. Expected shape with base dim {base_dim} or {base_dim+1}.")
        else:
            nt = lst
    else:
        raise ValueError(f"Invalid type for graph data: {type(lst)}. Expected list or torch.Tensor.")
    return nt

def _collate_nested_tensor(lst: List[torch.Tensor]) -> torch.Tensor:
    """
    lst is (n_batch,) list of tensors with shape (n_steps, n_edges, ...),
    where n_edges can be jagged.  The method collates the edge dimension over the batch,
    resulting in shape (n_steps, n_total_edges, ...).
    """
    cache = [l.unbind() for l in lst]  # list of (n_steps,) lists of (n_edges, ...)
    collated = []
    for step_items in zip(*cache):  # step_items is (n_batch,) tuple of (n_edges, ...)
        collated.append(torch.concatenate(step_items, dim=0))
    return torch.nested.nested_tensor(collated, layout=torch.jagged)

def _slice(tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """
    Slice a NestedTensor along the first dimension (time steps).
    Works for regular tensor as well.

    Args:
        tensor (torch.Tensor): Input NestedTensor of shape (n_steps, ...).
        start (int): Start index for slicing.
        end (int): End index for slicing.

    Returns:
        torch.Tensor: Sliced NestedTensor of shape (end - start, ...).
    """
    if tensor.is_nested:
        offsets = tensor.offsets()
        offset_0 = offsets[start]
        offset_1 = offsets[end]
        sliced_values = tensor.values()[offset_0:offset_1]
        return torch.nested.nested_tensor_from_jagged(
            sliced_values,
            offsets[start:end + 1] - offsets[start],
        )
    return tensor[start:end]

def _unfold(tensor: torch.Tensor, window: int, stride: int, offset: int = 0) -> torch.Tensor:
    """
    Unfold a NestedTensor along the first dimension (time steps).
    Works for regular tensor as well.

    In this specialized unfolding, the graph size is expanded by n_windows times.

    Args:
        tensor (torch.Tensor): Input NestedTensor of shape (n_steps, n_edges, ...).
        window (int): Size of the sliding window.
        stride (int): Step size for the sliding window.
        offset (int): Offset to be added to edge indices.

    Returns:
        torch.Tensor: Unfolded NestedTensor of shape (window, n_windows x n_edges, ...).
    """
    if tensor is None:
        return None
    data = tensor.unbind()
    buffer = []
    for i in range(window):
        buffer.append(torch.cat(data[i::stride], dim=0) + i*offset)
    if tensor.is_nested:
        offsets = [len(b) for b in buffer]
        return torch.nested.nested_tensor_from_jagged(
            buffer,
            [0] + torch.tensor(offsets).cumsum(dim=0).tolist()
        )
    return torch.tensor(buffer)

@dataclass
class DynData:
    """
    Data structure for time series data.

    Operates in normal mode or graph mode depending on the presence of edge_index.

    - In normal mode, data shape is (batch_size, n_steps, ...).
    - In graph mode, batch_size is always 1, so

        - Regular data shape is (1, n_steps, ...),
        - Graph data shape is (n_steps, n_edges, ...)

    The reason is to always aggregate graphs as much as possible for GNN evaluation efficiency.
    As a result, in the two modes, the collate and unfold methods behave differently.

    In addition, in graph mode, the number of nodes are assumed to be the same across
    all time steps (for now).  But the number of edges can vary per time step; if it varies,
    the edge-related data members are stored as NestedTensors.
    """
    # Generic data
    t: Optional[torch.Tensor] = None
    """t (torch.Tensor): Time tensor of shape (batch_size, n_steps)."""

    x: torch.Tensor = field(default_factory=torch.Tensor)
    """
    x (torch.Tensor): Feature tensor of shape (batch_size, n_steps, n_features).
    The only data member that is always required.
    """

    y: Optional[torch.Tensor] = None
    """
    y (torch.Tensor): Auxiliary tensor of shape (batch_size, n_steps, n_aux).
    Additional data that may be used for supervised learning, e.g., additional observations.
    """

    u: Optional[torch.Tensor] = None
    """u (torch.Tensor): Control tensor of shape (batch_size, n_steps, n_controls)."""

    p: Optional[torch.Tensor] = None
    """
    p (torch.Tensor): Parameter tensor of shape (batch_size, n_parameters).
    Meant for constants that do not vary with time.
    If there are intermittent step changes in parameters, they should be treated as controls in u.
    """

    # Graph-only
    ei: Optional[torch.Tensor] = None
    """
    edge_index (torch.NestedTensor): Edge index tensor for graph structure, shape (n_steps, n_edges, 2).
    """

    ew: Optional[torch.Tensor] = None
    """
    edge_weight (torch.NestedTensor): Edge weight tensor for graph structure, shape (n_edges, n_steps).
    """

    ea: Optional[torch.Tensor] = None
    """
    edge_attr (torch.NestedTensor): Edge attribute tensor for graph structure, shape (n_edges, n_steps, n_edge_features).
    """

    n_nodes: int = 0
    """n_nodes (int): Number of nodes in the graph structure."""

    # Other data
    meta: List[dict] = field(default_factory=list)
    """meta (List[dict]): Metadata for each time series."""

    _has_graph: bool = False
    """_has_graph (bool): Whether the data has graph structure."""

    def __post_init__(self):
        if self.ei is not None:
            # There is graph structure
            self._has_graph = True
            # Ensure NestedTensor
            self.ei = _ensure_graph_format(self.ei, base_dim=2)
            self.ew = _ensure_graph_format(self.ew, base_dim=1)
            self.ea = _ensure_graph_format(self.ea, base_dim=2)

            self.n_nodes = self.ei.max().item() + 1
            self.x_reshape = self.x.shape[:-1] + (self.n_nodes, -1) if self.x is not None else None
            self.y_reshape = self.y.shape[:-1] + (self.n_nodes, -1) if self.y is not None else None
            self.u_reshape = self.u.shape[:-1] + (self.n_nodes, -1) if self.u is not None else None
            self.p_reshape = self.p.shape[:-1] + (self.n_nodes, -1) if self.p is not None else None
        else:
            self._has_graph = False

    def to(self, device: torch.device, non_blocking: bool = False) -> "DynData":
        """
        Move data tensors to a different device.

        Args:
            device (torch.device): The target device.
            non_blocking (bool, optional): If True, the operation will be non-blocking.

        Returns:
            DynData: A DynData instance with tensors on the target device.
        """
        self.t = self.t.to(device, non_blocking=non_blocking) if self.t is not None else None
        self.x = self.x.to(device, non_blocking=non_blocking)
        self.y = self.y.to(device, non_blocking=non_blocking) if self.y is not None else None
        self.u = self.u.to(device, non_blocking=non_blocking) if self.u is not None else None
        self.p = self.p.to(device, non_blocking=non_blocking) if self.p is not None else None
        if self._has_graph:
            self.ei = self.ei.to(device, non_blocking=non_blocking) # self.ei is always not None in graph mode
            self.ew = self.ew.to(device, non_blocking=non_blocking) if self.ew is not None else None
            self.ea = self.ea.to(device, non_blocking=non_blocking) if self.ea is not None else None
        return self

    @classmethod
    def collate(cls, batch_list: List["DynData"]) -> "DynData":
        """
        Collate a list of DynData instances into a single DynData instance.
        Needed by DataLoader to stack data tensors.

        The graph version follows PyGData, that assembles the graphs of each sample
        into a single large graph, so that the subsequent GNN evaluation operates
        on a single graph to maximize parallelism.

        Args:
            batch_list (List[DynData]): List of DynData instances to collate.

        Returns:
            DynData: A single DynData instance with stacked data tensors.
        """
        ms = []
        for b in batch_list:
            ms += b.meta

        if batch_list[0]._has_graph:
            ts = torch.concatenate([b.t for b in batch_list], dim=-1).unsqueeze(0) if batch_list[0].t is not None else None
            xs = torch.concatenate([b.x for b in batch_list], dim=-1).unsqueeze(0)
            ys = torch.concatenate([b.y for b in batch_list], dim=-1).unsqueeze(0) if batch_list[0].y is not None else None
            us = torch.concatenate([b.u for b in batch_list], dim=-1).unsqueeze(0) if batch_list[0].u is not None else None
            ps = torch.concatenate([b.p for b in batch_list], dim=-1).unsqueeze(0) if batch_list[0].p is not None else None

            n_nodes = [0] + [b.n_nodes for b in batch_list[:-1]]
            offset = torch.tensor(n_nodes).cumsum(dim=0)
            tmp = [b.ei + offset[i] for i, b in enumerate(batch_list)]

            ei = _collate_nested_tensor(tmp)
            ew = _collate_nested_tensor([b.ew for b in batch_list]) if batch_list[0].ew is not None else None
            ea = _collate_nested_tensor([b.ea for b in batch_list]) if batch_list[0].ea is not None else None

            return DynData(t=ts, x=xs, y=ys, u=us, p=ps, ei=ei, ew=ew, ea=ea, meta=ms)

        ts = torch.stack([b.t for b in batch_list], dim=0) if batch_list[0].t is not None else None
        xs = torch.stack([b.x for b in batch_list], dim=0)
        ys = torch.stack([b.y for b in batch_list], dim=0) if batch_list[0].y is not None else None
        us = torch.stack([b.u for b in batch_list], dim=0) if batch_list[0].u is not None else None
        ps = torch.stack([b.p for b in batch_list], dim=0) if batch_list[0].p is not None else None
        return DynData(t=ts, x=xs, y=ys, u=us, p=ps, meta=ms)

    def truncate(self, num_step):
        return DynData(
            t = self.t[:, :num_step] if self.t is not None else None,
            x = self.x[:, :num_step, :],
            y = self.y[:, :num_step, :] if self.y is not None else None,
            u = self.u[:, :num_step, :] if self.u is not None else None,
            p = self.p,
            ei = _slice(self.ei, 0, num_step) if self._has_graph else None,
            ew = _slice(self.ew, 0, num_step) if self.ew is not None else None,
            ea = _slice(self.ea, 0, num_step) if self.ea is not None else None,
            n_nodes = self.n_nodes,
            meta = self.meta
        )

    def unfold(self, window: int, stride: int) -> "DynData":
        """
        Unfold the data into overlapping windows.

        The regular data array is assumed to be of shape (batch_size, n_steps, ...).
        unfold produces a tensor of shape (batch_size, n_window, ..., window).

            - In normal mode, merge the first two dimensions and permute the last two gives
              (batch_size*n_window, window, ...)
            - In graph mode, we always have batch_size=1, and we permute to
              (1, window, n_window x ...), where x is the merged edge dimension over all windows.

        Accompanying the unfold in graph mode, the edge indices and attributes are also unfolded.
        The result is a graph of n_window times the size of the original graph.

        Args:
            window (int): Size of the sliding window.
            stride (int): Step size for the sliding window.

        Returns:
            DynData: A new DynData instance with unfolded data.
        """
        if self._has_graph:
            # Graph mode:
            _unf = lambda z: z.unfold(1, window, stride).permute(0, 3, 1, 2).reshape(1, window, -1) if z is not None else None
            t_unfolded = self.t.unfold(1, window, stride).reshape(-1, window).transpose(1, 0) if self.t is not None else None
            x_unfolded = _unf(self.x)
            y_unfolded = _unf(self.y)
            u_unfolded = _unf(self.u)
            n_windows  = x_unfolded.size(-1) // self.x.size(-1)
            p_unfolded = self.p.tile((n_windows,)) if self.p is not None else None

            ei_unfolded = _unfold(self.ei, window, stride, offset=self.n_nodes)
            ew_unfolded = _unfold(self.ew, window, stride)
            ea_unfolded = _unfold(self.ea, window, stride)
            return DynData(
                t=t_unfolded, x=x_unfolded, y=y_unfolded, u=u_unfolded, p=p_unfolded,
                ei=ei_unfolded, ew=ew_unfolded, ea=ea_unfolded, meta=self.meta)

        # Normal mode:
        _unf = lambda z: z.unfold(1, window, stride).reshape(-1, z.size(-1), window).permute(0, 2, 1) if z is not None else None
        t_unfolded = self.t.unfold(1, window, stride).reshape(-1, window) if self.t is not None else None
        x_unfolded = _unf(self.x)
        y_unfolded = _unf(self.y)
        u_unfolded = _unf(self.u)
        n_windows  = x_unfolded.size(0) // self.x.size(0)
        p_unfolded = self.p.repeat_interleave(n_windows, dim=0) if self.p is not None else None
        return DynData(t=t_unfolded, x=x_unfolded, y=y_unfolded, u=u_unfolded, p=p_unfolded, meta=self.meta)

    @property
    def xg(self) -> torch.Tensor:
        """
        Get the feature tensor with shape (batch_size, n_steps, n_nodes, n_features_per_node).
        """
        return self.x.reshape(*self.x_reshape)

    @property
    def yg(self) -> torch.Tensor:
        """
        Get the auxiliary tensor with shape (batch_size, n_steps, n_nodes, n_aux_per_node).
        """
        return self.y.reshape(*self.y_reshape)

    @property
    def ug(self) -> Union[torch.Tensor, None]:
        """
        Get the control tensor with shape (batch_size, n_steps, n_nodes, n_controls).
        """
        return self.u.reshape(*self.u_reshape)

    @property
    def pg(self) -> Union[torch.Tensor, None]:
        """
        Get the parameter tensor with shape (batch_size, n_nodes, n_parameters).
        """
        return self.p.reshape(*self.p_reshape)

    def g(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reshape a tensor to have shape (batch_size, n_steps, n_nodes, -1).

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, n_steps, n_nodes * features).

        Returns:
            torch.Tensor: Reshaped tensor of shape (batch_size, n_steps, n_nodes, features).
        """
        out_shape = z.shape[:-1] + (self.n_nodes, -1)
        return z.reshape(*out_shape)

    def G(self, z: torch.Tensor) -> torch.Tensor:
        """
        The reverse of g()
        """
        out_shape = z.shape[:-2] + (-1,)
        return z.reshape(*out_shape)
