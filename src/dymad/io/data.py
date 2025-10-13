from dataclasses import dataclass, field
import torch
from typing import List, Union

@dataclass
class DynData:
    """
    Data structure for time series data.

    Operates in normal mode or graph mode depending on the presence of edge_index.
    In the two modes, the collate and unfold methods behave differently.
    """
    # Generic data
    t: torch.Tensor = torch.empty((0,))
    """t (torch.Tensor): Time tensor of shape (batch_size, n_steps)."""

    x: torch.Tensor = torch.empty((0,))
    """x (torch.Tensor): State tensor of shape (batch_size, n_steps, n_states)."""

    y: torch.Tensor = torch.empty((0,))
    """y (torch.Tensor): Observation tensor of shape (batch_size, n_steps, n_observations)."""

    u: torch.Tensor = torch.empty((0,))
    """u (torch.Tensor): Control tensor of shape (batch_size, n_steps, n_controls)."""

    p: torch.Tensor = torch.empty((0,))
    """p (torch.Tensor): Parameter tensor of shape (batch_size, n_steps, n_parameters)."""

    # Graph-only
    ei: Union[torch.Tensor, None] = None
    """edge_index (torch.Tensor): Edge index tensor for graph structure, shape (batch_size, n_steps, 2, n_edges)."""

    ew: torch.Tensor = torch.empty((0,))
    """edge_weight (torch.Tensor): Edge weight tensor for graph structure, shape (batch_size, n_steps, n_edges)."""

    ea: torch.Tensor = torch.empty((0,))
    """edge_attr (torch.Tensor): Edge attribute tensor for graph structure, shape (batch_size, n_steps, n_edges, n_edge_features)."""

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
            self.n_nodes = self.ei.max().item() + 1
            self.x_reshape = self.x.shape[:-1] + (self.n_nodes, -1)
            self.y_reshape = self.y.shape[:-1] + (self.n_nodes, -1)
            self.u_reshape = self.u.shape[:-1] + (self.n_nodes, -1)
            self.p_reshape = self.p.shape[:-1] + (self.n_nodes, -1)
        self._has_graph = False

    def to(self, device: torch.device, non_blocking: bool = False) -> "DynData":
        """
        Move the state and control tensors to a different device.

        Args:
            device (torch.device): The target device.
            non_blocking (bool, optional): If True, the operation will be non-blocking.

        Returns:
            DynData: A DynData instance with tensors on the target device.
        """
        self.t = self.t.to(device, non_blocking=non_blocking)
        self.x = self.x.to(device, non_blocking=non_blocking)
        self.y = self.y.to(device, non_blocking=non_blocking)
        self.u = self.u.to(device, non_blocking=non_blocking)
        self.p = self.p.to(device, non_blocking=non_blocking)
        if self._has_graph:
            self.ei = self.ei.to(device, non_blocking=non_blocking)
            self.ew = self.ew.to(device, non_blocking=non_blocking)
            self.ea = self.ea.to(device, non_blocking=non_blocking)
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
            DynData: A single DynData instance with stacked state and control tensors.
        """
        ms = []
        for b in batch_list:
            ms += b.meta

        if batch_list[0]._has_graph:
            ts = torch.concatenate([b.t for b in batch_list], dim=-1).unsqueeze(0)
            xs = torch.concatenate([b.x for b in batch_list], dim=-1).unsqueeze(0)
            ys = torch.concatenate([b.y for b in batch_list], dim=-1).unsqueeze(0)
            us = torch.concatenate([b.u for b in batch_list], dim=-1).unsqueeze(0)
            ps = torch.concatenate([b.p for b in batch_list], dim=-1).unsqueeze(0)

            n_nodes = [0] + [b.n_nodes for b in batch_list[:-1]]
            offset = torch.tensor(n_nodes).cumsum(dim=0)
            ei = torch.concatenate([
                b.ei + offset[i] for i, b in enumerate(batch_list)],
                dim=-1).unsqueeze(0)
            
            ew = torch.concatenate([b.ew for b in batch_list], dim=-1).unsqueeze(0)
            ea = torch.concatenate([b.ea for b in batch_list], dim=-1).unsqueeze(0)

            return DynData(t=ts, x=xs, y=ys, u=us, p=ps, ei=ei, ew=ew, ea=ea, meta=ms)

        ts = torch.stack([b.t for b in batch_list], dim=0)
        xs = torch.stack([b.x for b in batch_list], dim=0)
        ys = torch.stack([b.y for b in batch_list], dim=0)
        us = torch.stack([b.u for b in batch_list], dim=0)
        ps = torch.stack([b.p for b in batch_list], dim=0)
        return DynData(t=ts, x=xs, y=ys, u=us, p=ps, meta=ms)

    def truncate(self, num_step):
        return DynData(
            t = self.t[:, :num_step] if self.t.numel() > 0 else self.t,
            x = self.x[:, :num_step, :] if self.x.numel() > 0 else self.x,
            y = self.y[:, :num_step, :] if self.y.numel() > 0 else self.y,
            u = self.u[:, :num_step, :] if self.u.numel() > 0 else self.u,
            p = self.p[:, :num_step, :] if self.p.numel() > 0 else self.p,
            ei = self.ei[:, :num_step, :, :] if self._has_graph else None,
            ew = self.ew[:, :num_step, :]    if self.ew.numel() > 0 else self.ew,
            ea = self.ea[:, :num_step, :, :] if self.ea.numel() > 0 else self.ea,
            n_nodes = self.n_nodes,
            meta = self.meta
        )

    def unfold(self, window: int, stride: int) -> "DynData":
        """
        Unfold the data into overlapping windows.

        Args:
            window (int): Size of the sliding window.
            stride (int): Step size for the sliding window.

        Returns:
            DynData: A new DynData instance with unfolded data.
        """
        # The array is assumed to be of shape (batch_size, n_steps, :)
        # unfold produces a tensor of shape (batch_size, n_window, :, window)
        # merge the first two dimensions and permute the last two gives (batch_size*n_window, window, :)
        t_unfolded = self.t.unfold(1, window, stride).reshape(-1, window) if self.t.numel() > 0 else None
        x_unfolded = self.x.unfold(1, window, stride).reshape(-1, self.x.size(-1), window).permute(0, 2, 1) if self.x.numel() > 0 else None
        y_unfolded = self.y.unfold(1, window, stride).reshape(-1, self.y.size(-1), window).permute(0, 2, 1) if self.y.numel() > 0 else None
        u_unfolded = self.u.unfold(1, window, stride).reshape(-1, self.u.size(-1), window).permute(0, 2, 1) if self.u.numel() > 0 else None
        p_unfolded = self.p.unfold(1, window, stride).reshape(-1, self.p.size(-1), window).permute(0, 2, 1) if self.p.numel() > 0 else None
        if self._has_graph:
            ei_unfolded = self.ei.unfold(1, window, stride).reshape(-1, 2, self.ei.size(-1), window).permute(0, 3, 1, 2) if self.ei.numel() > 0 else None
            ew_unfolded = self.ew.unfold(1, window, stride).reshape(-1, self.ew.size(-1), window).permute(0, 2, 1) if self.ew.numel() > 0 else None
            ea_unfolded = self.ea.unfold(1, window, stride).reshape(-1, self.ea.shape[-2:], window).permute(0, 3, 1, 2) if self.ea.numel() > 0 else None
            return DynData(
                t=t_unfolded, x=x_unfolded, y=y_unfolded, u=u_unfolded, p=p_unfolded,
                ei=ei_unfolded, ew=ew_unfolded, ea=ea_unfolded, meta=self.meta)
        return DynData(t=t_unfolded, x=x_unfolded, y=y_unfolded, u=u_unfolded, p=p_unfolded, meta=self.meta)

    @property
    def xg(self) -> torch.Tensor:
        """
        Get the state tensor with shape (batch_size, n_steps, n_nodes, n_states_per_node).
        """
        return self.x.reshape(*self.x_reshape)

    @property
    def yg(self) -> torch.Tensor:
        """
        Get the observation tensor with shape (batch_size, n_steps, n_nodes, n_obs_per_node).
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
        Get the parameter tensor with shape (batch_size, n_steps, n_nodes, n_parameters).
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
