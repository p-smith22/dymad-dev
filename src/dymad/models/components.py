import torch
from typing import Tuple

from dymad.io import DynData
from dymad.models.model_base import Decoder, Dynamics, Encoder

# ------------------
# Encoder modules
# ------------------

class EncIden(Encoder):
    """Identity transform."""
    def forward(self, w: DynData) -> torch.Tensor:
        return w.x

class EncSmplAuto(Encoder):
    """Only encodes states."""
    def forward(self, w: DynData) -> torch.Tensor:
        return self.net(w.x)

class EncSmplCtrl(Encoder):
    """Encodes states and controls."""
    def forward(self, w: DynData) -> torch.Tensor:
        return self.net(torch.cat([w.x, w.u], dim=-1))

class EncGraphIden(Encoder):
    """Identity transform."""
    def forward(self, w: DynData) -> torch.Tensor:
        return w.xg

class EncGraphAuto(Encoder):
    """Using GNN in EncAuto."""
    def forward(self, w: DynData) -> torch.Tensor:
        return w.g(self.net(w.xg, w.ei, w.ew, w.ea))

class EncGraphCtrl(Encoder):
    """Using GNN in EncCtrl."""
    def forward(self, w: DynData) -> torch.Tensor:
        xu_cat = torch.cat([w.xg, w.ug], dim=-1)
        return w.g(self.net(xu_cat, w.ei, w.ew, w.ea))

class EncNodeAuto(Encoder):
    """Using EncAuto for each node of graph."""
    def forward(self, w: DynData) -> torch.Tensor:
        return w.G(self.net(w.xg))      # G is needed for unified data structure

class EncNodeCtrl(Encoder):
    """Using EncCtrl for each node of graph."""
    def forward(self, w: DynData) -> torch.Tensor:
        xu_cat = torch.cat([w.xg, w.ug], dim=-1)
        return w.G(self.net(xu_cat))    # G is needed for unified data structure

ENC_MAP = {
    "iden"       : EncIden,
    "smpl_auto"  : EncSmplAuto,
    "smpl_ctrl"  : EncSmplCtrl,
    "graph_iden" : EncGraphIden,
    "graph_auto" : EncGraphAuto,
    "graph_ctrl" : EncGraphCtrl,
    "node_iden"  : EncIden,       # Effectively same as regular iden
    "node_auto"  : EncNodeAuto,
    "node_ctrl"  : EncNodeCtrl
}


# ------------------
# Decoder modules
# ------------------

class DecIden(Decoder):
    """Identity transform."""
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return z

class DecAuto(Decoder):
    """Only decodes states."""
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(z)

class DecGraph(Decoder):
    """Decode by GNN."""
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(z, w.ei, w.ew, w.ea)

class DecNode(Decoder):
    """Using DecAuto for each node of graph."""
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return w.G(self.net(w.g(z)))    # G is needed for unified data structure

DEC_MAP = {
    "iden"  : DecIden,
    "auto"  : DecAuto,
    "graph" : DecGraph,
    "node"  : DecNode
}


# ------------------
# Dynamics modules - features
# ------------------

def zu_cat_none(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """No concatenation, just return z."""
    return z

def zu_cat_smpl(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Simple concatenation of z and u."""
    return torch.cat([z, w.u], dim=-1)

def zu_blin_no_const(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Compute bilinear features without constant term."""
    z_u = (z.unsqueeze(-1) * w.u.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    return torch.cat([z, z_u], dim=-1)

def zu_blin_with_const(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Compute bilinear features with constant term."""
    z_u = (z.unsqueeze(-1) * w.u.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    return torch.cat([z, z_u, w.u], dim=-1)

def zu_cat_smpl_graph(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Simple concatenation of z and u on graph."""
    return torch.cat([z, w.ug], dim=-1)

def zu_blin_no_const_graph(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Compute bilinear features without constant term for graph data."""
    u_reshaped = w.ug
    z_u = (z.unsqueeze(-1) * u_reshaped.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    return torch.cat([z, z_u], dim=-1)

def zu_blin_with_const_graph(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Compute bilinear features with constant term for graph data."""
    u_reshaped = w.ug
    z_u = (z.unsqueeze(-1) * u_reshaped.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    return torch.cat([z, z_u, u_reshaped], dim=-1)

FZU_MAP = {
    "none"                  : zu_cat_none,
    "cat"                   : zu_cat_smpl,
    "blin_no_const"         : zu_blin_no_const,
    "blin_with_const"       : zu_blin_with_const,
    "graph_cat"             : zu_cat_smpl_graph,
    "graph_blin_no_const"   : zu_blin_no_const_graph,
    "graph_blin_with_const" : zu_blin_with_const_graph
}


# ------------------
# Dynamics modules
# ------------------

class DynDirect(Dynamics):
    """Processing without control inputs."""
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(self.features(z, w))

class DynSkip(Dynamics):
    """Processing with skip connection."""
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return z + self.net(self.features(z, w))

class DynGraphDirect(Dynamics):
    """Processing by GNN."""
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(w.g(self.features(z, w)), w.ei, w.ew, w.ea)   # G is effectively applied in the net

class DynGraphSkip(Dynamics):
    """Processing by GNN with skip connection."""
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return z + self.net(w.g(self.features(z, w)), w.ei, w.ew, w.ea)   # G is effectively applied in the net

DYN_MAP = {
    "direct"       : DynDirect,
    "skip"         : DynSkip,
    "graph_direct" : DynGraphDirect,
    "graph_skip"   : DynGraphSkip
}


# ------------------
# Dynamics modules - linear features
# ------------------

def linear_eval_smpl(mdl, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute linear evaluation, dz, and states, z, for the model."""
    z = mdl.encoder(w)
    z_dot = mdl.dynamics(z, w)
    return z_dot, z

def linear_features_smpl(mdl, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute linear features, f, and outputs, dz, for the model."""
    z = mdl.encoder(w)
    return mdl.dynamics.features(z, w), z

def linear_eval_graph(mdl, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute linear evaluation, dz, and states, z, for the model."""
    z = mdl.encoder(w)
    z_dot = mdl.dynamics(z, w)
    return z_dot.permute(0, 2, 1, 3), z.permute(0, 2, 1, 3)

def linear_features_graph(mdl, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute linear features, f, and outputs, dz, for the model."""
    z = mdl.encoder(w)
    f = mdl.dynamics.features(z, w)
    return f.permute(0, 2, 1, 3), z.permute(0, 2, 1, 3)

LIN_MAP = {
    "smpl"  : (linear_eval_smpl, linear_features_smpl),
    "graph" : (linear_eval_graph, linear_features_graph)
}
