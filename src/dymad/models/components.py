import torch

from dymad.io import DynData
from dymad.models.model_base import Decoder, Encoder, Processor

# ------------------
# Encoder modules
# ------------------

class EncIden(Encoder):
    """Identity transform."""
    GRAPH = False
    AUTO = True
    def forward(self, w: DynData) -> torch.Tensor:
        return w.x

class EncSmplAuto(Encoder):
    """Only encodes states."""
    GRAPH = False
    AUTO = True
    def forward(self, w: DynData) -> torch.Tensor:
        return self.net(w.x)

class EncSmplCtrl(Encoder):
    """Encodes states and controls."""
    GRAPH = False
    AUTO = False
    def forward(self, w: DynData) -> torch.Tensor:
        return self.net(torch.cat([w.x, w.u], dim=-1))

class EncGraphIden(Encoder):
    """Identity transform."""
    GRAPH = True
    AUTO = True
    def forward(self, w: DynData) -> torch.Tensor:
        return w.xg

class EncGraphAuto(Encoder):
    """Using GNN in EncAuto."""
    GRAPH = True
    AUTO = True
    def forward(self, w: DynData) -> torch.Tensor:
        return w.g(self.net(w.xg, w.ei, w.ew, w.ea))

class EncGraphCtrl(Encoder):
    """Using GNN in EncCtrl."""
    GRAPH = True
    AUTO = False
    def forward(self, w: DynData) -> torch.Tensor:
        xu_cat = torch.cat([w.xg, w.ug], dim=-1)
        return w.g(self.net(xu_cat, w.ei, w.ew, w.ea))

class EncNodeAuto(Encoder):
    """Using EncAuto for each node of graph."""
    GRAPH = True
    AUTO = True
    def forward(self, w: DynData) -> torch.Tensor:
        return w.G(self.net(w.xg))      # G is needed for unified data structure

class EncNodeCtrl(Encoder):
    """Using EncCtrl for each node of graph."""
    GRAPH = True
    AUTO = False
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
    GRAPH = False
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return z

class DecAuto(Decoder):
    """Only decodes states."""
    GRAPH = False
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(z)

class DecGraph(Decoder):
    """Decode by GNN."""
    GRAPH = True
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(z, w.ei, w.ew, w.ea)

class DecNode(Decoder):
    """Using DecAuto for each node of graph."""
    GRAPH = True
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
    cat = torch.cat([z, w.u], dim=-1)
    return cat

def zu_blin_no_const(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Compute bilinear features without constant term."""
    z_u = (z.unsqueeze(-1) * w.u.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    cat = torch.cat([z, z_u], dim=-1)
    return cat

def zu_blin_with_const(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Compute bilinear features with constant term."""
    z_u = (z.unsqueeze(-1) * w.u.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    cat = torch.cat([z, z_u, w.u], dim=-1)
    return cat

def zu_cat_smpl_graph(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Simple concatenation of z and u on graph."""
    cat = torch.cat([z, w.ug], dim=-1)
    return cat

def zu_blin_no_const_graph(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Compute bilinear features without constant term for graph data."""
    u_reshaped = w.ug
    z_u = (z.unsqueeze(-1) * u_reshaped.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    cat = torch.cat([z, z_u], dim=-1)
    return cat

def zu_blin_with_const_graph(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Compute bilinear features with constant term for graph data."""
    u_reshaped = w.ug
    z_u = (z.unsqueeze(-1) * u_reshaped.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    cat = torch.cat([z, z_u, u_reshaped], dim=-1)
    return cat

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
# Dynamics modules - processors
# ------------------

class ProcDirect(Processor):
    """Processing without control inputs."""
    GRAPH = False
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(self.zu_cat(z, w))

class ProcSkip(Processor):
    """Processing with skip connection."""
    GRAPH = False
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return z + self.net(self.zu_cat(z, w))

class ProcGraphDirect(Processor):
    """Processing by GNN."""
    GRAPH = True
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(w.g(self.zu_cat(z, w)), w.ei, w.ew, w.ea)   # G is effectively applied in the net

class ProcGraphSkip(Processor):
    """Processing by GNN with skip connection."""
    GRAPH = True
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return z + self.net(w.g(self.zu_cat(z, w)), w.ei, w.ew, w.ea)   # G is effectively applied in the net

PROC_MAP = {
    "direct"       : ProcDirect,
    "skip"         : ProcSkip,
    "graph_direct" : ProcGraphDirect,
    "graph_skip"   : ProcGraphSkip
}
