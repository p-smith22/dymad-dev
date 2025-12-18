import torch

from dymad.io import DynData
from dymad.models.model_base import Decoder, Dynamics, Encoder

# ------------------
# Encoder modules
# ------------------

class EncAuto(Encoder):
    GRAPH = False
    AUTO = True
    def forward(self, w: DynData) -> torch.Tensor:
        return self.net(w.x)

class EncCtrl(Encoder):
    GRAPH = False
    AUTO = False
    def forward(self, w: DynData) -> torch.Tensor:
        return self.net(torch.cat([w.x, w.u], dim=-1))

class EncAutoCat(Encoder):
    GRAPH = False
    AUTO = True
    def forward(self, w: DynData) -> torch.Tensor:
        return torch.cat([w.x, self.net(w.x)], dim=-1)

class EncCtrlCat(Encoder):
    GRAPH = False
    AUTO = False
    def forward(self, w: DynData) -> torch.Tensor:
        return torch.cat(
            [w.x, self.net(torch.cat([w.x, w.u], dim=-1))],
            dim=-1)

class EncAutoGraph(Encoder):
    GRAPH = True
    AUTO = True
    def forward(self, w: DynData) -> torch.Tensor:
        return w.g(self.net(w.xg, w.ei, w.ew, w.ea))

class EncCtrlGraph(Encoder):
    GRAPH = True
    AUTO = False
    def forward(self, w: DynData) -> torch.Tensor:
        xu_cat = torch.cat([w.xg, w.ug], dim=-1)
        return w.g(self.net(xu_cat, w.ei, w.ew, w.ea))

class EncAutoNode(Encoder):
    GRAPH = True
    AUTO = True
    def forward(self, w: DynData) -> torch.Tensor:
        return w.G(self.net(w.xg))      # G is needed for unified data structure

class EncCtrlNode(Encoder):
    GRAPH = True
    AUTO = False
    def forward(self, w: DynData) -> torch.Tensor:
        xu_cat = torch.cat([w.xg, w.ug], dim=-1)
        return w.G(self.net(xu_cat))    # G is needed for unified data structure


# ------------------
# Decoder modules
# ------------------

class DecAuto(Decoder):
    GRAPH = False

class DecGraph(Decoder):
    GRAPH = True
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(z, w.ei, w.ew, w.ea)

class DecNode(Decoder):
    GRAPH = True
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return w.G(self.net(w.g(z)))    # G is needed for unified data structure


# ------------------
# Dynamics modules
# ------------------

def zu_blin_no_const(z: torch.Tensor, w: DynData) -> torch.Tensor:
    z_u = (z.unsqueeze(-1) * w.u.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    cat = torch.cat([z, z_u], dim=-1)
    return cat

def zu_blin_with_const(z: torch.Tensor, w: DynData) -> torch.Tensor:
    z_u = (z.unsqueeze(-1) * w.u.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    cat = torch.cat([z, z_u, w.u], dim=-1)
    return cat

def zu_blin_no_const_graph(z: torch.Tensor, w: DynData) -> torch.Tensor:
    u_reshaped = w.ug
    z_u = (z.unsqueeze(-1) * u_reshaped.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    cat = torch.cat([z, z_u], dim=-1)
    return cat

def zu_blin_with_const_graph(z: torch.Tensor, w: DynData) -> torch.Tensor:
    u_reshaped = w.ug
    z_u = (z.unsqueeze(-1) * u_reshaped.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    cat = torch.cat([z, z_u, u_reshaped], dim=-1)
    return cat


class DynAuto(Dynamics):
    GRAPH = False

class DynGraph(Dynamics):
    GRAPH = True
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(w.g(z), w.ei, w.ew, w.ea)   # G is effectively applied in the net


class DynBLin(Dynamics):
    GRAPH = None
    zu_blin = None
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(self.zu_blin(z, w))

class DynBLinNoConst(DynBLin):
    GRAPH = False
    zu_blin = zu_blin_no_const

class DynBLinWithConst(DynBLin):
    GRAPH = False
    zu_blin = zu_blin_with_const

class DynBLinNoConstGraph(DynBLin):
    GRAPH = True
    zu_blin = zu_blin_no_const_graph

class DynBLinWithConstGraph(DynBLin):
    GRAPH = True
    zu_blin = zu_blin_with_const_graph


class DynSkipBLin(Dynamics):
    GRAPH = None
    zu_blin = None
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return z + self.net(self.zu_blin(z, w))

class DynSkipBLinNoConst(DynSkipBLin):
    GRAPH = False
    zu_blin = zu_blin_no_const

class DynSkipBLinWithConst(DynSkipBLin):
    GRAPH = False
    zu_blin = zu_blin_with_const

class DynSkipBLinNoConstGraph(DynSkipBLin):
    GRAPH = True
    zu_blin = zu_blin_no_const_graph

class DynSkipBLinWithConstGraph(DynSkipBLin):
    GRAPH = True
    zu_blin = zu_blin_with_const_graph
