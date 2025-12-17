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

class DynBLinNoConst(Dynamics):
    GRAPH = False
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(zu_blin_no_const(z, w))

class DynBLinWithConst(Dynamics):
    GRAPH = False
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(zu_blin_with_const(z, w))

class DynSkipBLinNoConst(Dynamics):
    GRAPH = False
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return z + self.net(zu_blin_no_const(z, w))

class DynSkipBLinWithConst(Dynamics):
    GRAPH = False
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return z + self.net(zu_blin_with_const(z, w))


class DynGraph(Dynamics):
    GRAPH = True
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(w.g(z), w.ei, w.ew, w.ea)   # G is effectively applied in the net

class DynBLinNoConstGraph(Dynamics):
    GRAPH = True
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(zu_blin_no_const_graph(z, w))

class DynBLinWithConstGraph(Dynamics):
    GRAPH = True
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.net(zu_blin_with_const_graph(z, w))

class DynSkipBLinNoConstGraph(Dynamics):
    GRAPH = True
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return z + self.net(zu_blin_no_const_graph(z, w))

class DynSkipBLinWithConstGraph(Dynamics):
    GRAPH = True
    def forward(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return z + self.net(zu_blin_with_const_graph(z, w))

