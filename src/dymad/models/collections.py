from dataclasses import dataclass
from typing import Dict

from dymad.models.helpers import build_model
from dymad.models.recipes import CD_KM, CD_KMSK, CD_LDM, CD_LFM
from dymad.models.recipes_kmm import CD_KMM

@dataclass
class PredefinedModel:
    CONT: bool
    GRAPH: bool
    encoder: str
    feature: str
    dynamics: str
    decoder: str
    model_cls: object

    def __call__(self,
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None):
        model_spec = [
            self.CONT,
            self.GRAPH,
            self.encoder,
            self.feature,
            self.dynamics,
            self.decoder,
            self.model_cls
        ]
        return build_model(model_spec, model_config, data_meta, dtype, device)

#                       CONT,  GRAPH, encoder, feature, dynamics, decoder, model_cls
LDM   = PredefinedModel(True,  False, "smpl",  "none",  "direct", "auto",  CD_LDM)
DLDM  = PredefinedModel(False, False, "smpl",  "none",  "direct", "auto",  CD_LDM)
GLDM  = PredefinedModel(True,  True,  "smpl",  "none",  "direct", "auto",  CD_LDM)
DGLDM = PredefinedModel(False, True,  "smpl",  "none",  "direct", "auto",  CD_LDM)
LDMG  = PredefinedModel(True,  True,  "smpl",  "none",  "direct", "auto",  CD_LDM)
DLDMG = PredefinedModel(False, True,  "smpl",  "none",  "direct", "auto",  CD_LDM)

#                       CONT,  GRAPH, encoder,      feature,      dynamics, decoder,      model_cls
KBF   = PredefinedModel(True,  False, "smpl_auto",  "blin",       "direct", "auto",       CD_LFM)
DKBF  = PredefinedModel(False, False, "smpl_auto",  "blin",       "direct", "auto",       CD_LFM)
GKBF  = PredefinedModel(True,  True,  "graph_auto", "graph_blin", "direct", "graph_auto", CD_LFM)
DGKBF = PredefinedModel(False, True,  "graph_auto", "graph_blin", "direct", "graph_auto", CD_LFM)

LTI   = PredefinedModel(True,  False, "smpl_auto",  "cat",        "direct", "auto",       CD_LFM)
DLTI  = PredefinedModel(False, False, "smpl_auto",  "cat",        "direct", "auto",       CD_LFM)
GLTI  = PredefinedModel(True,  True,  "graph_auto", "graph_cat",  "direct", "graph_auto", CD_LFM)
DGLTI = PredefinedModel(False, True,  "graph_auto", "graph_cat",  "direct", "graph_auto", CD_LFM)

#                        CONT,  GRAPH, encoder,      feature,      dynamics, decoder,      model_cls
KM     = PredefinedModel(True,  False, "smpl_auto",  "blin",       "direct", "auto",       CD_KM)
KMM    = PredefinedModel(True,  False, "smpl_auto",  "blin",       "direct", "auto",       CD_KMM)
DKM    = PredefinedModel(False, False, "smpl_auto",  "blin",       "direct", "auto",       CD_KM)
GKM    = PredefinedModel(True,  True,  "graph_auto", "graph_blin", "direct", "graph_auto", CD_KM)
DGKM   = PredefinedModel(False, True,  "graph_auto", "graph_blin", "direct", "graph_auto", CD_KM)
DKMSK  = PredefinedModel(False, False, "smpl_auto",  "blin",       "direct", "auto",       CD_KMSK)
DGKMSK = PredefinedModel(False, True,  "graph_auto", "graph_blin", "direct", "graph_auto", CD_KMSK)
