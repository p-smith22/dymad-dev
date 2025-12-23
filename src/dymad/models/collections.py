from dataclasses import dataclass
from typing import Dict

from dymad.models.helpers import build_model
from dymad.models.recipes import CD_KM, CD_KMSK, CD_LDM, CD_LFM
from dymad.models.recipes_kmm import CD_KMM

@dataclass
class PredefinedModel:
    CONT: bool
    encoder: str
    feature: str
    dynamics: str
    decoder: str
    model_cls: object

    def __post_init__(self):
        self.GRAPH = \
            'graph' in self.encoder or 'node' in self.encoder or \
            'graph' in self.decoder or 'node' in self.decoder or \
            'graph' in self.dynamics

    def __call__(self,
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None):
        model_spec = [
            self.CONT,
            self.encoder,
            self.feature,
            self.dynamics,
            self.decoder,
            self.model_cls
        ]
        return build_model(model_spec, model_config, data_meta, dtype, device)

#                       CONT,  encoder, feature, dynamics, decoder, model_cls
LDM   = PredefinedModel(True,  "smpl",  "none",  "direct", "auto",  CD_LDM)
DLDM  = PredefinedModel(False, "smpl",  "none",  "direct", "auto",  CD_LDM)
GLDM  = PredefinedModel(True,  "smpl",  "none",  "direct", "auto",  CD_LDM)
DGLDM = PredefinedModel(False, "smpl",  "none",  "direct", "auto",  CD_LDM)
LDMG  = PredefinedModel(True,  "smpl",  "none",  "direct", "auto",  CD_LDM)
DLDMG = PredefinedModel(False, "smpl",  "none",  "direct", "auto",  CD_LDM)

#                       CONT,  encoder,      feature,      dynamics, decoder,      model_cls
KBF   = PredefinedModel(True,  "smpl_auto",  "blin",       "direct", "auto",       CD_LFM)
DKBF  = PredefinedModel(False, "smpl_auto",  "blin",       "direct", "auto",       CD_LFM)
GKBF  = PredefinedModel(True,  "graph_auto", "graph_blin", "direct", "graph_auto", CD_LFM)
DGKBF = PredefinedModel(False, "graph_auto", "graph_blin", "direct", "graph_auto", CD_LFM)

LTI   = PredefinedModel(True,  "smpl_auto",  "cat",        "direct", "auto",       CD_LFM)
DLTI  = PredefinedModel(False, "smpl_auto",  "cat",        "direct", "auto",       CD_LFM)
GLTI  = PredefinedModel(True,  "graph_auto", "graph_cat",  "direct", "graph_auto", CD_LFM)
DGLTI = PredefinedModel(False, "graph_auto", "graph_cat",  "direct", "graph_auto", CD_LFM)

#                        CONT,  encoder,      feature,      dynamics, decoder,      model_cls
KM     = PredefinedModel(True,  "smpl_auto",  "blin",       "direct", "auto",       CD_KM)
KMM    = PredefinedModel(True,  "smpl_auto",  "blin",       "direct", "auto",       CD_KMM)
DKM    = PredefinedModel(False, "smpl_auto",  "blin",       "direct", "auto",       CD_KM)
GKM    = PredefinedModel(True,  "graph_auto", "graph_blin", "direct", "graph_auto", CD_KM)
DGKM   = PredefinedModel(False, "graph_auto", "graph_blin", "direct", "graph_auto", CD_KM)
DKMSK  = PredefinedModel(False, "smpl_auto",  "blin",       "direct", "auto",       CD_KMSK)
DGKMSK = PredefinedModel(False, "graph_auto", "graph_blin", "direct", "graph_auto", CD_KMSK)
