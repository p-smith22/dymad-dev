from dymad.models.recipes import CD_KM, CD_KMSK, CD_LDM, CD_LFM
from dymad.models.recipes_kmm import CD_KMM

#        CONT,  encoder, zu_cat, dynamics,       decoder, model_cls
LDM   = [True,  "smpl",  "none", "direct",       "auto",  CD_LDM]
DLDM  = [False, "smpl",  "none", "direct",       "auto",  CD_LDM]
GLDM  = [True,  "graph", "none", "direct",       "graph", CD_LDM]
DGLDM = [False, "graph", "none", "direct",       "graph", CD_LDM]
LDMG  = [True,  "node",  "none", "graph_direct", "node",  CD_LDM]
DLDMG = [False, "node",  "none", "graph_direct", "node",  CD_LDM]

#        CONT,  encoder,      zu_cat,       dynamics, decoder,      model_cls
KBF   = [True,  "smpl_auto",  "blin",       "direct", "auto",       CD_LFM]
DKBF  = [False, "smpl_auto",  "blin",       "direct", "auto",       CD_LFM]
GKBF  = [True,  "graph_auto", "graph_blin", "direct", "graph_auto", CD_LFM]
DGKBF = [False, "graph_auto", "graph_blin", "direct", "graph_auto", CD_LFM]

LTI   = [True,  "smpl_auto",  "cat",        "direct", "auto",       CD_LFM]
DLTI  = [False, "smpl_auto",  "cat",        "direct", "auto",       CD_LFM]
GLTI  = [True,  "graph_auto", "graph_cat",  "direct", "graph_auto", CD_LFM]
DGLTI = [False, "graph_auto", "graph_cat",  "direct", "graph_auto", CD_LFM]

#         CONT,  encoder,      zu_cat,       dynamics, decoder,      model_cls
KM     = [True,  "smpl_auto",  "blin",       "direct", "auto",       CD_KM]
KMM    = [True,  "smpl_auto",  "blin",       "direct", "auto",       CD_KMM]
DKM    = [False, "smpl_auto",  "blin",       "direct", "auto",       CD_KM]
GKM    = [True,  "graph_auto", "graph_blin", "direct", "graph_auto", CD_KM]
DGKM   = [False, "graph_auto", "graph_blin", "direct", "graph_auto", CD_KM]
DKMSK  = [False, "smpl_auto",  "blin",       "direct", "auto",       CD_KMSK]
DGKMSK = [False, "graph_auto", "graph_blin", "direct", "graph_auto", CD_KMSK]
