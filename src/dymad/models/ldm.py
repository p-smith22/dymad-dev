from typing import Dict

from dymad.models.components import \
    EncAuto, EncAutoGraph, EncAutoNode, EncCtrl, EncCtrlGraph, EncCtrlNode, \
    DecAuto, DecGraph, DecNode, \
    DynAuto, DynGraph
from dymad.models.helpers import build_autoencoder, build_predictor, build_processor
from dymad.models.model_base import ComposedDynamics


def build_ldm(
        encoder_cls_list, dynamics_cls, decoder_cls,
        model_config: Dict, data_meta: Dict,
        cont: bool, dtype=None, device=None):
    n_total_state_features = data_meta.get('n_total_state_features')
    n_total_control_features = data_meta.get('n_total_control_features')
    n_total_features = data_meta.get('n_total_features')

    latent_dimension = model_config.get('latent_dimension', 64)
    input_order = model_config.get('input_order', 'cubic')
    predictor_type = model_config.get('predictor_type', 'ode')

    if n_total_control_features > 0:
        if predictor_type == "exp":
            raise ValueError("Exponential predictor is not supported for model with control inputs.")
        encoder_cls = encoder_cls_list[1]  # Encoder with control
    else:
        encoder_cls = encoder_cls_list[0]  # Encoder without control
    graph_ae = encoder_cls.GRAPH
    graph_dyn = dynamics_cls.GRAPH
    assert encoder_cls.GRAPH == decoder_cls.GRAPH, "Encoder/Decoder graph compatibility mismatch."

    encoder_net, decoder_net, enc_out_dim, dec_inp_dim = build_autoencoder(
        model_config,
        latent_dimension, n_total_features, n_total_state_features,
        dtype, device,
        ifgnn = graph_ae)

    dynamics_net = build_processor(
            model_config,
            latent_dimension, enc_out_dim, dec_inp_dim,
            dtype, device,
            ifgnn = graph_dyn)

    predict = build_predictor(cont, input_order, predictor_type)

    model = ComposedDynamics(
        encoder  = encoder_cls(encoder_net),
        dynamics = dynamics_cls(dynamics_net),
        decoder  = decoder_cls(decoder_net),
        predict  = predict)
    model.CONT  = cont
    model.GRAPH = graph_ae or graph_dyn

    return model

def LDM(
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None) -> ComposedDynamics:
    """Latent Dynamics Model (LDM)

    The encoder, dynamics, and decoder networks are implemented as MLPs.
    """
    return build_ldm(
        [EncAuto, EncCtrl], DynAuto, DecAuto,
        model_config, data_meta,
        cont = True,
        dtype=dtype, device=device)

def DLDM(
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None) -> ComposedDynamics:
    """Discrete-time version of LDM.
    """
    return build_ldm(
        [EncAuto, EncCtrl], DynAuto, DecAuto,
        model_config, data_meta,
        cont = False,
        dtype=dtype, device=device)

def GLDM(
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None) -> ComposedDynamics:
    """Graph Latent Dynamics Model (GLDM).

    Uses GNN for encoder/decoder and MLP for dynamics.
    """
    return build_ldm(
        [EncAutoGraph, EncCtrlGraph], DynAuto, DecGraph,
        model_config, data_meta,
        cont = True,
        dtype=dtype, device=device)

def DGLDM(
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None) -> ComposedDynamics:
    """Discrete-time version of GLDM.
    """
    return build_ldm(
        [EncAutoGraph, EncCtrlGraph], DynAuto, DecGraph,
        model_config, data_meta,
        cont = False,
        dtype=dtype, device=device)

def LDMG(
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None) -> ComposedDynamics:
    """Latent Dynamics Model on Graph (LDMG).

    Uses MLP for node-wise encoder/decoder and GNN for dynamics.
    """
    return build_ldm(
        [EncAutoNode, EncCtrlNode], DynGraph, DecNode,
        model_config, data_meta,
        cont = True,
        dtype=dtype, device=device)

def DLDMG(
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None) -> ComposedDynamics:
    """Discrete-time version of LDMG.
    """
    return build_ldm(
        [EncAutoNode, EncCtrlNode], DynGraph, DecNode,
        model_config, data_meta,
        cont = False,
        dtype=dtype, device=device)
