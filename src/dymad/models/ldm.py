from typing import Dict, List

from dymad.models.components import ENC_MAP, DYN_MAP, DEC_MAP, FZU_MAP
from dymad.models.helpers import build_autoencoder, build_predictor, build_processor
from dymad.models.model_base import ComposedDynamics


def build_ldm(
        model_spec: List,
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None):
    cont, enc_type, fzu_type, dyn_type, dec_type = model_spec

    n_total_state_features = data_meta.get('n_total_state_features')
    n_total_control_features = data_meta.get('n_total_control_features')
    n_total_features = data_meta.get('n_total_features')

    latent_dimension = model_config.get('latent_dimension', 64)
    input_order = model_config.get('input_order', 'cubic')
    predictor_type = model_config.get('predictor_type', 'ode')

    if n_total_control_features > 0:
        if predictor_type == "exp":
            raise ValueError("Exponential predictor is not supported for model with control inputs.")
        enc_type += '_ctrl'  # Encoder with control
    else:
        enc_type += '_auto'  # Encoder without control
    graph_ae = ENC_MAP[enc_type].GRAPH
    graph_dyn = DYN_MAP[dyn_type].GRAPH
    assert ENC_MAP[enc_type].GRAPH == DEC_MAP[dec_type].GRAPH, "Encoder/Decoder graph compatibility mismatch."

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
        encoder  = ENC_MAP[enc_type](encoder_net),
        dynamics = DYN_MAP[dyn_type](dynamics_net),
        decoder  = DEC_MAP[dec_type](decoder_net),
        predict  = predict)
    model.CONT  = cont
    model.GRAPH = graph_ae or graph_dyn
    model.dynamics.zu_cat = FZU_MAP[fzu_type]

    return model

#        CONT, encoder, zu_cat, dynamics, decoder
LDM   = [True,  "smpl",  "none", "direct", "auto"]
DLDM  = [False, "smpl",  "none", "direct", "auto"]
GLDM  = [True,  "graph", "none", "direct", "graph"]
DGLDM = [False, "graph", "none", "direct", "graph"]
LDMG  = [True,  "node",  "none", "graph_direct", "node"]
DLDMG = [False, "node",  "none", "graph_direct", "node"]

