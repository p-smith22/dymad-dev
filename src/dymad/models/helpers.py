import copy
import logging
from typing import Dict, List

from dymad.models.components import ENC_MAP, DEC_MAP, FZU_MAP, DYN_MAP
from dymad.models.prediction import predict_continuous, predict_continuous_exp, predict_continuous_np, \
    predict_discrete, predict_discrete_exp
from dymad.modules import make_autoencoder

logger = logging.getLogger(__name__)

def get_dims(model_config, data_meta):
    # Basic dimensions
    dim_x  = data_meta.get('n_total_state_features')
    dim_u  = data_meta.get('n_total_control_features')
    dim_e  = dim_x + dim_u

    dim_l  = model_config.get('latent_dimension', 64)
    n_enc  = model_config.get('encoder_layers', 2)
    n_dec  = model_config.get('decoder_layers', 2)
    n_prc  = model_config.get('processor_layers', 2)

    # Derived dimensions - default options
    dim_z = dim_l if n_enc > 0 else dim_e
    dim_r = dim_s = dim_z
    dims = {
        'x'  : dim_x,
        'u'  : dim_u,
        'e'  : dim_e,   # Input dim to encoder
        'z'  : dim_z,
        's'  : dim_s,
        'r'  : dim_r,
        'l'  : dim_l,
        'enc': n_enc,
        'dec': n_dec,
        'prc': n_prc
    }
    return dims

def build_autoencoder(
        model_config, dims,
        dtype, device,
        ifgnn = False):
    # Determine other options for MLP layers
    opts = {
        'activation'     : model_config.get('activation', 'prelu'),
        'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
        'bias_init'      : model_config.get('bias_init', 'zeros'),
        'gain'           : model_config.get('gain', 1.0),
        'end_activation' : model_config.get('end_activation', True),
        'dtype'          : dtype,
        'device'         : device
    }
    if ifgnn:
        opts['gcl']      = model_config.get('gcl', 'sage')
        opts['gcl_opts'] = model_config.get('gcl_opts', {})
    aec_type = model_config.get('autoencoder_type', 'smp')

    # Build encoder/decoder networks
    pref = "gnn_" if ifgnn else "mlp_"
    encoder_net, decoder_net = make_autoencoder(
        type       = pref+aec_type,
        input_dim  = dims['e'],
        latent_dim = dims['l'],
        hidden_dim = dims['z'],
        enc_depth  = dims['enc'],
        dec_depth  = dims['dec'],
        output_dim = dims['x'],
        **opts
    )

    return encoder_net, decoder_net


def build_predictor(CONT, predictor_type, n_total_control_features):
    if CONT:
        if predictor_type == "exp":
            # Does not support inputs
            assert n_total_control_features == 0, "Exponential predictor does not support control inputs."
            return predict_continuous_exp
        elif predictor_type == "np":
            return predict_continuous_np
        else:
            return predict_continuous
    else:
        if predictor_type == "exp":
            return predict_discrete_exp
        else:
            return predict_discrete


def fzu_selector(fzu_type, n_total_control_features, const_term):
    _type = fzu_type
    if n_total_control_features > 0:
        if fzu_type in ["blin", "graph_blin"]:
            if const_term:
                _type += '_with_const'  # Encoder with control, bilinear with const
            else:
                _type += '_no_const'    # Encoder with control, bilinear without const
    else:
        _type = "none"                  # Encoder without control
    assert _type in FZU_MAP, f"Unknown zu_cat type {_type}."
    return _type


def build_model(
        model_spec: List,
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None):
    cont, enc_type, fzu_type, dyn_type, dec_type, model_cls = model_spec

    # Validate graph compatibility
    graph_ae  = enc_type.startswith("graph")
    graph_dyn = dyn_type.startswith("graph")
    tmp = dec_type.startswith("graph")
    assert graph_ae == tmp, "Encoder/Decoder graph compatibility mismatch."

    # Class specific processing
    dims, (enc_type, fzu_type, dec_type, prd_type), processor_net, input_order = \
        model_cls.build_core(
            (enc_type, fzu_type, dec_type),
            model_config, data_meta,
            dtype, device, ifgnn = graph_dyn)

    # Autoencoder
    encoder_net, decoder_net = build_autoencoder(
        model_config, dims,
        dtype, device,
        ifgnn = graph_ae)

    # Prediction
    predict = build_predictor(cont, prd_type, dims['u'])

    # The full model
    model = model_cls(
        encoder  = ENC_MAP[enc_type](encoder_net),
        dynamics = DYN_MAP[dyn_type](processor_net),
        decoder  = DEC_MAP[dec_type](decoder_net),
        predict  = (predict, input_order),
        model_config = copy.deepcopy(model_config),
        dims     = copy.deepcopy(dims)
    )
    model.CONT  = cont
    model.GRAPH = graph_ae or graph_dyn
    model.dynamics.features = FZU_MAP[fzu_type]

    logger.info(f"Built model: {model_cls.__name__}")
    logger.info(f"- Encoder: {enc_type}, Dynamics: {dyn_type}, Decoder: {dec_type}, Features: {fzu_type}")
    logger.info(f"- Using predictor: {predict.__name__}, input order: {input_order}")
    logger.info(f"- If graph model: {model.GRAPH}, continuous-time: {model.CONT}")

    return model
