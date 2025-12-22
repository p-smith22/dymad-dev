import copy
from typing import Dict, List

from dymad.models.components import ENC_MAP, DEC_MAP, FZU_MAP, PROC_MAP
from dymad.models.prediction import predict_continuous, predict_continuous_exp, predict_continuous_np, \
    predict_discrete, predict_discrete_exp
from dymad.modules import make_autoencoder

def build_autoencoder(
        model_config,
        latent_dimension, n_total_features, n_total_state_features,
        dtype, device,
        ifgnn = False):
    # Get layer depths from config
    enc_depth = model_config.get('encoder_layers', 2)
    dec_depth = model_config.get('decoder_layers', 2)

    # Determine dimensions
    enc_out_dim = latent_dimension if enc_depth > 0 else n_total_features
    dec_inp_dim = latent_dimension if dec_depth > 0 else n_total_features

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
        input_dim  = n_total_features,
        latent_dim = latent_dimension,
        hidden_dim = enc_out_dim,
        enc_depth  = enc_depth,
        dec_depth  = dec_depth,
        output_dim = n_total_state_features,
        **opts
    )

    return encoder_net, decoder_net, enc_out_dim, dec_inp_dim


def build_predictor(CONT, input_order, predictor_type, n_total_control_features):
    if CONT:
        if predictor_type == "exp":
            # Does not support inputs
            assert n_total_control_features == 0, "Exponential predictor does not support control inputs."
            def predict(model, x0, w, ts, method, **kwargs):
                return predict_continuous_exp(
                    model, x0, ts, w, method=method, **kwargs)
        elif predictor_type == "np":
            def predict(model, x0, w, ts, method, order=input_order, **kwargs):
                return predict_continuous_np(
                    model, x0, ts, w, method=method, order=order, **kwargs)
        else:
            def predict(model, x0, w, ts, method, order=input_order, **kwargs):
                return predict_continuous(
                    model, x0, ts, w, method=method, order=order, **kwargs)
    else:
        if predictor_type == "exp":
            # Does not support inputs
            def predict(model, x0, w, ts, method, **kwargs):
                return predict_discrete_exp(
                    model, x0, ts, w, method=method, **kwargs)
        else:
            def predict(model, x0, w, ts, method, order=input_order, **kwargs):
                return predict_discrete(
                    model, x0, ts, w, method=method, order=order, **kwargs)

    return predict


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
    return FZU_MAP[_type]


def build_model(
        model_spec: List,
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None):
    cont, graph, enc_type, fzu_type, proc_type, dec_type, model_cls = model_spec

    n_total_state_features = data_meta.get('n_total_state_features')
    n_total_control_features = data_meta.get('n_total_control_features')
    n_total_features = data_meta.get('n_total_features')

    latent_dimension = model_config.get('latent_dimension', 64)
    input_order = model_config.get('input_order', 'cubic')
    prd_type = model_config.get('predictor_type', 'ode')

    graph_ae   = ENC_MAP[enc_type].GRAPH
    graph_proc = PROC_MAP[proc_type].GRAPH
    assert ENC_MAP[enc_type].GRAPH == DEC_MAP[dec_type].GRAPH, "Encoder/Decoder graph compatibility mismatch."
    assert graph == (graph_ae or graph_proc), f"Model spec graph compatibility mismatch, got {graph}/{graph_ae or graph_proc}."

    encoder_net, decoder_net, enc_out_dim, dec_inp_dim = build_autoencoder(
        model_config,
        latent_dimension, n_total_features, n_total_state_features,
        dtype, device,
        ifgnn = graph_ae)

    _cfg = copy.deepcopy(model_config)
    _cfg['n_total_control_features'] = n_total_control_features
    _cfg['n_total_state_features'] = n_total_state_features
    _cfg['n_total_features'] = n_total_features
    _cfg['enc_out_dim'] = enc_out_dim
    _cfg['latent_dimension'] = latent_dimension
    _cfg['dec_inp_dim'] = dec_inp_dim
    _cfg['cont'] = cont
    _cfg['types'] = (enc_type, fzu_type, dec_type, prd_type)
    processor_net, (enc_type, fzu_func, dec_type, prd_type) = model_cls.build_core(
        _cfg, dtype, device, ifgnn = graph_proc)

    predict = build_predictor(
        cont, input_order, prd_type, n_total_control_features)

    model = model_cls(
        encoder   = ENC_MAP[enc_type](encoder_net),
        processor = PROC_MAP[proc_type](processor_net),
        decoder   = DEC_MAP[dec_type](decoder_net),
        predict   = predict)
    model.CONT  = cont
    model.GRAPH = graph
    model.processor.zu_cat = fzu_func

    return model
