from dymad.models.prediction import predict_continuous, predict_continuous_exp, predict_discrete, predict_discrete_exp
from dymad.modules import FlexLinear, GNN, make_autoencoder, MLP

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


def build_processor(
        model_config,
        latent_dimension, enc_out_dim, dec_inp_dim,
        dtype, device,
        ifgnn = False):
    proc_depth = model_config.get('processor_layers', 2)
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

    MDL = GNN if ifgnn else MLP
    dynamics_net = MDL(
        input_dim  = enc_out_dim,
        latent_dim = latent_dimension,
        output_dim = dec_inp_dim,
        n_layers   = proc_depth,
        **opts
    )

    return dynamics_net


def build_kbf(
        koopman_dimension, n_total_control_features, const_term, blin_term,
        dtype, device):
    if n_total_control_features > 0:
        if blin_term:
            if const_term:
                dyn_dim = koopman_dimension * (n_total_control_features + 1) + n_total_control_features
            else:
                dyn_dim = koopman_dimension * (n_total_control_features + 1)
        else:
            dyn_dim = koopman_dimension + n_total_control_features
    else:
        dyn_dim = koopman_dimension
    dynamics_net = FlexLinear(dyn_dim, koopman_dimension, bias=False, dtype=dtype, device=device)

    return dynamics_net


def build_predictor(CONT, input_order, predictor_type):
    if CONT:
        if predictor_type == "exp":
            # Does not support inputs
            def predict(model, x0, w, ts, method, **kwargs):
                return predict_continuous_exp(
                    model, x0, ts, w, method=method, **kwargs)
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
