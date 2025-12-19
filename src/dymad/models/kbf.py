
import torch
from typing import Dict, List, Tuple

from dymad.models.components import ENC_MAP, DYN_MAP, DEC_MAP, FZU_MAP
from dymad.models.helpers import build_autoencoder, build_kbf, build_predictor
from dymad.models.model_base import ComposedDynamics
from dymad.io import DynData

class ComposedDynamicsKBF(ComposedDynamics):
    def linear_features(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear features, f, and outputs, dz, for the model.

        dz = Af(z)

        dz is the output of the dynamics, z_dot for cont-time, z_next for disc-time.
        """
        z = self.encoder(w)
        return self.dynamics.zu_cat(z, w), z

    def linear_eval(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear evaluation, dz, and states, z, for the model.

        dz = Af(z)

        z is the encoded state, which will be used to compute the expected output.
        """
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        return z_dot, z


def build_kbf(
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
    const_term = model_config.get('const_term', True)

    if n_total_control_features > 0:
        if predictor_type == "exp":
            raise ValueError("Exponential predictor is not supported for model with control inputs.")
        if fzu_type == "blin":
            if const_term:
                fzu_type += '_with_const'  # Encoder with control, bilinear with const
            else:
                fzu_type += '_no_const'    # Encoder with control, bilinear without const
    else:
        fzu_type = "none"              # Encoder without control
    graph_ae = ENC_MAP[enc_type].GRAPH
    graph_dyn = DYN_MAP[dyn_type].GRAPH
    assert ENC_MAP[enc_type].GRAPH == DEC_MAP[dec_type].GRAPH, "Encoder/Decoder graph compatibility mismatch."

    encoder_net, decoder_net, enc_out_dim, dec_inp_dim = build_autoencoder(
        model_config,
        latent_dimension, n_total_features, n_total_state_features,
        dtype, device,
        ifgnn = graph_ae)

    dynamics_net = build_kbf(
            model_config,
            latent_dimension, enc_out_dim, dec_inp_dim,
            dtype, device,
            ifgnn = graph_dyn)

    predict = build_predictor(cont, input_order, predictor_type)

    model = ComposedDynamicsKBF(
        encoder  = ENC_MAP[enc_type](encoder_net),
        dynamics = DYN_MAP[dyn_type](dynamics_net),
        decoder  = DEC_MAP[dec_type](decoder_net),
        predict  = predict)
    model.CONT  = cont
    model.GRAPH = graph_ae or graph_dyn
    model.dynamics.zu_cat = FZU_MAP[fzu_type]

    model.set_linear_weights = dynamics_net.set_weights

    return model


#        CONT,  encoder,      zu_cat,       dynamics, decoder
KBF   = [True,  "smpl_auto",  "blin",       "direct", "auto"]
DKBF  = [False, "smpl_auto",  "blin",       "direct", "auto"]
GKBF  = [True,  "graph_auto", "graph_blin", "direct", "graph_auto"]
DGKBF = [False, "graph_auto", "graph_blin", "direct", "graph_auto"]

LTI   = [True,  "smpl_auto",  "cat",        "direct", "auto"]
DLTI  = [False, "smpl_auto",  "cat",        "direct", "auto"]
GLTI  = [True,  "graph_auto", "graph_cat",  "direct", "graph_auto"]
DGLTI = [False, "graph_auto", "graph_cat",  "direct", "graph_auto"]
