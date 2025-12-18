
import torch
from typing import Dict, Tuple

from dymad.models.components import \
    EncAuto, EncAutoGraph, EncAutoNode, \
    DecAuto, DecGraph, DecNode, \
    DynAuto, DynBLinNoConst, DynBLinNoConstGraph, DynBLinWithConst, DynBLinWithConstGraph, \
    DynGraph
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
        return self.dynamics.zu_blin(z, w), z

    def linear_eval(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear evaluation, dz, and states, z, for the model.

        dz = Af(z)

        z is the encoded state, which will be used to compute the expected output.
        """
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        return z_dot, z


def build_kbf(
        encoder_cls, dynamics_cls_list, decoder_cls,
        model_config: Dict, data_meta: Dict,
        cont: bool, dtype=None, device=None):
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
        if const_term:
            dynamics_cls = dynamics_cls_list[2]  # Encoder with control, bilinear with const
        else:
            dynamics_cls = dynamics_cls_list[1]  # Encoder with control, bilinear without const
    else:
        dynamics_cls = dynamics_cls_list[0]  # Encoder without control
    graph_ae = encoder_cls.GRAPH
    graph_dyn = dynamics_cls.GRAPH
    assert encoder_cls.GRAPH == decoder_cls.GRAPH, "Encoder/Decoder graph compatibility mismatch."

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
        encoder  = encoder_cls(encoder_net),
        dynamics = dynamics_cls(dynamics_net),
        decoder  = decoder_cls(decoder_net),
        predict  = predict)
    model.CONT  = cont
    model.GRAPH = graph_ae or graph_dyn

    model.set_linear_weights = dynamics_net.set_weights

    return model

def KBF(
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None) -> ComposedDynamics:
    """Koopman Bilinear Form (KBF) model.

    Uses MLP encoder/decoder and KBF operators for dynamics.
    """
    return build_kbf(
        EncAuto, [DynAuto, DynBLinNoConst, DynBLinWithConst], DecAuto,
        model_config, data_meta,
        cont = True,
        dtype=dtype, device=device)

def DKBF(
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None) -> ComposedDynamics:
    """Discrete-time version of KBF.
    """
    return build_kbf(
        EncAuto, [DynAuto, DynBLinNoConst, DynBLinWithConst], DecAuto,
        model_config, data_meta,
        cont = False,
        dtype=dtype, device=device)

def GKBF(
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None) -> ComposedDynamics:
    """Graph Koopman Bilinear Form (GKBF) model.

    Uses GNN encoder/decoder and KBF operators for dynamics.

    Koopman dimension is defined per node.
    """
    return build_kbf(
        EncAutoGraph, [DynGraph, DynBLinNoConstGraph, DynBLinWithConstGraph], DecGraph,
        model_config, data_meta,
        cont = True,
        dtype=dtype, device=device)

def DGKBF(
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None) -> ComposedDynamics:
    """Discrete-time version of GKBF.
    """
    return build_kbf(
        EncAutoGraph, [DynGraph, DynBLinNoConstGraph, DynBLinWithConstGraph], DecGraph,
        model_config, data_meta,
        cont = False,
        dtype=dtype, device=device)

