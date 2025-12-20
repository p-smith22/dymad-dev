import torch
from typing import Callable, Tuple

from dymad.models.components import FZU_MAP
from dymad.models.helpers import fzu_selector
from dymad.models.model_base import ComposedDynamics, Decoder, Processor, Encoder
from dymad.modules import FlexLinear, GNN, make_krr, MLP
from dymad.io import DynData

class CD_COR(ComposedDynamics):

    def __init__(
            self,
            encoder: Encoder,
            processor: Processor,
            decoder: Decoder,
            predict: Callable | None = None,
            model_config: dict | None = None):
        super().__init__(encoder, processor, decoder, predict, model_config)

        self.extra_setup()

    @classmethod
    def build_core(cls, model_config, dtype, device, ifgnn=False):
        n_total_control_features = model_config.get('n_total_control_features')
        n_total_state_features = model_config.get('n_total_state_features')
        n_total_features = model_config.get('n_total_features')
        latent_dimension = model_config.get('latent_dimension')
        residual_dimension = model_config.get('residual_dimension', n_total_state_features)
        residual_layers = model_config.get('residual_layers', 2)
        const_term = model_config.get('const_term', True)
        enc_type, fzu_type, dec_type, prd_type = model_config.get('types')
        cont = model_config.get('cont')

        prd_type = 'np' if cont else 'exp'

        opts = {
            'activation'     : model_config.get('activation', 'prelu'),
            'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
            'bias_init'      : model_config.get('bias_init', 'zeros'),
            'gain'           : model_config.get('gain', 1.0),
            'end_activation' : model_config.get('end_activation', True),
            'dtype'          : dtype,
            'device'         : device
        }
        processor_net = MLP(
            input_dim  = n_total_features,
            latent_dim = latent_dimension,
            output_dim = residual_dimension,
            n_layers   = residual_layers,
            **opts
        )

        fzu_func = fzu_selector(fzu_type, n_total_control_features, const_term)

        return processor_net, (enc_type, fzu_func, dec_type, prd_type)

    def extra_setup(self):
        """
        Additional setup in derived classes.  Called at end of __init__.
        """
        pass

    def base_dynamics(self, x: torch.Tensor, u: torch.Tensor, f: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        The minimal that the user must implement.

        `f` is the residual correction term computed by `self.residual`, and needs
        to be incorporated into the base dynamics.
        """
        raise NotImplementedError("Implement in derived class.")

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Compute corrected dynamics.
        """
        if z.ndim == 3:
            w_p = w.p.unsqueeze(-2)
        else:
            w_p = w.p
        _l = self.processor(z, w)
        _f = self.base_dynamics(z, w.u, _l, w_p)
        return _f

