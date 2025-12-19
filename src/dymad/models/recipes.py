import torch
from typing import Tuple

from dymad.models.components import FZU_MAP
from dymad.models.helpers import fzu_selector
from dymad.models.model_base import ComposedDynamics
from dymad.modules import FlexLinear, GNN, make_krr, MLP
from dymad.io import DynData

class CD_LDM(ComposedDynamics):

    @classmethod
    def build_core(cls, model_config, enc_type, fzu_type, dec_type, dtype, device, ifgnn=False):
        n_total_control_features = model_config.get('n_total_control_features')
        enc_out_dim = model_config.get('enc_out_dim')
        latent_dimension = model_config.get('latent_dimension')
        dec_inp_dim = model_config.get('dec_inp_dim')

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

        if n_total_control_features > 0:
            enc_type += '_ctrl'  # Encoder with control
        else:
            enc_type += '_auto'  # Encoder without control
        fzu_func = FZU_MAP[fzu_type]

        return dynamics_net, enc_type, fzu_func, dec_type


class CD_LFM(ComposedDynamics):

    @classmethod
    def build_core(cls, model_config, enc_type, fzu_type, dec_type, dtype, device, ifgnn=False):
        koopman_dimension = model_config.get('koopman_dimension')
        n_total_control_features = model_config.get('n_total_control_features')
        const_term = model_config.get('const_term', True)
        blin_term = model_config.get('blin_term', True)

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

        fzu_func = fzu_selector(fzu_type, n_total_control_features, const_term)

        return dynamics_net, enc_type, fzu_func, dec_type

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

    def set_linear_weights(self, W: torch.Tensor) -> None:
        """Set the weights of the linear dynamics module."""
        self.dynamics.net.set_linear_weights(W)


class CD_KM(ComposedDynamics):

    @classmethod
    def build_core(cls, model_config, enc_type, fzu_type, dec_type, dtype, device, ifgnn=False):
        n_total_control_features = model_config.get('n_total_control_features')
        const_term = model_config.get('const_term', True)

        opts = {
            'type'       : model_config.get('type', 'share'),
            'kernel'     : model_config.get('kernel', None),
            'ridge_init' : model_config.get('ridge_init', 1e-10),
            'jitter'     : model_config.get('jitter', 1e-12),
            'dtype'      : dtype,
            'device'     : device
        }
        dynamics_net = make_krr(**opts)

        fzu_func = fzu_selector(fzu_type, n_total_control_features, const_term)

        return dynamics_net, enc_type, fzu_func, dec_type

    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit the kernel dynamics using input-output pairs.
        """
        self.dynamics.net.set_train_data(inp, out)
        residual = self.dynamics.net.fit()
        return self.dynamics.net._alphas, residual

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        KRR relies on data, and when initialized some parameters are placeholders.
        Here we first update the shapes of those parameters to match the checkpoint,
        then call the standard load_state_dict to load values and do checks.
        """
        with torch.no_grad():
            for name, p in self.named_parameters(recurse=True):
                if name in state_dict:
                    saved = state_dict[name]
                    if p.shape != saved.shape:
                        # keep the same Parameter object (so it's still registered,
                        # and any optimizer state tied to id(p) can be preserved)
                        p.set_(torch.empty_like(saved))

        # Then do standard loading and checks
        return super().load_state_dict(state_dict, strict=strict)


class CD_KMSK(CD_KM):
    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        self.dynamics.net.set_train_data(inp, out-inp[..., :self.kernel_dimension])
        residual = self.dynamics.net.fit()
        return self.dynamics.net._alphas, residual

