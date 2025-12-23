import torch
from typing import Tuple

from dymad.models.helpers import fzu_selector, get_dims
from dymad.models.model_base import ComposedDynamics
from dymad.modules import FlexLinear, GNN, make_krr, MLP
from dymad.io import DynData

class CD_LDM(ComposedDynamics):

    @classmethod
    def build_core(cls, types, model_config, data_meta, dtype, device, ifgnn=False):
        enc_type, fzu_type, dec_type = types

        # Dimensions
        dims = get_dims(model_config, data_meta)

        # Autoencoder
        suffix = "_ctrl" if dims['u'] > 0 else "_auto"
        enc_type += suffix

        # Features in the dynamics
        # As is

        # Processor in the dynamics
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
        processor_net = MDL(
            input_dim  = dims['s'],
            latent_dim = dims['l'],
            output_dim = dims['r'],
            n_layers   = dims['prc'],
            **opts
        )

        # Prediction options
        input_order = model_config.get('input_order', 'cubic')
        prd_type = model_config.get('predictor_type', 'ode')

        return dims, (enc_type, fzu_type, dec_type, prd_type), processor_net, input_order


class CD_LFM(ComposedDynamics):

    @classmethod
    def build_core(cls, types, model_config, data_meta, dtype, device, ifgnn=False):
        enc_type, fzu_type, dec_type = types

        # Options
        const_term = model_config.get('const_term', True)

        # Dimensions
        dims = get_dims(model_config, data_meta)
        dims['e'] = dims['x']
        dims['z'] = model_config.get('koopman_dimension')

        blin_term = 'blin' in fzu_type
        if dims['u'] > 0:
            if blin_term:
                if const_term:
                    dyn_dim = dims['z'] * (dims['u'] + 1) + dims['u']
                else:
                    dyn_dim = dims['z'] * (dims['u'] + 1)
            else:
                dyn_dim = dims['z'] + dims['u']
        else:
            dyn_dim = dims['z']
        dims['s'] = dyn_dim

        # Autoencoder
        assert 'auto' in enc_type, "LFM model needs state-only encoder."

        # Features in the dynamics
        fzu_type = fzu_selector(fzu_type, dims['u'], const_term)

        # Processor in the dynamics
        processor_net = FlexLinear(dims['s'], dims['z'], bias=False, dtype=dtype, device=device)

        # Prediction options
        input_order = model_config.get('input_order', 'cubic')
        prd_type = model_config.get('predictor_type', 'ode')

        return dims, (enc_type, fzu_type, dec_type, prd_type), processor_net, input_order

    def linear_features(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear features, f, and outputs, dz, for the model.

        dz = Af(z)

        dz is the output of the dynamics, z_dot for cont-time, z_next for disc-time.
        """
        z = self.encoder(w)
        return self.dynamics.features(z, w), z

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
    def build_core(cls, types, model_config, data_meta, dtype, device, ifgnn=False):
        enc_type, fzu_type, dec_type = types

        # Options
        const_term = model_config.get('const_term', True)

        # Dimensions
        dims = get_dims(model_config, data_meta)

        # Autoencoder
        assert 'auto' in enc_type, "KM model needs state-only encoder."

        # Features in the dynamics
        fzu_type = fzu_selector(fzu_type, dims['u'], const_term)

        # Processor in the dynamics
        opts = {
            'type'       : model_config.get('type', 'share'),
            'kernel'     : model_config.get('kernel', None),
            'ridge_init' : model_config.get('ridge_init', 1e-10),
            'jitter'     : model_config.get('jitter', 1e-12),
            'dtype'      : dtype,
            'device'     : device
        }
        processor_net = make_krr(**opts)

        # Prediction options
        input_order = model_config.get('input_order', 'cubic')
        prd_type = model_config.get('predictor_type', 'ode')

        return dims, (enc_type, fzu_type, dec_type, prd_type), processor_net, input_order

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
