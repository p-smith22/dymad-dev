import numpy as np
import torch
from typing import Dict, Union, Tuple

from dymad.data import DynData
from dymad.models import ModelBase
from dymad.modules import make_autoencoder, make_krr
from dymad.utils import predict_continuous, predict_discrete

class KM(ModelBase):
    """
    Dynamics based on kernel machine, continuous-time version.

    The architecture is very close to LDM, mainly that MLP is replaced by KRR.
    The linear structure of KRR allows more efficient linear updates during training.

    - z = encoder(x, u)
    - z_dot = A phi(z)
    - x_hat = decoder(z)
    """
    GRAPH = False
    CONT  = True

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super(KM, self).__init__()
        self.n_total_state_features = data_meta.get('n_total_state_features')
        self.n_total_control_features = data_meta.get('n_total_control_features')
        self.n_total_features = data_meta.get('n_total_features')
        self.latent_dimension = model_config.get('latent_dimension', 64)

        # Method for input handling
        self.input_order = model_config.get('input_order', 'cubic')

        # Get layer depths from config
        enc_depth = model_config.get('encoder_layers', 2)
        dec_depth = model_config.get('decoder_layers', 2)

        # Determine dimensions
        enc_out_dim = self.latent_dimension if enc_depth > 0 else self.n_total_features
        dec_inp_dim = self.latent_dimension if dec_depth > 0 else self.n_total_features

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
        aec_type = model_config.get('autoencoder_type', 'smp')

        # Build encoder/decoder networks
        self.encoder_net, self.decoder_net = make_autoencoder(
            type="mlp_"+aec_type,
            input_dim=self.n_total_features,
            latent_dim=self.latent_dimension,
            hidden_dim=enc_out_dim,
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            output_dim=self.n_total_state_features,
            **opts
        )

        # Build kernel-based dynamics
        opts = {
            'type'       : model_config.get('type', 'share'),
            'kernel'     : model_config.get('kernel', None),
            'ridge_init' : model_config.get('ridge_init', 1e-10),
            'dtype'      : dtype,
            'device'     : device
        }
        self.dynamics_net = make_krr(**opts)

        if self.n_total_control_features == 0:
            self.encoder = self._encoder_auto
        else:
            self.encoder = self._encoder_ctrl

    def diagnostic_info(self) -> str:
        model_info = super(KM, self).diagnostic_info()
        model_info += f"Encoder: {self.encoder_net.diagnostic_info()}\n"
        model_info += f"Dynamics: {self.dynamics_net}\n"
        model_info += f"Decoder: {self.decoder_net.diagnostic_info()}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def _encoder_ctrl(self, w: DynData) -> torch.Tensor:
        """
        Map features to latent space for systems with inputs.
        """
        return self.encoder_net(torch.cat([w.x, w.u], dim=-1))

    def _encoder_auto(self, w: DynData) -> torch.Tensor:
        """
        Map features to latent space for autonomous systems.
        """
        return self.encoder_net(w.x)

    def decoder(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Map from latent space back to state space.
        """
        return self.decoder_net(z)

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Compute latent dynamics (derivative).
        """
        return self.dynamics_net(z)

    def forward(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        """
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        x_hat = self.decoder(z, w)
        return z, z_dot, x_hat

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor],
                method: str = 'dopri5', **kwargs) -> torch.Tensor:
        """
        Predict trajectory using continuous-time integration.
        """
        return predict_continuous(self, x0, ts, us=w.u, method=method, order=self.input_order, **kwargs)

    def linear_features(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear features, f, and outputs, dz, for (D)KM model.

        dz = Af(z)

        For KM, f contains the kernel features, which will be handled internally by KRR;
        but the call to KRR will be handled by LSUpdater.
        dz is the output of KM dynamics, z_dot for cont-time, z_next for disc-time.
        """
        z = self.encoder(w)
        return z, z

    def linear_eval(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear evaluation, dz, and states, z, for (D)KM model.

        dz = Af(z)

        For KM, dz is the output of kernel dynamics.
        z is the encoded state, which will be used to compute the expected output.
        """
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        return z_dot, z

    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit the kernel dynamics using input-output pairs.
        """
        self.dynamics_net.set_train_data(inp, out)
        residual = self.dynamics_net.fit()
        return self.dynamics_net._alphas, residual

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

class DKM(KM):
    """
    Dynamics based on kernel machine, discrete-time version.

    In this case, the forward pass effectively does the following:

    ```
    z_n = self.encoder(w_n)
    z_{n+1} = self.dynamics(z_n, w_n)
    x_hat_n = self.decoder(z_n, w_n)
    ```
    """
    GRAPH = False
    CONT  = False

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super(DKM, self).__init__(model_config, data_meta, dtype=dtype, device=device)

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        return predict_discrete(self, x0, ts, us=w.u)

class DKMSK(KM):
    """
    Dynamics based on kernel machine with skip connections in dynamics.
    """
    GRAPH = False
    CONT  = False

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super(DKMSK, self).__init__(model_config, data_meta, dtype=dtype, device=device)

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Compute latent dynamics (derivative).
        """
        return z + self.dynamics_net(z)

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        return predict_discrete(self, x0, ts, us=w.u)

    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit the kernel dynamics using input-output pairs.
        """
        self.dynamics_net.set_train_data(inp, out-inp)
        residual = self.dynamics_net.fit()
        return self.dynamics_net._alphas, residual
