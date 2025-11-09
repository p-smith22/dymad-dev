import numpy as np
import torch
from typing import Dict, Union, Tuple

from dymad.io import DynData
from dymad.models.model_base import ModelBase
from dymad.modules import make_autoencoder

class TemplateUEnc(ModelBase):
    """
    Template class for joint encoding of states and inputs.
    Handles MLP-based encoder/decoder construction and common methods.
    Same architecture for both cont-time and disc-time.

    - z = encoder(x, u)
    - z_dot/z_next = Custom_Dynamics(z)
    - x_hat = decoder(z)
    """
    GRAPH = False
    CONT = None

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__()
        self.n_total_state_features = data_meta.get('n_total_state_features')
        self.n_total_control_features = data_meta.get('n_total_control_features')
        self.n_total_features = data_meta.get('n_total_features')
        self.latent_dimension = model_config.get('latent_dimension', 64)
        self.dtype = dtype
        self.device = device

        # Method for input handling
        self.input_order = model_config.get('input_order', 'cubic')

        # Input concatenation
        if self.n_total_control_features == 0:
            self.encoder = self._encoder_auto
        else:
            self.encoder = self._encoder_ctrl

        # Cache
        self.encoder_net  = None
        self.dynamics_net = None
        self.decoder_net  = None

    def _build_autoencoder(self, model_config, dtype, device):
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

        return enc_out_dim, dec_inp_dim

    def diagnostic_info(self) -> str:
        model_info = super().diagnostic_info()
        model_info += f"Encoder: {self.encoder_net.diagnostic_info()}\n"
        model_info += f"Dynamics: {self.dynamics_net}\n"
        model_info += f"Decoder: {self.decoder_net.diagnostic_info()}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def _encoder_ctrl(self, w: DynData) -> torch.Tensor:
        """
        Map features to latent space for systems with inputs.

        Args:
            w (DynData): Raw features

        Returns:
            torch.Tensor: Latent representation
        """
        return self.encoder_net(torch.cat([w.x, w.u], dim=-1))

    def _encoder_auto(self, w: DynData) -> torch.Tensor:
        """
        Map features to latent space for autonomous systems.

        Args:
            w (DynData): Raw features

        Returns:
            torch.Tensor: Latent representation
        """
        return self.encoder_net(w.x)

    def decoder(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Map from latent space back to state space.

        Args:
            z (torch.Tensor): Latent state

        Returns:
            torch.Tensor: Reconstructed state
        """
        return self.decoder_net(z)

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Compute latent dynamics, derivative or next state.

        Args:
            z (torch.Tensor): Latent state

        Returns:
            torch.Tensor: Latent state derivative
        """
        return self.dynamics_net(z)

    def forward(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            w (DynData): Input data containing state and control tensors

        Returns:
            Tuple of (latent, latent_derivative/latent_next, reconstruction)
        """
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        x_hat = self.decoder(z, w)
        return z, z_dot, x_hat

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor],
                method: str = 'dopri5', **kwargs) -> torch.Tensor:
        """Predict trajectory using cont-time or disc-time integration.

        Args:
            x0: Initial state tensor(s):

                - Single: (n_state_features,)

            us: Control inputs:

                - Single: (time_steps, n_control_features)

            ts: Time points for prediction
            method: ODE solver method (default: 'dopri5')

        Returns:
            Predicted trajectory tensor(s):

                - Single: (time_steps, n_state_features)
                - Batch: (time_steps, batch_size, n_state_features)
        """
        raise NotImplementedError("Implement in derived class.")


class TemplateUEncGraphAE(ModelBase):
    """Graph version of TemplateUEnc.

    The MLP autoencoder is replaced by GNN-based one.

    The dynamics is defined per node.
    """
    GRAPH = True
    CONT  = None

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__()
        self.n_total_state_features = data_meta.get('n_total_state_features')
        self.n_total_control_features = data_meta.get('n_total_control_features')
        self.n_total_features = data_meta.get('n_total_features')
        self.latent_dimension = model_config.get('latent_dimension', 64)
        self.dtype = dtype
        self.device = device

        # Method for input handling
        self.input_order = model_config.get('input_order', 'cubic')

        # Input concatenation
        if self.n_total_control_features == 0:
            self.encoder = self._encoder_auto
        else:
            self.encoder = self._encoder_ctrl

        # Cache
        self.encoder_net  = None
        self.dynamics_net = None
        self.decoder_net  = None

    def _build_autoencoder(self, model_config, dtype, device):
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
            'gcl'            : model_config.get('gcl', 'sage'),
            'gcl_opts'       : model_config.get('gcl_opts', {}),
            'end_activation' : model_config.get('end_activation', True),
            'dtype'          : dtype,
            'device'         : device
        }
        aec_type = model_config.get('autoencoder_type', 'smp')

        # Build encoder/decoder networks
        self.encoder_net, self.decoder_net = make_autoencoder(
            type="gnn_"+aec_type,
            input_dim=self.n_total_features,
            latent_dim=self.latent_dimension,
            hidden_dim=enc_out_dim,
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            output_dim=self.n_total_state_features,
            **opts
        )

        return enc_out_dim, dec_inp_dim

    def diagnostic_info(self) -> str:
        model_info = super().diagnostic_info()
        model_info += f"Encoder: {self.encoder_net.diagnostic_info()}\n"
        model_info += f"Dynamics: {self.dynamics_net}\n"
        model_info += f"Decoder: {self.decoder_net.diagnostic_info()}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def _encoder_ctrl(self, w: DynData) -> torch.Tensor:
        xu_cat = torch.cat([w.xg, w.ug], dim=-1)
        return w.g(self.encoder_net(xu_cat, w.ei, w.ew, w.ea))

    def _encoder_auto(self, w: DynData) -> torch.Tensor:
        return w.g(self.encoder_net(w.xg, w.ei, w.ew, w.ea))

    def decoder(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.decoder_net(z, w.ei, w.ew, w.ea)

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.dynamics_net(z)

    def forward(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        x_hat = self.decoder(z, w)
        return z, z_dot, x_hat

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], method: str = 'dopri5', **kwargs) -> torch.Tensor:
        raise NotImplementedError("Implement in derived class.")


class TemplateUEncGraphDyn(TemplateUEnc):
    """Graph version of TemplateUEnc.

    The autoencoder is still MLP, but dynamics is expected to be GNN-based.

    The autoencoder is applied per node.

    Since n_total_state_features etc are per node, most of TemplateUEnc can be reused,
    and only the encoder-dynamics-decoder interface needs to be changed.
    """
    GRAPH = True
    CONT  = None

    def _encoder_ctrl(self, w: DynData) -> torch.Tensor:
        xu_cat = torch.cat([w.xg, w.ug], dim=-1)
        return w.G(self.encoder_net(xu_cat))  # G is needed for external data structure

    def _encoder_auto(self, w: DynData) -> torch.Tensor:
        return w.G(self.encoder_net(w.xg))    # G is needed for external data structure

    def decoder(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return w.G(self.decoder_net(w.g(z)))  # G is needed for external data structure

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        return self.dynamics_net(w.g(z), w.ei, w.ew, w.ea)   # G is effectively applied in dynamics_net
