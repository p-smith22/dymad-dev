import numpy as np
import torch
from typing import Dict, Union, Tuple

from dymad.data import DynData, DynGeoData
from dymad.models import ModelBase
from dymad.modules import make_autoencoder

class ModelTempUCat(ModelBase):
    """
    Base class for state-encoding with input concatenation.
    Handles MLP-based encoder/decoder construction and common methods.
    Same architecture for both cont-time and disc-time.

    - z = encoder(x)
    - z_dot/z_next = Custom_Dynamics(z, u)
    - x_hat = decoder(z)
    """
    GRAPH = False
    CONT = None

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__()
        self.n_total_state_features = data_meta.get('n_total_state_features')
        self.n_total_control_features = data_meta.get('n_total_control_features')
        self.latent_dimension = model_config.get('latent_dimension', 64)
        self.dtype = dtype
        self.device = device

        # Method for input handling
        self.input_order = model_config.get('input_order', 'cubic')

        # Input concatenation
        if self.n_total_control_features == 0:
            self._zu_cat = self._zu_cat_auto
        else:
            self._zu_cat = self._zu_cat_ctrl

        # Cache
        self.encoder_net  = None
        self.dynamics_net = None
        self.decoder_net  = None

    def _build_autoencoder(self, hidden_dim, model_config, dtype, device):
        enc_depth = model_config.get('encoder_layers', 2)
        dec_depth = model_config.get('decoder_layers', 2)

        if self.n_total_state_features != hidden_dim:
            if enc_depth == 0 or dec_depth == 0:
                raise ValueError(f"Encoder depth {enc_depth}, decoder depth {dec_depth}: "
                                 f"but n_total_state_features ({self.n_total_state_features}) "
                                 f"must match hidden_dim ({hidden_dim})")

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

        self.encoder_net, self.decoder_net = make_autoencoder(
            type="mlp_"+aec_type,
            input_dim=self.n_total_state_features,
            latent_dim=self.latent_dimension,
            hidden_dim=hidden_dim,
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            output_dim=self.n_total_state_features,
            **opts
        )

    def diagnostic_info(self) -> str:
        model_info = super().diagnostic_info()
        model_info += f"Encoder: {self.encoder_net.diagnostic_info()}\n"
        model_info += f"Dynamics: {self.dynamics_net}\n"
        model_info += f"Decoder: {self.decoder_net.diagnostic_info()}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def encoder(self, w: DynData) -> torch.Tensor:
        """Encode combined features to embedded space."""
        return self.encoder_net(w.x)

    def decoder(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """Decode from embedded space back to state space."""
        return self.decoder_net(z)

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """Compute dynamics in embedded space."""
        return self.dynamics_net(self._zu_cat(z, w))

    def _zu_cat_ctrl(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """Concatenate state and control inputs."""
        raise NotImplementedError("Implement in derived class.")

    def _zu_cat_auto(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """Concatenate state and control inputs."""
        return z

    def forward(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for the model.

        Args:
            w: DynData obejct, containing state (x) and control (u) tensors.

        Returns:
            Tuple of (latent, latent_derivative, reconstruction)
        """
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        x_hat = self.decoder(z, w)
        return z, z_dot, x_hat

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor],
                method: str = 'dopri5', **kwargs) -> torch.Tensor:
        """Predict trajectory using continuous-time integration.

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

    def linear_features(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear features, f, and outputs, dz, for the model.

        dz = Af(z)

        dz is the output of the dynamics, z_dot for cont-time, z_next for disc-time.
        """
        z = self.encoder(w)
        return self._zu_cat(z, w), z

    def linear_eval(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear evaluation, dz, and states, z, for the model.

        dz = Af(z)

        z is the encoded state, which will be used to compute the expected output.
        """
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        return z_dot, z


class ModelTempUCatGraph(ModelBase):
    """Graph version of ModelTempUCat.

    The MLP autoencoder is replaced by GNN-based one.

    The dynamics is defined per node.
    """
    GRAPH = True
    CONT  = None

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super().__init__()
        self.n_total_state_features = data_meta.get('n_total_state_features')
        self.n_total_control_features = data_meta.get('n_total_control_features')
        self.latent_dimension = model_config.get('latent_dimension', 64)
        self.dtype = dtype
        self.device = device

        # Method for input handling
        self.input_order = model_config.get('input_order', 'cubic')

        # Input concatenation
        if self.n_total_control_features == 0:
            self._zu_cat = self._zu_cat_auto
        else:
            self._zu_cat = self._zu_cat_ctrl

        # Cache
        self.encoder_net  = None
        self.dynamics_net = None
        self.decoder_net  = None

    def _build_autoencoder(self, hidden_dim, model_config, dtype, device):
        # Get layer depths from config
        enc_depth = model_config.get('encoder_layers', 2)
        dec_depth = model_config.get('decoder_layers', 2)

        # Determine other options for GNN layers
        opts = {
            'gcl'            : model_config.get('gcl', 'sage'),
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
            type="gnn_"+aec_type,
            input_dim=self.n_total_state_features,
            latent_dim=self.latent_dimension,
            hidden_dim=hidden_dim,
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            output_dim=self.n_total_state_features,
            **opts
        )

    def diagnostic_info(self) -> str:
        model_info = super().diagnostic_info()
        model_info += f"Encoder: {self.encoder_net.diagnostic_info()}\n"
        model_info += f"Dynamics: {self.dynamics_net}\n"
        model_info += f"Decoder: {self.decoder_net.diagnostic_info()}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def encoder(self, w: DynGeoData) -> torch.Tensor:
        # The GNN implementation outputs flattened features
        # Here internal dynamics are node-wise, so we need to reshape
        # the features to node*features_per_node again
        return w.g(self.encoder_net(w.xg, w.edge_index))

    def decoder(self, z: torch.Tensor, w: DynGeoData) -> torch.Tensor:
        # Since the decoder outputs to the original space,
        # which is assumed to be flattened, we can use the GNN decoder directly
        # Note: the input, though, is still node-wise
        return self.decoder_net(z, w.edge_index)

    def dynamics(self, z: torch.Tensor, w: DynGeoData) -> torch.Tensor:
        """Compute dynamics in Koopman space using bilinear form."""
        return self.dynamics_net(self._zu_cat(z, w))

    def _zu_cat_ctrl(self, z: torch.Tensor, w: DynGeoData) -> torch.Tensor:
        """Concatenate state and control inputs."""
        raise NotImplementedError("Implement in derived class.")

    def _zu_cat_auto(self, z: torch.Tensor, w: DynGeoData) -> torch.Tensor:
        """Concatenate state and control inputs."""
        return z

    def forward(self, w: DynGeoData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        x_hat = self.decoder(z, w)
        return z, z_dot, x_hat

    def predict(self, x0: torch.Tensor, w: DynGeoData, ts: Union[np.ndarray, torch.Tensor], method: str = 'dopri5', **kwargs) -> torch.Tensor:
        raise NotImplementedError("Implement in derived class.")

    def linear_features(self, w: DynGeoData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear features, f, and outputs, dz, for the model.

        Main difference with ModelTempUCat: the middle two dimensions are permuted, so that
        the time dimension is the second last dimension, this is needed in
        linear trainer to match the expected shape.
        """
        z = self.encoder(w)
        f = self._zu_cat(z, w)
        return f.permute(0, 2, 1, 3), z.permute(0, 2, 1, 3)

    def linear_eval(self, w: DynGeoData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear evaluation, dz, and states, z, for the model.

        Same idea as in linear_features about the permutation.
        """
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        return z_dot.permute(0, 2, 1, 3), z.permute(0, 2, 1, 3)
