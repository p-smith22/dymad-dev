import torch
import torch.nn as nn
from typing import Callable, Union, Optional

from dymad.modules.helpers import _resolve_activation, _resolve_init, _INIT_MAP_W, _INIT_MAP_B


class SequentialBase(nn.Module):
    """Interface module that handles stacked input sequences for LinearRNN."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        *,
        n_layers: int = 2,
        activation: Union[str, nn.Module, Callable[[], nn.Module]] = nn.ReLU,
        weight_init: Union[str, Callable[[torch.Tensor, float], None]] = nn.init.xavier_uniform_,
        bias_init: Callable[[torch.Tensor], None] = nn.init.zeros_,
        gain: Optional[float] = 1.0,
        dtype=None, device=None,
        **kwargs
    ):
        super().__init__()
        self.seq_len = seq_len
        assert input_dim % seq_len == 0, f"input_dim ({input_dim}) must be divisible by seq_len ({seq_len})."

        _act = _resolve_activation(activation, dtype, device)

        self._build_rnn(input_dim // seq_len, hidden_dim, output_dim, n_layers, _act, dtype, device, **kwargs)

        # Cache init kwargs for later use in self.apply
        self._weight_init = _resolve_init(weight_init, _INIT_MAP_W)
        self._bias_init = _resolve_init(bias_init, _INIT_MAP_B)

        # Compute gain
        act_name = _act().__class__.__name__.lower()
        _g = nn.init.calculate_gain(act_name if act_name not in ["gelu", "prelu", "identity"] else "relu")
        self._gain = gain*_g

        # Initialise weights & biases
        self.apply(self._init_linear)

    def _init_linear(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            self._weight_init(m.weight, self._gain)
            self._bias_init(m.bias)

    def _build_rnn(self, input_dim, hidden_dim, output_dim, n_layers, _act, dtype, device, **kwargs):
        """Build the internal RNN module."""
        raise NotImplementedError("_build_rnn must be implemented in subclasses.")

    def _run_rnn(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Evaluate the recurrent model."""
        raise NotImplementedError("_run_rnn must be implemented in subclasses.")

    def forward(self, x: torch.Tensor, u: torch.Tensor | None = None):
        """
        The network does concatenation internally, because x and u are
        both time-delayed and concatenated, and applying the sequential model
        requires to stack x and u of the same steps and then concatenate.

        Args:
            x: Stacked input tensor of shape (batch_size, seq_len * x_dim)
            u: (Optional) Stacked control tensor of shape (batch_size, seq_len * u_dim)

        Returns:
            output: Stacked output tensor of shape (batch_size, output_dim),
                    where the last slot is the output of the RNN at the last step, and
                    the rest are inputs shifted by one step.
        """
        # Infer dimensions from stacked input
        batch_size = x.shape[:-1]
        stacked_input_dim = x.shape[-1]
        input_dim = stacked_input_dim // self.seq_len

        # Reshape from stacked to sequence format
        x_seq = x.view(-1, self.seq_len, input_dim)

        # Forward through RNN
        z = self._run_rnn(x_seq).view(*batch_size, -1)

        return z


class StandardRNN(SequentialBase):
    """Standard RNN from pytorch."""

    def _build_rnn(self, input_dim, hidden_dim, output_dim, n_layers, _act, dtype, device, **kwargs):
        assert _act().__class__.__name__.lower() in ['tanh', 'relu'], "Only 'tanh' and 'relu' activations are supported for nn.RNN."
        assert output_dim == hidden_dim, "For StandardRNN, output_dim must equal hidden_dim."
        self.net = nn.RNN(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = n_layers,
            nonlinearity = _act().__class__.__name__.lower(),
            batch_first = True,
            device=device,
            dtype=dtype,
            **kwargs
        )
        self._run_rnn = self.net


class SimpleRNN(SequentialBase):
    """A simple recurrent neural network module
    
    One layer, unidirectional, but supports arbitrary activations and adds a linear readout.
    """

    def _build_rnn(self, input_dim, hidden_dim, output_dim, n_layers, _act, dtype, device, **kwargs):
        assert n_layers == 1, f"SimpleRNN only supports n_layers=1, got {n_layers}"

        self.hidden_dim = hidden_dim

        # Linear transformations
        self.i2h = nn.Linear(input_dim, hidden_dim, device=device, dtype=dtype)
        self.h2h = nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype)
        self.h2o = nn.Linear(hidden_dim, output_dim, device=device, dtype=dtype)
        self.activation = _act()

    def _run_rnn(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden: Initial hidden state of shape (batch_size, hidden_dim)

        Returns:
            output: Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.size()

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # Process sequence
        for t in range(seq_len):
            hidden = self.activation(self.i2h(x[:, t, :]) + self.h2h(hidden))
        output = self.activation(self.h2o(hidden))

        return output


class SeqEncoder(nn.Module):
    """Naive application of a network to each step of a sequence."""

    def __init__(self, net: nn.Module, seq_len: int):
        super().__init__()
        self.net = net           # Decoder for one step
        self.seq_len = seq_len   # Sequence length

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: Stacked input tensor of shape (..., seq_len * x_dim)

        Returns:
            output: Stacked output tensor of shape (..., seq_len * z_dim).
        """
        # Infer dimensions from stacked input
        stacked_x_dim = x_seq.shape[-1]
        x_dim = stacked_x_dim // self.seq_len

        # Forward through encoder
        x_seq_reshaped = x_seq.view(..., self.seq_len, x_dim)
        z_seq = self.net(x_seq_reshaped)  # (..., seq_len, z_dim)
        output = z_seq.view(..., self.seq_len * z_seq.shape[-1])

        return output


class ShiftDecoder(nn.Module):
    """Decoder for a sequence that decodes only one step and shifts the rest."""

    def __init__(self, net: nn.Module, seq_len: int):
        super().__init__()
        self.net = net           # Decoder for one step
        self.seq_len = seq_len   # Sequence length

    def forward(
            self,
            z: torch.Tensor,
            x_seq: torch.Tensor,
            x_prv: torch.Tensor | None = None) -> torch.Tensor:
        """
        Expect z as the fully encoded x sequence of length `seq_len`, but only decode z
        into the last step; the rest is filled using the given x sequence.

        There are two modes of operation, let N=`seq_len`

        - Using z and x_seq: this is in standard autoencoding, and z should decode to
          the last step of x_seq, while the rest are the first (N-1) steps of x_seq.
        - x_prv is given: this is in prediction mode, where x_prv is the previous full sequence,
          and z is decoded to the next step, while the rest are the last (N-1) steps of x_prv.

        Args:
            z: Latent tensor of shape (..., z_dim)
            x_seq: Stacked input tensor of shape (..., seq_len * x_dim)
            x_prv: (Optional) Stacked previous input tensor of shape (..., seq_len * x_dim)

        Returns:
            output: Stacked output tensor of shape (..., seq_len * x_dim).
        """
        # Infer dimensions from stacked input
        stacked_x_dim = x_seq.shape[-1]
        x_dim = stacked_x_dim // self.seq_len

        # Forward through decoder
        x_next = self.net(z).unsqueeze(-2)  # (..., 1, output_dim)

        # Form output
        if x_prv is None:
            # Standard autoencoding mode
            x_seq_reshaped = x_seq.view(..., self.seq_len, x_dim)
            output = torch.cat([x_seq_reshaped[..., :-1, :], x_next], dim=-2)
        else:
            # Prediction mode
            x_prv_reshaped = x_prv.view(..., self.seq_len, x_dim)
            output = torch.cat([x_prv_reshaped[..., 1:, :], x_next], dim=-2)

        return output
