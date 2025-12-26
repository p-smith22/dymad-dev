import torch
import torch.nn as nn


class LinearRNN(nn.Module):
    """A simple linear recurrent neural network module."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Linear transformations
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Initial hidden state of shape (batch_size, hidden_size)

        Returns:
            output: Output tensor of shape (batch_size, seq_len, output_size)
            hidden: Final hidden state of shape (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []

        # Process sequence
        for t in range(seq_len):
            hidden = self.i2h(x[:, t, :]) + self.h2h(hidden)
            output = self.h2o(hidden)
            outputs.append(output)

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)

        return outputs, hidden


class SequentialMLP(nn.Module):
    """Module that applies an mlp to each timestep in a stacked sequence."""

    def __init__(self, mlp, seq_len):
        """
        Args:
            mlp: An mlp module with forward(x) method
            seq_len: Length of the sequence
        """
        super().__init__()
        self.mlp = mlp
        self.seq_len = seq_len

    def forward(self, x, **kwargs):
        """
        Args:
            x: Stacked input tensor of shape (batch_size, seq_len * input_size)

        Returns:
            output: Stacked output tensor of shape (batch_size, seq_len * output_size)
        """
        x_shape = x.shape[:-1]
        stacked_input_size = x.shape[-1]
        input_size = stacked_input_size // self.seq_len

        # Apply mlp to each step
        x_flat = x.view(-1, input_size)
        outputs_flat = self.mlp(x_flat, **kwargs)

        # Reshape directly to stacked format (batch_size, seq_len * output_size)
        output_size = outputs_flat.shape[-1]
        outputs_stacked = outputs_flat.view(*x_shape, self.seq_len * output_size)

        return outputs_stacked


class SequentialCT(nn.Module):
    """Interface module that handles stacked input sequences for LinearRNN."""

    def __init__(self, rnn, seq_len):
        super().__init__()
        self.rnn = rnn
        self.seq_len = seq_len

    def forward(self, x, hidden=None):
        """
        Args:
            x: Stacked input tensor of shape (batch_size * seq_len, input_size)
            hidden: Initial hidden state of shape (batch_size, hidden_size)

        Returns:
            output: Stacked output tensor of shape (batch_size * seq_len, output_size)
            hidden: Final hidden state of shape (batch_size, hidden_size)
        """
        # Infer dimensions from stacked input
        batch_size = x.size(0)
        stacked_input_size = x.size(1)
        input_size = stacked_input_size // self.seq_len

        # Reshape from stacked to sequence format
        x_seq = x.view(batch_size, self.seq_len, input_size)

        # Forward through RNN
        outputs, _ = self.rnn(x_seq, hidden)

        # Reshape back to stacked format
        output_size = outputs.size(2)
        outputs_stacked = outputs.view(batch_size * self.seq_len, output_size)

        return outputs_stacked

