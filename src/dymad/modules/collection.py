import torch.nn as nn
from typing import Tuple

from dymad.modules.mlp import MLP, ResBlockMLP, IdenCatMLP
from dymad.modules.gnn import GNN, ResBlockGNN, IdenCatGNN
from dymad.modules.misc import TakeFirst, TakeFirstGraph

def make_autoencoder(
        type: str,
        input_dim: int, latent_dim: int, hidden_dim: int, enc_depth: int, dec_depth: int,
        output_dim: int = None, **kwargs) -> Tuple[nn.Module, nn.Module]:
    """
    Factory function to create preset autoencoder models. Including:

    - [mlp_smp] Simple version: MLP-in MLP-out
    - [mlp_res] Simple version but with ResBlockMLP
    - [mlp_cat] Concatenation as encoder [x MLP(x)], then TakeFirst as decoder
    - The graph version of the above: gnn_smp, gnn_res, gnn_cat

    Args:
        type (str): Type of autoencoder to create.
            One of {'mlp_smp', 'mlp_res', 'mlp_cat', 'gnn_smp', 'gnn_res', 'gnn_cat'}.
        input_dim (int): Dimension of the input features.
        latent_dim (int): Width of the latent layers (not the encoded space).
        hidden_dim (int): Dimension of the encoded space.
        enc_depth (int): Number of layers in the encoder.
        dec_depth (int): Number of layers in the decoder.
        output_dim (int, optional): Dimension of the output features, defaults to `input_dim`.
        **kwargs: Additional keyword arguments passed to the MLP or GNN constructors.
    """
    # Prepare the arguments
    if output_dim is None:
        output_dim = input_dim

    encoder_args = dict(
        input_dim=input_dim,
        latent_dim=latent_dim,
        output_dim=hidden_dim,
        n_layers=enc_depth,
    )
    encoder_args.update(kwargs)
    decoder_args = dict(
        input_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        n_layers=dec_depth,
    )
    decoder_args.update(kwargs)

    # Generate the encoder and decoder based on the type
    _type = type.lower()
    encoder, decoder = None, None
    if _type[:3] == "mlp":
        if _type == "mlp_smp":
            encoder = MLP(**encoder_args)
            decoder = MLP(**decoder_args)

        elif _type == "mlp_res":
            encoder = ResBlockMLP(**encoder_args)
            decoder = ResBlockMLP(**decoder_args)

        elif _type == "mlp_cat":
            encoder = IdenCatMLP(**encoder_args)
            decoder = TakeFirst(output_dim)

    elif _type[:3] == "gnn":
        if _type == "gnn_smp":
            encoder = GNN(**encoder_args)
            decoder = GNN(**decoder_args)

        elif _type == "gnn_res":
            encoder = ResBlockGNN(**encoder_args)
            decoder = ResBlockGNN(**decoder_args)

        elif _type == "gnn_cat":
            encoder = IdenCatGNN(**encoder_args)
            decoder = TakeFirstGraph(output_dim)

    if encoder is None or decoder is None:
        raise ValueError(f"Unknown autoencoder type '{type}'.")

    return encoder, decoder
