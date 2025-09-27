import copy
import torch.nn as nn
from typing import Dict, List, Tuple, Union

from dymad.modules.gnn import GNN, ResBlockGNN, IdenCatGNN
from dymad.modules.kernel import KernelScDM, KernelScExp, KernelScRBF, KernelOpSeparable, KernelOpTangent
from dymad.modules.krr import KRRMultiOutputShared, KRRMultiOutputIndep, KRROperatorValued, KRRTangent
from dymad.modules.misc import TakeFirst, TakeFirstGraph
from dymad.modules.mlp import MLP, ResBlockMLP, IdenCatMLP

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

def _make_scalar_kernel(type: str, input_dim: int, dtype=None, **kwargs) -> nn.Module:
    if type == "rbf":
        return KernelScRBF(in_dim=input_dim, dtype=dtype, **kwargs)
    elif type == "dm":
        return KernelScDM(in_dim=input_dim, dtype=dtype, **kwargs)
    elif type == "exp":
        return KernelScExp(in_dim=input_dim, dtype=dtype, **kwargs)
    else:
        raise ValueError(f"Unknown kernel type '{type}'.")

def make_kernel(type: str, input_dim: int, output_dim: int = None, kopts: List=None, dtype=None, **kwargs) -> nn.Module:
    """
    Factory function to create preset kernels. Including:

    - [sc_rbf] Scalar: Radial basis function kernel
    - [sc_dm]  Scalar: Diffusion Map kernel
    - [op_sep] Operator-valued: Separable kernel with multiple scalar kernels

    Args:
        type (str): Type of kernel to create.
            One of {'sc_rbf', 'sc_dm'}.
        input_dim (int): Dimension of the input features.
        output_dim (int, optional): Dimension of the output features.
        kopts (List, optional): List of scalar kernel options (for operator-valued kernels).
        dtype: Data type of the kernel parameters.
        **kwargs: Additional keyword arguments passed to the kernel constructors.
    """
    _type = type.lower().split('_')
    if len(_type) < 2:
        raise ValueError(f"Unknown kernel type '{type}'.")

    if _type[0] == "sc":
        # Scalar-valued kernels
        if output_dim is not None and output_dim != 1:
            raise ValueError(f"Scalar-valued kernel cannot have output_dim>1, got {output_dim}.")
        return _make_scalar_kernel(_type[1], input_dim, dtype=dtype, **kwargs)

    elif _type[0] == "op":
        # Operator-valued kernels
        if output_dim is None or output_dim < 2:
            raise ValueError(f"Operator-valued kernel must have output_dim>=2, got {output_dim}.")

        if _type[1] == "sep":
            if kopts is None or len(kopts) == 0:
                raise ValueError(f"Operator-valued kernel of type 'sep' must have at least one scalar kernel option, got {kopts}.")

            kernels = []
            for _k in kopts:
                _opt = copy.deepcopy(_k)
                _k_type = _opt.pop("type").split('_')[-1]
                _k_input_dim = _opt.pop("input_dim")
                kernels.append(_make_scalar_kernel(_k_type, _k_input_dim, dtype=dtype, **_opt))
            return KernelOpSeparable(kernels, out_dim=output_dim, dtype=dtype, **kwargs)

        if _type[1] == "tan":
            if not isinstance(kopts, dict):
                raise ValueError(f"Operator-valued kernel of type 'tan' must have one and only one scalar kernel, got {kopts}.")

            _opt = copy.deepcopy(kopts)
            _k_type = _opt.pop("type").split('_')[-1]
            _k_input_dim = _opt.pop("input_dim")
            kernel = _make_scalar_kernel(_k_type, _k_input_dim, dtype=dtype, **_opt)
            return KernelOpTangent(kernel, out_dim=output_dim, dtype=dtype, **kwargs)

        else:
            raise ValueError(f"Unknown operator-valued kernel type '{_type[1]}'.")
    else:
        raise ValueError(f"Unknown kernel type '{type}'.")

def make_krr(
        type: str, kernel: Union[Dict, List[Dict]],
        ridge_init=0, jitter=1e-10, dtype=None, device=None) -> nn.Module:
    """
    Factory function to create preset Kernel Ridge Regression (KRR) models. Including:

    - [share] Multi-output KRR with shared scalar kernel
    - [indep] Multi-output KRR with independent scalar kernels
    - [opval] Multi-output KRR with operator-valued kernel
    - [tangent] KRR for vector fields on manifolds

    Args:
        type (str): Type of KRR model to create.
            One of {'krr_shared', 'krr_indep', 'krr_opval', 'krr_tangent'}.
        kernel (Union[Dict, List[Dict]]): Kernel configuration(s).
        ridge_init (float, optional): Initial value for the ridge regularization parameter.
        jitter (float, optional): Jitter added to the diagonal for numerical stability.
    """
    if isinstance(kernel, dict):
        _ker = [kernel]
    elif isinstance(kernel, list):
        _ker = kernel
    else:
        raise ValueError(f"Kernel must be a dictionary or a list of dictionaries, got {type(kernel)}.")

    k_module = []
    for _k in _ker:
        k_type   = _k["type"]
        k_input_dim  = _k["input_dim"]
        k_output_dim = _k.get("output_dim", None)
        k_kopts  = _k.get("kopts", None)
        k_kwargs = {k: v for k, v in _k.items() if k not in {"type", "input_dim", "output_dim", "kopts"}}
        k_module.append(make_kernel(k_type, input_dim=k_input_dim, output_dim=k_output_dim, kopts=k_kopts, dtype=dtype, **k_kwargs))
    if isinstance(kernel, dict):
        # Consistency with input type
        k_module = k_module[0]

    _type = type.lower()
    if _type == "share":
        return KRRMultiOutputShared(kernel=k_module, ridge_init=ridge_init, jitter=jitter, device=device)
    elif _type == "indep":
        return KRRMultiOutputIndep(kernel=k_module, ridge_init=ridge_init, jitter=jitter, device=device)
    elif _type == "opval":
        return KRROperatorValued(kernel=k_module, ridge_init=ridge_init, jitter=jitter, device=device)
    elif _type == "tangent":
        return KRRTangent(kernel=k_module, ridge_init=ridge_init, jitter=jitter, device=device)
    else:
        raise ValueError(f"Unknown KRR type '{type}'.")
