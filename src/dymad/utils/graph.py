import logging
import numpy as np
import torch

try:
    from torch_geometric.utils import dense_to_sparse
except:
    logging.warning("torch_geometric is not installed. GNN-related functionality will be unavailable.")
    dense_to_sparse = None

def adj_to_edge(adj):
    if dense_to_sparse is None:
        raise ImportError("torch_geometric is not installed. GNN-related functionality will be unavailable.")
    if isinstance(adj, np.ndarray):
        adj = torch.Tensor(adj)
    edge_index, edge_weights = dense_to_sparse(adj)
    return edge_index.numpy(), edge_weights.numpy()
