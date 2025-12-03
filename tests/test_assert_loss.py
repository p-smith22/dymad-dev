import numpy as np
import pytest
import torch

from dymad.losses import WMSELoss

def _wmse_loss_ref(predictions, targets, alpha):
    n_steps = predictions.shape[-2]
    weights = np.exp(-alpha * np.arange(n_steps))
    weights = weights / weights.sum()
    weights = weights.reshape(1, -1, 1)
    loss = weights * (predictions - targets) ** 2
    return loss.mean()

n_steps = 5
batch_size = 3
n_features = 2
predictions = torch.randn(batch_size, n_steps, n_features, dtype=torch.float64)
targets = torch.randn(batch_size, n_steps, n_features, dtype=torch.float64)

@pytest.mark.parametrize("alpha", [-0.5, 0, 0.5])
def test_wmse_loss(alpha):
    eps = 1e-14

    loss_fn = WMSELoss(alpha=alpha)
    loss = loss_fn(predictions, targets)
    loss_ref = _wmse_loss_ref(predictions.numpy(), targets.numpy(), alpha=alpha)

    assert np.abs(loss.item() - loss_ref) < eps, f"WMSELoss alpha={alpha}"
