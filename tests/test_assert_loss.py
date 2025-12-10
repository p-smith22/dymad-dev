import numpy as np
import pytest
import torch

from dymad.losses import vpt_loss, VPTLoss, wmse_loss, WMSELoss

def _wmse_loss_ref(predictions, targets, alpha):
    n_steps = predictions.shape[-2]
    weights = np.exp(-alpha * np.arange(n_steps))
    weights = weights / weights.sum()
    weights = weights.reshape(1, -1, 1)
    loss = weights * (predictions - targets) ** 2
    return loss.mean()

eps = 1e-15
batch_size = 3
n_steps = 5
n_features = 2
predictions = torch.randn(batch_size, n_steps, n_features, dtype=torch.float64)
targets = torch.randn(batch_size, n_steps, n_features, dtype=torch.float64)

@pytest.mark.parametrize("alpha", [-0.5, 0, 0.5])
def test_wmse_loss(alpha):
    loss = WMSELoss(alpha=alpha)
    loss_cls = loss(predictions, targets)
    loss_fun = wmse_loss(predictions, targets, alpha=alpha)
    loss_ref = _wmse_loss_ref(predictions.numpy(), targets.numpy(), alpha=alpha)

    assert np.abs(loss_cls.item() - loss_ref) < eps, f"WMSELoss alpha={alpha}"
    assert np.abs(loss_fun.item() - loss_cls.item()) == 0, f"wmse_loss alpha={alpha}"

def test_vpt_loss():
    STP = 4
    GMM = 0.08
    ERR = 0.05

    std = torch.std(targets, dim=-2, keepdim=True)
    std = torch.clamp(std, min=1e-8)
    preds = targets.clone() + ERR * std
    preds[:, STP:, :] += ERR * std

    loss = VPTLoss(gamma=GMM, scl=10.0)
    loss_cls = loss(preds, targets)                  # Softmax version
    loss_fun = vpt_loss(preds, targets, gamma=GMM)   # Exact version

    # Reference
    tmp = np.ones(n_steps) * ERR
    tmp[STP:] += ERR
    w = np.exp(10*(tmp-GMM))
    loss_ref = 1 / (np.sum(w) + 1e-8)

    assert np.abs(loss_cls.item() - loss_ref.item()) < eps, "VPTLoss softmax"
    assert loss_fun.item() == STP, "vpt_loss exact"
