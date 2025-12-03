import torch
import torch.nn as nn

class WMSELoss(nn.Module):
    r"""Weighted Mean Squared Error Loss

    At step i, the loss is defined as: w_i(x_i-\hat{x}_i)^2,
    where w_i is the weight for step i.

    Currently, an exponential weighting is used; let v_i = exp(-alpha*i),
    then w_i = v_i / sum_j v_j.

    When alpha=0, this reduces to the standard MSE loss.
    Note that alpha can be both positive and negative, favoring early or late steps
    """
    def __init__(self, alpha=0) -> None:
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha))

    def __str__(self):
        return super().__str__() + f"(alpha={self.alpha.item()})"

    def forward(self, predictions, targets):
        _p = torch.as_tensor(predictions)
        _t = torch.as_tensor(targets)
        n_steps = predictions.shape[-2]
        if self.alpha == 0:
            loss = (_p - _t) ** 2
            return torch.mean(loss) / n_steps
        weights = torch.exp(-self.alpha * torch.arange(n_steps, device=_p.device, dtype=_p.dtype))
        weights = weights / weights.sum()
        weights = weights.view(*[1]*(_p.dim()-2), -1, 1)
        loss = weights * (_p - _t) ** 2
        return torch.mean(loss)


LOSS_MAP = {
    "mse": torch.nn.MSELoss,
    "mae": torch.nn.L1Loss,
    "wmse": WMSELoss,
}
