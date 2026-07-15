from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[float] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha if alpha is not None else -1.0
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).to(logits.dtype)
        loss = sigmoid_focal_loss(logits, targets_one_hot, alpha=self.alpha, gamma=self.gamma, reduction="none")
        return loss.sum(dim=1).mean()
