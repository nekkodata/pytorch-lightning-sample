from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss


def compute_class_balanced_alpha(class_counts: Sequence[int], beta: float = 0.999) -> torch.Tensor:
    """Class-balanced weights from the effective number of samples (Cui et al., 2019)."""
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    effective_num = 1.0 - torch.pow(torch.tensor(beta), counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(counts)
    return weights


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[float] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha if alpha is not None else -1.0
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).to(logits.dtype)
        loss = sigmoid_focal_loss(logits, targets_one_hot, alpha=self.alpha, gamma=self.gamma, reduction="none")
        return loss.sum(dim=1).mean()
