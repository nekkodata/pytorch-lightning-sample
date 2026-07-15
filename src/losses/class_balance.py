from typing import Sequence

import torch


def compute_class_balanced_alpha(class_counts: Sequence[int], beta: float = 0.999) -> torch.Tensor:
    """Class-balanced weights from the effective number of samples (Cui et al., 2019)."""
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    effective_num = 1.0 - torch.pow(torch.tensor(beta), counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(counts)
    return weights
