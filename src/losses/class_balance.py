from typing import Sequence

import torch


def compute_class_balanced_alpha(class_counts: Sequence[int], beta: float = 0.999) -> torch.Tensor:
    """Class-balanced weights from the effective number of samples (Cui et al., 2019).

    Currently unused. Usage (not wired into training):
        targets = [datamodule.train_set.dataset.targets[i] for i in datamodule.train_set.indices]
        class_counts = np.bincount(targets, minlength=num_classes)
        weights = compute_class_balanced_alpha(class_counts.tolist())  # shape: (num_classes,)
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    effective_num = 1.0 - torch.pow(torch.tensor(beta), counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(counts)
    return weights
