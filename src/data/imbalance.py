from typing import List, Sequence

import numpy as np


def make_long_tailed_indices(
    targets: Sequence[int],
    num_classes: int,
    imbalance_factor: float = 0.1,
    seed: int = 42,
) -> List[int]:
    """Return indices (relative to `targets`) forming an exponentially decaying,
    long-tailed class distribution. `imbalance_factor` is the ratio between the
    smallest and largest resulting class size (e.g. 0.1 == 10x imbalance)."""
    targets_arr = np.asarray(targets)
    rng = np.random.default_rng(seed)
    class_counts = np.bincount(targets_arr, minlength=num_classes)
    max_count = int(class_counts.max())

    kept_indices: List[int] = []
    for cls in range(num_classes):
        cls_indices = np.where(targets_arr == cls)[0]
        ratio = imbalance_factor ** (cls / max(num_classes - 1, 1))
        n_keep = min(len(cls_indices), max(1, round(max_count * ratio)))
        chosen = rng.choice(cls_indices, size=n_keep, replace=False)
        kept_indices.extend(chosen.tolist())

    rng.shuffle(kept_indices)
    return kept_indices
