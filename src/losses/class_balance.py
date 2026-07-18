from typing import Sequence

import torch


def compute_class_balanced_alpha(class_counts: Sequence[int], beta: float = 0.999) -> torch.Tensor:
    """有効サンプル数(Cui et al., 2019)に基づくクラスバランス重み。

    現状どこからも呼ばれていない未使用ユーティリティ。使用例(学習には未組み込み):
        targets = [datamodule.train_set.dataset.targets[i] for i in datamodule.train_set.indices]
        class_counts = np.bincount(targets, minlength=num_classes)
        weights = compute_class_balanced_alpha(class_counts.tolist())  # shape: (num_classes,)
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    effective_num = 1.0 - torch.pow(torch.tensor(beta), counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(counts)
    return weights
