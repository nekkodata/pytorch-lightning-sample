from typing import Optional

from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
from torchvision.transforms import v2


def build_train_transforms(
    weights: EfficientNet_B0_Weights = EfficientNet_B0_Weights.DEFAULT,
    img_size: Optional[int] = None,
) -> transforms.Compose:
    preprocess = weights.transforms(**_size_overrides(img_size))
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
            preprocess,
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            # preprocess で正規化済み(値域は [0, 1] ではない)のため clip=False
            v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.05, clip=False)], p=0.2),
        ]
    )


def build_eval_transforms(
    weights: EfficientNet_B0_Weights = EfficientNet_B0_Weights.DEFAULT,
    img_size: Optional[int] = None,
) -> transforms.Compose:
    return weights.transforms(**_size_overrides(img_size))


def _size_overrides(img_size: Optional[int]) -> dict:
    if img_size is None:
        return {}
    return {"crop_size": img_size, "resize_size": img_size}
