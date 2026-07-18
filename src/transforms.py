from typing import Optional

import torch
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
from torchvision.transforms import v2


class RandomCutPaste(torch.nn.Module):
    """CutPaste (Li et al., CVPR 2021, arXiv:2104.04015) 系のパッチコピー&ペースト拡張。

    産業用サーフェス欠陥検出におけるcopy-paste系augmentationの有効性は
    2025年の査読付き論文でも確認されている
    (Mohammadzadeh et al., Journal of Advanced Manufacturing and Processing, 2025,
    DOI:10.1002/amp2.70011)。
    元手法は自己教師あり異常検知(one-class)向けだが、本リポジトリは教師あり分類のため
    局所テクスチャの不連続を模した正則化(RandomErasing の親戚)として用いる。
    """

    def __init__(
        self,
        p: float = 0.3,
        scale: tuple[float, float] = (0.02, 0.15),
        ratio: tuple[float, float] = (0.3, 3.3),
    ) -> None:
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return img

        _, height, width = img.shape
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.empty(1).uniform_(self.ratio[0], self.ratio[1]).item()

            patch_h = min(int(round((target_area * aspect_ratio) ** 0.5)), height)
            patch_w = min(int(round((target_area / aspect_ratio) ** 0.5)), width)
            if patch_h < 1 or patch_w < 1:
                continue

            src_top = torch.randint(0, height - patch_h + 1, (1,)).item()
            src_left = torch.randint(0, width - patch_w + 1, (1,)).item()
            dst_top = torch.randint(0, height - patch_h + 1, (1,)).item()
            dst_left = torch.randint(0, width - patch_w + 1, (1,)).item()

            patch = img[:, src_top : src_top + patch_h, src_left : src_left + patch_w].clone()
            img = img.clone()
            img[:, dst_top : dst_top + patch_h, dst_left : dst_left + patch_w] = patch
            return img

        return img


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
