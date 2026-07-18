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


class RandomFrequencyJitter(torch.nn.Module):
    """PCB欠陥検出文献(PCB-FS: Symmetry 2025, DOI:10.3390/sym17122020;
    CM-UNetv2: Sensors 2025, DOI:10.3390/s25164919 等)における
    「高周波成分が微小欠陥検出の鍵である」という着想を、
    ネットワーク内モジュールではなく入力画像への周波数領域augmentationへ転用したもの。
    位相(構造・エッジ情報)は保持したまま、高周波成分の振幅のみをランダムスケールする。
    """

    def __init__(
        self,
        p: float = 0.2,
        high_freq_scale_range: tuple[float, float] = (0.7, 1.3),
        cutoff: float = 0.3,
    ) -> None:
        super().__init__()
        self.p = p
        self.high_freq_scale_range = high_freq_scale_range
        self.cutoff = cutoff

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return img

        _, height, width = img.shape
        spectrum = torch.fft.rfft2(img)
        amplitude = spectrum.abs()
        phase = torch.angle(spectrum)

        freq_h = torch.fft.fftfreq(height, device=img.device).abs()
        freq_w = torch.fft.rfftfreq(width, device=img.device).abs()
        radial = torch.sqrt(freq_h[:, None] ** 2 + freq_w[None, :] ** 2)
        radial = radial / radial.max()
        mask = torch.clamp((radial - self.cutoff) / (1.0 - self.cutoff + 1e-8), min=0.0, max=1.0)

        scale = torch.empty(1).uniform_(*self.high_freq_scale_range).item()
        amplitude = amplitude * (1.0 + mask * (scale - 1.0))

        spectrum = torch.polar(amplitude, phase)
        return torch.fft.irfft2(spectrum, s=(height, width))


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
            RandomCutPaste(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            # preprocess で正規化済み(値域は [0, 1] ではない)のため clip=False
            v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.05, clip=False)], p=0.2),
            RandomFrequencyJitter(p=0.2, high_freq_scale_range=(0.7, 1.3), cutoff=0.3),
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
