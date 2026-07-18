# PCB-Inspired Data Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two PCB外観検査文献に着想を得た `torch.nn.Module` ベースのdata augmentation変換(`RandomCutPaste`, `RandomFrequencyJitter`)を `src/transforms.py` に追加し、学習パイプラインへ組み込み、READMEに出典を明記する。

**Architecture:** `src/transforms.py` に2つの自己完結型(`RandomErasing`と同様、内部に確率`p`を持つ)`torch.nn.Module`サブクラスを追加し、`build_train_transforms` の `preprocess` 後のセクションに追加する。正規化済みテンソル `(C, H, W)` に対して動作する。

**Tech Stack:** PyTorch (`torch.nn.Module`, `torch.fft`), torchvision.transforms。既存の依存関係のみで完結し、新規パッケージ追加は不要。

## Global Constraints

- リポジトリにはpytest等のテストフレームワークが存在しない([design doc](../specs/2026-07-18-pcb-inspired-augmentations-design.md)参照)。新規テストスイートは追加せず、`.venv/bin/python` での手動スモークテストスクリプトで検証する。
- 実行コマンドは `.venv/bin/python` を使用する(このマシンでは `uv run python` が依存解決に時間がかかり120秒でタイムアウトするため。`.venv` には既に torch 2.9.1+cpu がインストール済みであることを確認済み)。
- 新規クラスは `RandomErasing` と同じ呼び出しスタイル(コンストラクタに `p` を持つ自己完結型)に揃える。`RandomApply` でラップしない。
- パラメータのデフォルト値は設計ドキュメントの値を厳守する: `RandomCutPaste(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3))`、`RandomFrequencyJitter(p=0.2, high_freq_scale_range=(0.7, 1.3), cutoff=0.3)`。

---

### Task 1: `RandomCutPaste` クラスの実装

**Files:**
- Modify: `src/transforms.py`(先頭に `import torch` を追加、クラスを `build_train_transforms` の前に追加)
- 検証スクリプト(一時ファイル): `/tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_cutpaste.py`

**Interfaces:**
- Produces: `RandomCutPaste(torch.nn.Module)` — コンストラクタ `__init__(self, p: float = 0.3, scale: tuple[float, float] = (0.02, 0.15), ratio: tuple[float, float] = (0.3, 3.3)) -> None`、`forward(self, img: torch.Tensor) -> torch.Tensor`(入力・出力とも `(C, H, W)` の同一shape/dtype)。

- [ ] **Step 1: 検証スクリプトを書く(この時点ではimport元のクラスが無いので失敗する)**

```python
# /tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_cutpaste.py
import torch

from src.transforms import RandomCutPaste

torch.manual_seed(0)

for shape in [(3, 64, 64), (3, 224, 224), (1, 33, 47)]:
    img = torch.randn(*shape)

    # p=1.0: 常に適用され、shape/dtypeが保たれること
    cp = RandomCutPaste(p=1.0)
    out = cp(img)
    assert out.shape == img.shape, f"shape mismatch: {out.shape} vs {img.shape}"
    assert out.dtype == img.dtype, f"dtype mismatch: {out.dtype} vs {img.dtype}"

    # p=0.0: 何も変更されない(no-op)こと
    cp0 = RandomCutPaste(p=0.0)
    assert torch.equal(cp0(img), img), "p=0.0 should be a no-op"

# パッチ面積が画像面積を超えないような極端に小さい画像でもエラーにならないこと
tiny = torch.randn(3, 4, 4)
RandomCutPaste(p=1.0)(tiny)

print("RandomCutPaste: ALL CHECKS PASSED")
```

- [ ] **Step 2: 検証スクリプトを実行し、失敗することを確認する**

Run: `cd /home/sh70k/dev/fujikura/pytorch-lightning-sample && .venv/bin/python /tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_cutpaste.py`
Expected: `ImportError: cannot import name 'RandomCutPaste' from 'src.transforms'`

- [ ] **Step 3: `src/transforms.py` に `RandomCutPaste` を実装する**

`src/transforms.py` の先頭 import ブロックを以下に変更(既存の `from typing import Optional` の前に `torch` の import を追加):

```python
from typing import Optional

import torch
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
from torchvision.transforms import v2
```

`build_train_transforms` 関数定義の直前に、以下のクラスを追加する:

```python
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
```

- [ ] **Step 4: 検証スクリプトを再実行し、成功することを確認する**

Run: `cd /home/sh70k/dev/fujikura/pytorch-lightning-sample && .venv/bin/python /tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_cutpaste.py`
Expected: `RandomCutPaste: ALL CHECKS PASSED`

- [ ] **Step 5: コミット**

```bash
cd /home/sh70k/dev/fujikura/pytorch-lightning-sample
git add src/transforms.py
git commit -m "$(cat <<'EOF'
Add RandomCutPaste augmentation to transforms.py

CutPaste-style patch copy augmentation, adapted from CVPR 2021 /
validated for limited-data industrial defect detection in a 2025
peer-reviewed paper (DOI:10.1002/amp2.70011).
EOF
)"
```

---

### Task 2: `RandomFrequencyJitter` クラスの実装

**Files:**
- Modify: `src/transforms.py`(Task 1 で追加した `RandomCutPaste` の直後にクラスを追加)
- 検証スクリプト(一時ファイル): `/tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_freqjitter.py`

**Interfaces:**
- Consumes: Task 1で追加した `import torch`(既にファイル先頭にあるためそのまま利用)。
- Produces: `RandomFrequencyJitter(torch.nn.Module)` — コンストラクタ `__init__(self, p: float = 0.2, high_freq_scale_range: tuple[float, float] = (0.7, 1.3), cutoff: float = 0.3) -> None`、`forward(self, img: torch.Tensor) -> torch.Tensor`(入力・出力とも `(C, H, W)` の同一shape/dtype)。

- [ ] **Step 1: 検証スクリプトを書く**

```python
# /tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_freqjitter.py
import torch

from src.transforms import RandomFrequencyJitter

torch.manual_seed(0)

for shape in [(3, 64, 64), (3, 224, 224), (1, 33, 47)]:
    img = torch.randn(*shape)

    fj = RandomFrequencyJitter(p=1.0)
    out = fj(img)
    assert out.shape == img.shape, f"shape mismatch: {out.shape} vs {img.shape}"
    assert out.dtype == img.dtype, f"dtype mismatch: {out.dtype} vs {img.dtype}"
    assert torch.isfinite(out).all(), "output contains non-finite values"

    fj0 = RandomFrequencyJitter(p=0.0)
    assert torch.equal(fj0(img), img), "p=0.0 should be a no-op"

print("RandomFrequencyJitter: ALL CHECKS PASSED")
```

- [ ] **Step 2: 検証スクリプトを実行し、失敗することを確認する**

Run: `cd /home/sh70k/dev/fujikura/pytorch-lightning-sample && .venv/bin/python /tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_freqjitter.py`
Expected: `ImportError: cannot import name 'RandomFrequencyJitter' from 'src.transforms'`

- [ ] **Step 3: `src/transforms.py` に `RandomFrequencyJitter` を実装する**

`RandomCutPaste` クラスの直後(`build_train_transforms` 関数定義の前)に追加する:

```python
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
```

- [ ] **Step 4: 検証スクリプトを再実行し、成功することを確認する**

Run: `cd /home/sh70k/dev/fujikura/pytorch-lightning-sample && .venv/bin/python /tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_freqjitter.py`
Expected: `RandomFrequencyJitter: ALL CHECKS PASSED`

- [ ] **Step 5: コミット**

```bash
cd /home/sh70k/dev/fujikura/pytorch-lightning-sample
git add src/transforms.py
git commit -m "$(cat <<'EOF'
Add RandomFrequencyJitter augmentation to transforms.py

FFT amplitude-spectrum jitter that perturbs high-frequency content
while preserving phase, adapted from the frequency-domain emphasis
in 2025 PCB defect detection papers (PCB-FS: DOI:10.3390/sym17122020,
CM-UNetv2: DOI:10.3390/s25164919) — those papers use frequency
processing as a network module; this is an input-level adaptation.
EOF
)"
```

---

### Task 3: `build_train_transforms` パイプラインへの組み込み

**Files:**
- Modify: `src/transforms.py:8-24`(`build_train_transforms` 関数)
- 検証スクリプト(一時ファイル): `/tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_pipeline.py`

**Interfaces:**
- Consumes: Task 1 の `RandomCutPaste(p, scale, ratio)`、Task 2 の `RandomFrequencyJitter(p, high_freq_scale_range, cutoff)`。両方ともデフォルト引数のみで呼び出す。
- Produces: `build_train_transforms` が返す `transforms.Compose` に2変換が組み込まれた状態(既存の呼び出しシグネチャ `build_train_transforms(weights=..., img_size=...) -> transforms.Compose` は変更しない)。

- [ ] **Step 1: 検証スクリプトを書く(現時点では `RandomCutPaste`/`RandomFrequencyJitter` がパイプラインに未組込みなので、Compose内のtransformリストにクラスが含まれず失敗する)**

```python
# /tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_pipeline.py
import torch
from PIL import Image

from src.transforms import RandomCutPaste, RandomFrequencyJitter, build_train_transforms

pipeline = build_train_transforms(img_size=64)

transform_types = {type(t) for t in pipeline.transforms}
assert RandomCutPaste in transform_types, "RandomCutPaste is not wired into build_train_transforms"
assert RandomFrequencyJitter in transform_types, "RandomFrequencyJitter is not wired into build_train_transforms"

# エンドツーエンドでエラーなく動作すること(PILの適当な画像を通す)
dummy = Image.new("RGB", (96, 96), color=(128, 64, 32))
for _ in range(5):
    out = pipeline(dummy)
    assert isinstance(out, torch.Tensor)
    assert out.shape[-2:] == (64, 64)
    assert torch.isfinite(out).all()

print("build_train_transforms pipeline: ALL CHECKS PASSED")
```

- [ ] **Step 2: 検証スクリプトを実行し、失敗することを確認する**

Run: `cd /home/sh70k/dev/fujikura/pytorch-lightning-sample && .venv/bin/python /tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_pipeline.py`
Expected: `AssertionError: RandomCutPaste is not wired into build_train_transforms`

- [ ] **Step 3: `build_train_transforms` を修正する**

`src/transforms.py` の `build_train_transforms` 関数を以下に置き換える(既存の docstring は無いのでそのまま関数本体のみ変更):

```python
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
```

- [ ] **Step 4: 検証スクリプトを再実行し、成功することを確認する**

Run: `cd /home/sh70k/dev/fujikura/pytorch-lightning-sample && .venv/bin/python /tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_pipeline.py`
Expected: `build_train_transforms pipeline: ALL CHECKS PASSED`

- [ ] **Step 5: コミット**

```bash
cd /home/sh70k/dev/fujikura/pytorch-lightning-sample
git add src/transforms.py
git commit -m "$(cat <<'EOF'
Wire RandomCutPaste and RandomFrequencyJitter into build_train_transforms

Both run after the EfficientNet preprocess step, alongside the
existing RandomErasing/GaussianNoise, matching the file's convention
of applying photometric/texture augmentations to the normalized
tensor.
EOF
)"
```

---

### Task 4: README.md の Data Augmentation 表を更新

**Files:**
- Modify: `README.md:53-65`(Data Augmentation セクションの表)

**Interfaces:**
- Consumes: なし(ドキュメントのみの変更)。
- Produces: なし(このタスクの後続タスクはない)。

- [ ] **Step 1: README.md の Data Augmentation 表を更新する**

`README.md` の以下の表(53-65行目)を:

```markdown
### Data Augmentation(`src/transforms.py`)

学習データ(`build_train_transforms`)にのみ以下を適用。検証・テストデータ(`build_eval_transforms`)は EfficientNet の標準前処理のみでaugmentationなし。

| 変換 | 内容 |
| --- | --- |
| `RandomHorizontalFlip(p=0.5)` | 左右反転 |
| `RandomVerticalFlip(p=0.2)` | 上下反転 |
| `RandomRotation(degrees=15)` | ランダム回転 |
| `RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2)` | ランダムぼかし |
| EfficientNet 標準前処理 | リサイズ/クロップ/正規化 |
| `RandomErasing(p=0.3, scale=(0.02, 0.15))` | ランダム矩形マスク(Cutout的効果) |
| `RandomApply([v2.GaussianNoise(sigma=0.05, clip=False)], p=0.2)` | ランダムガウシアンノイズ(正規化後の値域のため `clip=False`) |
```

以下に置き換える:

```markdown
### Data Augmentation(`src/transforms.py`)

学習データ(`build_train_transforms`)にのみ以下を適用。検証・テストデータ(`build_eval_transforms`)は EfficientNet の標準前処理のみでaugmentationなし。

| 変換 | 内容 |
| --- | --- |
| `RandomHorizontalFlip(p=0.5)` | 左右反転 |
| `RandomVerticalFlip(p=0.2)` | 上下反転 |
| `RandomRotation(degrees=15)` | ランダム回転 |
| `RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2)` | ランダムぼかし |
| EfficientNet 標準前処理 | リサイズ/クロップ/正規化 |
| `RandomErasing(p=0.3, scale=(0.02, 0.15))` | ランダム矩形マスク(Cutout的効果) |
| `RandomCutPaste(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3))` | パッチのコピー&ペーストで局所テクスチャ不連続を模した拡張。産業用サーフェス欠陥検出向けの [CutPaste](https://arxiv.org/abs/2104.04015)(Li et al., CVPR 2021)系手法。限定データでの有効性は2025年の査読付き論文([Mohammadzadeh et al., 2025, DOI:10.1002/amp2.70011](https://doi.org/10.1002/amp2.70011))でも確認されている。※元手法は自己教師あり異常検知向けであり、本リポジトリでは教師あり分類向けの正則化として転用している |
| `RandomApply([v2.GaussianNoise(sigma=0.05, clip=False)], p=0.2)` | ランダムガウシアンノイズ(正規化後の値域のため `clip=False`) |
| `RandomFrequencyJitter(p=0.2, high_freq_scale_range=(0.7, 1.3), cutoff=0.3)` | FFTで振幅・位相に分解し、位相(構造情報)を保持したまま高周波成分の振幅のみをランダムスケール。2025年のPCB欠陥検出論文群([PCB-FS, *Symmetry* 2025, DOI:10.3390/sym17122020](https://doi.org/10.3390/sym17122020); [CM-UNetv2, *Sensors* 2025, DOI:10.3390/s25164919](https://doi.org/10.3390/s25164919))が強調する「高周波成分が微小欠陥検出の鍵」という着想を、ネットワークモジュールではなく入力拡張に転用したもの |
```

- [ ] **Step 2: 表が正しくレンダリングされることを目視確認する**

Run: `cat README.md | sed -n '50,70p'`
Expected: 更新後の表がMarkdownとして崩れずに表示される(パイプ `|` の数がヘッダー行と一致している)こと。

- [ ] **Step 3: コミット**

```bash
cd /home/sh70k/dev/fujikura/pytorch-lightning-sample
git add README.md
git commit -m "$(cat <<'EOF'
Document RandomCutPaste and RandomFrequencyJitter in README

Adds the two new augmentations to the Data Augmentation table with
their source citations and honest framing of what is a direct
implementation vs. an adaptation of PCB defect-detection research.
EOF
)"
```

---

### Task 5: 全体の最終スモークテスト

**Files:**
- なし(既存ファイルへの変更なし。検証のみ)

**Interfaces:**
- Consumes: Task 1〜4 の全ての成果物。
- Produces: なし(最終確認タスク)。

- [ ] **Step 1: `src/transforms.py` 全体をインポートしてエラーが出ないことを確認する**

Run: `cd /home/sh70k/dev/fujikura/pytorch-lightning-sample && .venv/bin/python -c "from src.transforms import build_train_transforms, build_eval_transforms, RandomCutPaste, RandomFrequencyJitter; print('import OK')"`
Expected: `import OK`

- [ ] **Step 2: 学習パイプラインをデフォルト引数(img_sizeなし)で通し、複数回実行してもエラーが出ないことを確認する**

Run:
```bash
cd /home/sh70k/dev/fujikura/pytorch-lightning-sample && .venv/bin/python -c "
import torch
from PIL import Image
from src.transforms import build_train_transforms, build_eval_transforms

train_pipeline = build_train_transforms()
eval_pipeline = build_eval_transforms()
dummy = Image.new('RGB', (96, 96), color=(200, 100, 50))

for _ in range(20):
    out = train_pipeline(dummy)
    assert torch.isfinite(out).all()

out_eval = eval_pipeline(dummy)
assert torch.isfinite(out_eval).all()
print('full pipeline smoke test OK, train shape:', out.shape, 'eval shape:', out_eval.shape)
"
```
Expected: `full pipeline smoke test OK, train shape: torch.Size([3, 224, 224]) eval shape: torch.Size([3, 224, 224])`(EfficientNet_B0_Weights.DEFAULTのデフォルト解像度に応じて実際のサイズは変わる可能性があるが、エラーなく完了しshapeが表示されること)

- [ ] **Step 3: 一時検証スクリプトを削除する**

Run: `rm -f /tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_cutpaste.py /tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_freqjitter.py /tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_pipeline.py /tmp/claude-1000/-home-sh70k-dev-fujikura-pytorch-lightning-sample/96f7f42d-00e1-4d03-9302-72d4c62c8778/scratchpad/verify_augs.py`
Expected: コマンドが正常終了する(出力なし)

- [ ] **Step 4: `git status` と `git log --oneline -6` で全コミットが揃っていることを確認する**

Run: `cd /home/sh70k/dev/fujikura/pytorch-lightning-sample && git status && git log --oneline -6`
Expected: `git status` が clean(スモークテストの一時ファイルは `/tmp` 配下のため対象外)、`git log` に Task 1〜4 の4コミットが順に並んでいる
