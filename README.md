# pytorch-lightning-sample

PyTorch Lightning を用いた、クラス不均衡(long-tailed)な CIFAR-10 データセットに対する画像分類サンプル実装。EfficientNet-B0 をバックボーンとし、Focal Loss で不均衡データに対応する。CPU 実行を前提としている。

## セットアップ

パッケージ管理には [uv](https://docs.astral.sh/uv/) を使用する。

```bash
uv sync
```

## 実行

```bash
uv run python -m src.train
```

主なオプション(`src/train.py` の `argparse` 定義に準拠):

| オプション | デフォルト | 説明 |
| --- | --- | --- |
| `--data-dir` | `./data` | CIFAR-10 のダウンロード/保存先 |
| `--num-classes` | `10` | クラス数 |
| `--img-size` | `64` | 入力画像サイズ |
| `--batch-size` | `32` | バッチサイズ |
| `--num-workers` | `2` | DataLoader のワーカー数 |
| `--imbalance-factor` | `0.1` | 最小/最大クラスサイズの比率(不均衡度) |
| `--val-split` | `0.1` | 学習データからの検証データ分割比率 |
| `--gamma` | `2.0` | Focal Loss の gamma |
| `--lr` | `1e-3` | 学習率 |
| `--weight-decay` | `1e-4` | AdamW の weight decay |
| `--max-epochs` | `10` | 最大エポック数 |
| `--pretrained` | `False` | EfficientNet-B0 の事前学習済み重みを使用するか |
| `--seed` | `42` | 乱数シード |

## コード構成

```
src/
├── train.py               # エントリポイント。DataModule/Model/Trainer を組み立てて学習・テストを実行
├── transforms.py          # 学習用・評価用の画像変換(EfficientNet の前処理 + 学習時のみ Flip/Rotation/Blur/Erasing/Noise 等の augmentation)
├── data/
│   ├── datamodule.py      # CIFAR10ImbalancedDataModule (LightningDataModule)。CIFAR-10 をダウンロードし、long-tailed 化した train/val/test の DataLoader を提供
│   └── imbalance.py       # make_long_tailed_indices。クラスごとに指数的に減衰するサンプル数を持つ long-tailed な部分集合のインデックスを生成
├── losses/
│   ├── focal_loss.py      # FocalLoss ([torchvision.ops.sigmoid_focal_loss](https://docs.pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html) ベース。one-hotターゲットに対しクラスごとに独立した2値分類として計算するマルチラベル的な定式化)
│   └── class_balance.py   # compute_class_balanced_alpha。クラスごとの有効サンプル数に基づく重み (Cui et al., 2019)。現状どこからも呼ばれていない未使用ユーティリティ
└── models/
    └── classifier.py      # EfficientNetClassifier (LightningModule)。EfficientNet-B0 + FocalLoss、Accuracy/F1 のロギング、AdamW + CosineAnnealingLR を定義
```

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

### 学習フロー(`src/train.py`)

1. `CIFAR10ImbalancedDataModule` で CIFAR-10 を取得し、`imbalance-factor` に応じて long-tailed な学習データを構築(検証データは元の分布のまま分割)。
2. `EfficientNetClassifier` を構築(`torchvision` の EfficientNet-B0 をベースに、出力層をクラス数に合わせて置換)。
3. `val_f1`(macro F1)を監視する `ModelCheckpoint` と `EarlyStopping`(patience=5)をコールバックとして設定。
4. `pl.Trainer`(CPU, `max_epochs` 分)で学習後、テストデータで評価。
