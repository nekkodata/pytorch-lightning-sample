# PCB外観検査文献に着想を得たData Augmentation追加 設計

## 背景・目的

`src/transforms.py` の学習用データ拡張(`build_train_transforms`)は、Flip/Rotation/Blur/RandomErasing/GaussianNoise といった一般的な手法のみで構成されている。スマートファクトリーにおけるプリント基板(PCB)の画像ベース異常検知分野で使われている手法を調査し、実装可能な形で追加する。

### 文献調査のまとめ(前提として明記)

- 「2025年以降・査読付き・PCB特化」の条件を完全に満たし、かつ単一画像に対する `torchvision.transforms.Compose` 互換のtransformとしてそのまま実装できる論文は発見できなかった。
  - PCB分野の2025年論文の多くは幾何変換/ノイズ/色空間変換など既存実装と重複する内容の言い換え。
  - GAN/diffusionベースの欠陥合成([GAN Enhanced YOLOv11, arXiv:2501.06879](https://arxiv.org/pdf/2501.06879), 2025)は生成器学習と欠陥バンクが必要でCPU前提の本サンプルには不向き。
  - 周波数領域処理を使う2025年PCB論文([PCB-FS, *Symmetry* 2025, DOI:10.3390/sym17122020](https://doi.org/10.3390/sym17122020)、[CM-UNetv2, *Sensors* 2025, DOI:10.3390/s25164919](https://doi.org/10.3390/s25164919)、DHAEP, *Cluster Computing* 2025)は「ネットワーク内モジュール」であり、入力画像へのaugmentationではない。
- 本リポジトリは実データがCIFAR-10であり、実際のPCB画像・欠陥は存在しない。そのためどの手法を採用しても「PCB文献に着想を得た正則化」であり「本物の欠陥合成」にはならない。
- 上記の制約を踏まえ、実装可能で産業用途の異常検知/欠陥検出文脈での妥当性が確認されている2手法を採用する。ユーザーはこのギャップを理解した上で両手法の追加を承認済み。

## 追加する手法

### 1. `RandomCutPaste`

- **出典:** Li et al., *CutPaste: Self-Supervised Learning for Anomaly Detection and Localization*, CVPR 2021([arXiv:2104.04015](https://arxiv.org/abs/2104.04015))。産業用サーフェス欠陥検出向けのcopy-paste系augmentationとして、2025年に査読付きジャーナルで限定データ下での有効性が確認されている: Mohammadzadeh et al., *Utilization of Data Augmentation Techniques in Automated Inspection Systems for Defect Detection in Metals With Limited Data*, Journal of Advanced Manufacturing and Processing, 2025, DOI:[10.1002/amp2.70011](https://doi.org/10.1002/amp2.70011)。
- **位置づけ:** 元論文は自己教師あり異常検知(one-class)向け。本リポジトリは教師ありCIFAR-10分類のため、「局所テクスチャの不連続を模した正則化」(既存の`RandomErasing`の親戚。ただし塗りつぶしではなく実テクスチャを貼り付ける点が異なる)として実装する。docstringとREADMEにこの位置づけを明記する。
- **アルゴリズム:**
  1. 画像面積に対する比率 `scale=(0.02, 0.15)` とアスペクト比 `ratio=(0.3, 3.3)`(`RandomErasing`と同じレンジ)からパッチの高さ・幅を決定。
  2. 画像内のランダムな位置からパッチを切り出す(コピー元)。
  3. 別のランダムな位置に、切り出したパッチをそのまま貼り付ける(コピー先)。外部データ・欠陥バンク不要、自己参照のみ。
  4. パッチサイズが画像サイズを超える場合はクランプする。
- **パラメータ:** `p=0.3`(発生確率。既存の`RandomErasing(p=0.3)`と揃える)、`scale=(0.02, 0.15)`、`ratio=(0.3, 3.3)`。
- **実装形態:** `torch.nn.Module`、内部に確率判定を持つ自己完結型(`RandomErasing`と同様の呼び出し方)。正規化済みテンソル `(C, H, W)` に対して動作。

### 2. `RandomFrequencyJitter`

- **出典・着想:** 2025年のPCB欠陥検出論文群が「高周波成分が微小欠陥検出の鍵である」と強調している点([PCB-FS](https://doi.org/10.3390/sym17122020), [CM-UNetv2](https://doi.org/10.3390/s25164919), DHAEP)に着想を得て、ネットワーク内モジュールではなく入力画像への周波数領域augmentationとして転用する。**論文の直接実装ではなく着想の転用である**ことを明記する。
- **アルゴリズム:**
  1. `torch.fft.rfft2` で各チャンネルを振幅スペクトル・位相スペクトルに分解。
  2. 画像中心からの距離に基づく放射マスクを構築し、低周波(構造・輪郭)はほぼそのまま、高周波(テクスチャ・微細な反射特性)ほど強く影響するようスケールを滑らかに適用: `amplitude *= 1 + mask * (s - 1)`(`s` はサンプルごとの乱数)。
  3. 位相は変更せず(構造・エッジ情報を保持)、`torch.fft.irfft2` で逆変換して画像を再構成。
- **パラメータ:** `p=0.2`(既存の`GaussianBlur`/`GaussianNoise`の発生確率と揃える)、`high_freq_scale_range=(0.7, 1.3)`、`cutoff=0.3`(放射マスクの立ち上がり位置、正規化周波数 0〜1 のうちどこから高周波とみなすか)。
- **実装形態:** `torch.nn.Module`、内部に確率判定を持つ自己完結型。正規化済みテンソル `(C, H, W)` に対して動作。

## 配置場所

両方とも `build_train_transforms` 内の `preprocess`(リサイズ/正規化)の後、既存の `RandomErasing` / `GaussianNoise` と同じセクションに追加する(正規化済みテンソルに対して安全に動作するため)。順序は:

```
... preprocess
RandomErasing
RandomCutPaste       # 追加
GaussianNoise (v2.RandomApply)
RandomFrequencyJitter  # 追加
```

## エラー処理

特別なハンドリングは不要。`Compose`内でテンソル形状は保証される。`RandomCutPaste`はパッチサイズが画像サイズを超える場合にクランプする。

## テスト・検証方針

リポジトリにテストフレームワーク(pytest等)が存在しないため、新規テストスイート追加はスコープ外とする。実装後に以下を手動スモークテストで確認する:

- 出力テンソルの形状・dtypeが入力と一致すること
- `p=1.0`固定で複数回実行してもエラーが出ないこと(境界値含む)
- `build_train_transforms()` 全体を通したパイプラインが例外なく動作すること

## ドキュメント更新

`README.md` の Data Augmentation 表に2手法を追記し、出典論文へのリンクと「PCB文献に着想を得た/転用」である旨の注記を加える。
