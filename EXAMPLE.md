# PTBXLデータセットでのsuperdiagnostic class task実施例（ResNet系モデル）

このドキュメントでは、PTB-XLデータセットを使用してsuperdiagnostic class taskをResNet系モデルで実施する際の手順と前処理の詳細を説明します。

## 目次
1. [概要](#概要)
2. [環境セットアップ](#環境セットアップ)
3. [データセットの準備](#データセットの準備)
4. [Superdiagnostic Class Taskについて](#superdiagnostic-class-taskについて)
5. [ResNet系モデルについて](#resnet系モデルについて)
6. [データ前処理の詳細](#データ前処理の詳細)
7. [実行手順](#実行手順)
8. [評価方法](#評価方法)

---

## 概要

PTB-XLは、21,837件の12誘導心電図（ECG）を含む大規模な公開データセットです。このドキュメントでは、診断カテゴリのsuperclassレベルでの分類タスク（superdiagnostic task）をResNet系モデルで実施する方法を説明します。

### 使用するモデル
- **resnet1d_wang**: 軽量なResNet（3層のBasicBlock）
- **xresnet1d101**: 深層ResNet（101層）

これらのモデルは、論文のリーダーボードにおいてsuperdiagnostic taskで高い性能を示しています。

---

## 環境セットアップ

### 依存関係のインストール

condaを使用して環境を構築します：

```bash
conda env create -f ecg_env.yml
conda activate ecg_env
```

### 必要なライブラリ
- **wfdb**: 心電図データの読み込み
- **PyTorch**: ディープラーニングフレームワーク
- **fastai**: 高レベルAPIとトレーニングユーティリティ
- **scikit-learn**: データ前処理と評価
- **pandas/numpy**: データ処理

---

## データセットの準備

### データのダウンロード

提供されているスクリプトを使用してPTB-XLデータセットをダウンロードします：

```bash
./get_datasets.sh
```

このスクリプトは以下を実行します：
1. PhysioNetからPTB-XLデータセットをダウンロード
2. `data/ptbxl/`ディレクトリにデータを配置
3. メタデータ（`ptbxl_database.csv`）とSCPコード定義（`scp_statements.csv`）も取得

### データセットの構造

```
data/ptbxl/
├── ptbxl_database.csv          # メタデータとラベル
├── scp_statements.csv          # SCP診断コードの定義と分類
├── records100/                 # 100Hzサンプリングレート（低解像度）
├── records500/                 # 500Hzサンプリングレート（高解像度）
└── ...
```

---

## Superdiagnostic Class Taskについて

### タスクの定義

superdiagnostic taskは、診断ステートメントを5つの主要なカテゴリに分類するマルチラベル分類タスクです：

| クラス | 説明 | 英語名 |
|--------|------|--------|
| NORM | 正常ECG | Normal ECG |
| MI | 心筋梗塞 | Myocardial Infarction |
| STTC | ST/T変化 | ST/T Change |
| CD | 伝導障害 | Conduction Disturbance |
| HYP | 心肥大 | Hypertrophy |

### ラベル集約の仕組み

`code/utils/utils.py:171-245` の`compute_label_aggregations()`関数が以下を実行します：

1. `scp_statements.csv`から診断カテゴリ情報を読み込み
2. 各ECGのSCPコードを対応するsuperclassにマッピング
3. `diagnostic_class`列の値を使用してsuperdiagnosticラベルを生成

```python
# superdiagnosticラベルの集約例
# 元のSCPコード: {'NORM': 100, 'IMI': 80}
# → superdiagnostic: ['NORM', 'MI']
```

### データ分割

PTB-XLデータセットは、患者レベルで層別化された10-fold cross validationを使用します：

- **Fold 1-8**: 訓練データ（~17,470サンプル）
- **Fold 9**: 検証データ（~2,183サンプル）
- **Fold 10**: テストデータ（~2,184サンプル）

このフォールド情報は`ptbxl_database.csv`の`strat_fold`列に格納されています。

---

## ResNet系モデルについて

### 1. resnet1d_wang

**ファイル**: `code/models/resnet1d.py:176-190`

Wangらの論文に基づく軽量なResNetアーキテクチャ：

```python
# アーキテクチャの特徴
- 層数: 3つのBasicBlock（各1層）
- 初期フィルタ数: 128
- カーネルサイズ: [5, 3]（第1層と第2層）
- ステムカーネルサイズ: 7
- ステムストライド: 1
- プーリング: なし（ステム部分）
```

**設定**: `code/configs/fastai_configs.py:16-17`

```python
conf_fastai_resnet1d_wang = {
    'modelname': 'fastai_resnet1d_wang',
    'modeltype': 'fastai_model',
    'parameters': dict()
}
```

### 2. xresnet1d101

**ファイル**: `code/models/xresnet1d.py:184`

fastaiのXResNetアーキテクチャをベースにした深層モデル：

```python
# アーキテクチャの特徴
- 層数: [3, 4, 23, 3]の構成（101層）
- Expansion: 4（Bottleneckブロック）
- ステムサイズ: [32, 32, 64]
- カーネルサイズ: 5（デフォルト）
- BatchNorm → ReLU → Convの順序
```

**設定**: `code/configs/fastai_configs.py:57-58`

```python
conf_fastai_xresnet1d101 = {
    'modelname': 'fastai_xresnet1d101',
    'modeltype': 'fastai_model',
    'parameters': dict()
}
```

### モデルの入出力

**入力形状**:
- `(batch_size, time_steps, channels)`
- デフォルト: `(batch_size, 1000, 12)` （100Hz × 10秒 × 12誘導）

**出力**:
- `(batch_size, num_classes)`
- superdiagnostic task: `(batch_size, 5)`

### 学習設定（デフォルト）

`code/models/fastai_model.py:159-209` で定義：

```python
# ハイパーパラメータ
input_size = 2.5秒        # 入力サイズ（秒単位）→ 100Hz × 2.5 = 250サンプル
input_channels = 12       # 12誘導ECG
batch_size = 128
learning_rate = 1e-2
epochs = 50
weight_decay = 1e-2
kernel_size = 5
loss = "binary_cross_entropy"  # マルチラベル分類

# ヘッド設定
ps_head = 0.5           # ドロップアウト率
lin_ftrs_head = [128]   # 全結合層のユニット数
```

---

## データ前処理の詳細

### 1. データ読み込み

**関数**: `code/utils/utils.py:116-169` の`load_dataset()`と`load_raw_data_ptbxl()`

```python
# 100Hzサンプリングレートでの読み込み
X, Y = utils.load_dataset(datafolder, sampling_frequency=100)

# データキャッシング
# - 初回: wfdb.rdsamp()でファイルから読み込み
# - 2回目以降: raw100.npyから高速読み込み
```

**出力**:
- `X`: numpy配列、shape = `(21837, 1000, 12)`
  - 21,837サンプル × 1000タイムステップ（10秒 @ 100Hz） × 12誘導
- `Y`: pandas DataFrame（メタデータとSCPコード）

### 2. ラベル集約

**関数**: `code/utils/utils.py:171-245` の`compute_label_aggregations()`

```python
labels = utils.compute_label_aggregations(Y, datafolder, task='superdiagnostic')

# 処理内容:
# 1. scp_statements.csvから診断関連のコードを抽出
# 2. 各SCPコードをsuperdiagnosticクラスにマッピング
# 3. 新しい列'superdiagnostic'を追加
```

### 3. データ選択とマルチホットエンコーディング

**関数**: `code/utils/utils.py:247-314` の`select_data()`

```python
X, Y, y, mlb = utils.select_data(X, labels, 'superdiagnostic',
                                  min_samples=0, outputfolder)

# 処理内容:
# 1. superdiagnosticラベルが存在するサンプルのみ選択
# 2. MultiLabelBinarizerでone-hotエンコーディング
# 3. y: shape = (num_samples, 5) のバイナリ行列
```

**マルチホットエンコーディング例**:
```python
# 入力: ['NORM', 'STTC']
# 出力: [1, 0, 1, 0, 0]  (NORM, MI, STTC, CD, HYPの順)
```

### 4. データ分割

**コード**: `code/experiments/scp_experiment.py:48-56`

```python
# フォールド番号に基づく分割
X_test = X[labels.strat_fold == 10]
y_test = y[labels.strat_fold == 10]

X_val = X[labels.strat_fold == 9]
y_val = y[labels.strat_fold == 9]

X_train = X[labels.strat_fold <= 8]
y_train = y[labels.strat_fold <= 8]
```

### 5. 信号の標準化

**関数**: `code/utils/utils.py:316-333` の`preprocess_signals()`

```python
X_train, X_val, X_test = utils.preprocess_signals(
    X_train, X_val, X_test, outputfolder
)

# 処理内容:
# 1. 訓練データ全体から平均と標準偏差を計算
# 2. StandardScalerで各サンプルを正規化（平均0、分散1）
# 3. 検証データとテストデータにも同じスケーラーを適用
```

**重要**:
- スケーラーは訓練データのみでfitする（データリークを防止）
- `standard_scaler.pkl`として保存され、推論時に再利用可能

### 6. データ拡張（チャンク化）

**コード**: `code/models/fastai_model.py:168-180`

fastai_modelは、長いECG信号を複数のチャンクに分割できます：

```python
# デフォルト設定
input_size = int(2.5 * 100) = 250サンプル
chunk_length_train = 2 * input_size = 500サンプル
chunk_length_valid = input_size = 250サンプル
stride_length_train = input_size = 250サンプル
stride_length_valid = input_size // 2 = 125サンプル

# チャンク化の有効化/無効化
chunkify_train = False  # 訓練時はデフォルトでオフ
chunkify_valid = True   # 検証時はデフォルトでオン
```

**チャンク化の効果**:
- より多くのトレーニングサンプルを生成（データ拡張）
- 検証時は重複チャンクの予測を集約（平均またはmax）

---

## 実行手順

### 基本的な実行例

```python
# code/ディレクトリで実行
cd code

# 単一タスクの実行例
from experiments.scp_experiment import SCP_Experiment
from configs.fastai_configs import conf_fastai_resnet1d_wang, conf_fastai_xresnet1d101

datafolder = '../data/ptbxl/'
outputfolder = '../output/'

# ResNet系モデルを選択
models = [
    conf_fastai_resnet1d_wang,
    conf_fastai_xresnet1d101
]

# Superdiagnostic taskの実験を作成
e = SCP_Experiment(
    experiment_name='exp_superdiagnostic',
    task='superdiagnostic',
    datafolder=datafolder,
    outputfolder=outputfolder,
    models=models,
    sampling_frequency=100,  # 100Hzサンプリング
    min_samples=0,           # 最小サンプル数の閾値
    train_fold=8,            # 訓練用フォールド（1-8）
    val_fold=9,              # 検証用フォールド
    test_fold=10             # テスト用フォールド
)

# ステップ1: データ準備
e.prepare()

# ステップ2: モデル訓練
e.perform()

# ステップ3: 評価
e.evaluate()
```

### 実行ステップの詳細

#### ステップ1: `e.prepare()`

`code/experiments/scp_experiment.py:37-66`

```python
# 実行内容:
1. データセットをロード（100Hz）
2. ラベルをsuperdiagnosticクラスに集約
3. データを選択しone-hotエンコーディング
4. train/val/testに分割
5. 信号を標準化
6. データとラベルを保存
```

**出力ファイル**:
```
output/exp_superdiagnostic/data/
├── y_train.npy              # 訓練ラベル
├── y_val.npy                # 検証ラベル
├── y_test.npy               # テストラベル
├── mlb.pkl                  # MultiLabelBinarizer
└── standard_scaler.pkl      # StandardScaler
```

#### ステップ2: `e.perform()`

`code/experiments/scp_experiment.py:81-137`

```python
# 各モデルについて:
1. モデルを初期化
2. fit(X_train, y_train, X_val, y_val)で訓練
3. 訓練・検証・テストデータの予測を保存
```

**訓練プロセス（fastai_model）**:

`code/models/fastai_model.py:210-286`

```python
# 訓練の流れ:
1. データローダーを作成（TimeseriesDatasetCrops）
2. 学習率ファインダーを実行 → lr_find.png
3. One Cycle Policyで訓練（50エポック）
4. 損失曲線を保存 → losses.png
5. モデルを保存 → {modelname}.pth
```

**出力ファイル（モデルごと）**:
```
output/exp_superdiagnostic/models/fastai_resnet1d_wang/
├── fastai_resnet1d_wang.pth  # 訓練済みモデル
├── lr_find.png               # 学習率探索プロット
├── losses.png                # 訓練・検証損失
├── y_train_pred.npy          # 訓練データの予測
├── y_val_pred.npy            # 検証データの予測
└── y_test_pred.npy           # テストデータの予測
```

#### ステップ3: `e.evaluate()`

`code/experiments/scp_experiment.py:139-222`

```python
# 評価内容:
1. テストデータの予測を読み込み
2. ブートストラップサンプリング（オプション）
3. macro AUCを計算
4. 結果をCSVに保存
```

**評価指標**:
- **macro_auc**: クラスごとのAUCの平均値

**出力ファイル**:
```
output/exp_superdiagnostic/models/fastai_resnet1d_wang/results/
└── te_results.csv            # テスト結果（point, mean, lower, upper）
```

### 複数実験の一括実行

`code/reproduce_results.py`を参考にした例：

```python
from experiments.scp_experiment import SCP_Experiment
from configs.fastai_configs import *

datafolder = '../data/ptbxl/'
outputfolder = '../output/'

# 複数のモデルを定義
models = [
    conf_fastai_resnet1d_wang,
    conf_fastai_xresnet1d101,
    conf_fastai_inception1d,
]

# 複数のタスクを実行
experiments = [
    ('exp_all', 'all'),
    ('exp_diagnostic', 'diagnostic'),
    ('exp_subdiagnostic', 'subdiagnostic'),
    ('exp_superdiagnostic', 'superdiagnostic'),
    ('exp_form', 'form'),
    ('exp_rhythm', 'rhythm')
]

for name, task in experiments:
    e = SCP_Experiment(name, task, datafolder, outputfolder, models)
    e.prepare()
    e.perform()
    e.evaluate()
```

---

## 評価方法

### メトリクス

**Superdiagnostic taskの主要メトリクス**:

- **macro AUC**: クラスごとのROC-AUCの平均値

`code/utils/utils.py:35` で計算：

```python
from sklearn.metrics import roc_auc_score

macro_auc = roc_auc_score(y_true, y_pred, average='macro')
```

### ブートストラップ評価

信頼区間を計算するため、ブートストラップサンプリングを使用できます：

```python
# ブートストラップ評価を有効化
e.evaluate(bootstrap_eval=True, n_bootstraping_samples=100)

# 結果:
# - point: ポイント推定値
# - mean: ブートストラップ平均
# - lower: 5パーセンタイル
# - upper: 95パーセンタイル
```

### リーダーボード結果（論文より）

README.mdのセクション4: PTB-XL Diagnostic superclasses

| Model | AUC | 論文 | コード |
|------|-----|------|--------|
| resnet1d_wang | 0.930(05) | [論文](https://doi.org/10.1109/jbhi.2020.3022989) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/) |
| xresnet1d101 | 0.928(05) | [論文](https://doi.org/10.1109/jbhi.2020.3022989) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/) |

**注**: カッコ内は不確実性（標準誤差×1000）

---

## トラブルシューティング

### メモリ不足の場合

バッチサイズを減らす：

```python
conf_fastai_resnet1d_wang = {
    'modelname': 'fastai_resnet1d_wang',
    'modeltype': 'fastai_model',
    'parameters': {'bs': 64}  # デフォルト128 → 64
}
```

### GPU利用可能性の確認

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### データキャッシュの再生成

キャッシュファイル（`raw100.npy`）を削除すると、次回実行時に再生成されます：

```bash
rm data/ptbxl/raw100.npy
```

---

## カスタマイズ例

### 学習率の変更

```python
conf_custom_resnet = {
    'modelname': 'custom_resnet1d_wang',
    'modeltype': 'fastai_model',
    'parameters': {'lr': 5e-3}  # デフォルト1e-2
}
```

### エポック数の変更

```python
conf_custom_resnet = {
    'modelname': 'custom_resnet1d_wang',
    'modeltype': 'fastai_model',
    'parameters': {'epochs': 100}  # デフォルト50
}
```

### Early Stoppingの有効化

```python
conf_custom_resnet = {
    'modelname': 'custom_resnet1d_wang_earlystop',
    'modeltype': 'fastai_model',
    'parameters': {
        'early_stopping': 'macro_auc',  # または 'valid_loss', 'fmax'
        'epochs': 100
    }
}
```

### 500Hzサンプリングレートの使用

```python
e = SCP_Experiment(
    experiment_name='exp_superdiagnostic_500hz',
    task='superdiagnostic',
    datafolder='../data/ptbxl/',
    outputfolder='../output/',
    models=models,
    sampling_frequency=500  # 100 → 500に変更
)
```

---

## 参考文献

### PTB-XLデータセット

```bibtex
@article{Wagner:2020PTBXL,
    title = {{PTB-XL, a large publicly available electrocardiography dataset}},
    author = {Patrick Wagner and Nils Strodthoff and Ralf-Dieter Bousseljot and
              Dieter Kreiseler and Fatima I. Lunze and Wojciech Samek and
              Tobias Schaeffter},
    journal = {Scientific Data},
    volume = {7},
    number = {1},
    pages = {154},
    year = {2020},
    doi = {10.1038/s41597-020-0495-6}
}
```

### ベンチマーク論文

```bibtex
@article{Strodthoff:2020Deep,
    title = {Deep Learning for {ECG} Analysis: Benchmarks and Insights from {PTB}-{XL}},
    author = {Nils Strodthoff and Patrick Wagner and Tobias Schaeffter and Wojciech Samek},
    journal = {{IEEE} Journal of Biomedical and Health Informatics},
    volume = {25},
    number = {5},
    pages = {1519-1528},
    year = {2021},
    doi = {10.1109/jbhi.2020.3022989}
}
```

---

## まとめ

このドキュメントでは、PTB-XLデータセットでのsuperdiagnostic class taskの実施方法を詳しく説明しました：

1. **データ準備**: 100Hzサンプリング、ラベル集約、標準化
2. **モデル選択**: resnet1d_wang（軽量）、xresnet1d101（深層）
3. **訓練**: fastaiフレームワーク、One Cycle Policy、50エポック
4. **評価**: macro AUC、ブートストラップ信頼区間

このフレームワークを使用することで、最新のディープラーニングモデルを用いた心電図分類の研究を簡単に開始できます。
