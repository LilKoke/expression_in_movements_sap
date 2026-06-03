# expression_in_movements_sap

モーションキャプチャ（TRC）の歩行系列からの感情分類。2 つのモデリング手法を比較する:

- **A** — expert / 手作り特徴量 + 古典ML（RandomForest / SVM）
- **B** — 生の姿勢系列に対するニューラルネット（LSTM / 1D-CNN）

感情ラベルは TRC のファイル名コードに由来: `TRE=sad`, `COE=angry`, `NEE=neutral`,
`JOE=happy`。

全体設計は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)、ロードマップ / Phase 分割は
issue #1 を参照。指標の定義は [docs/FAQ.md](docs/FAQ.md)、結果は
[docs/RESULTS.md](docs/RESULTS.md)。

## セットアップ

Python 3.13 と [`uv`](https://docs.astral.sh/uv/) が必要。

```sh
uv sync          # .venv 作成 + 依存インストール（dev グループ込み）
uv run pytest    # テスト実行
```

元の `.trc` ファイルは `data/raw/` 配下に置く（`data/` ディレクトリは gitignore 対象）。

**生データ無しで作業する場合（例: expert-features チーム、GPU 無し）:** 処理済み
データセットは Google Drive で配布する — clone → `uv sync` → リポジトリ直下で
`expr-data-processed.zip` を解凍。手順は [docs/DATA_SETUP.md](docs/DATA_SETUP.md)。
GPU は不要で、`outputs/` 配下の NN の重みも不要。

## パイプライン（CLI）

各ステップは薄いエントリポイント。ロジックはすべて `src/expr_movements/` にある。

```sh
uv run expr-parse          --raw-dir data/raw        --out-dir data/interim
uv run expr-build-dataset  --interim-dir data/interim --out-dir data/processed   # 密テンソルが要るなら + --target-frames N
uv run expr-featurize      --manifest data/processed/manifest.jsonl --out data/processed/features.parquet
uv run expr-train          --config configs/experiment_rf.yaml      # 手法A
uv run expr-train          --config configs/experiment_lstm.yaml    # 手法B
uv run expr-evaluate       --compare outputs/rf_XXXX outputs/lstm_XXXX
uv run expr-report         --run-a outputs/rf_XXXX --run-b outputs/cnn1d_XXXX   # 比較バンドル一括生成
```

各学習 run は `outputs/` の下に不変ディレクトリを作り、解決済み config・モデル
チェックポイント・指標・来歴メタデータを格納する — よって全モデルが自身のフォルダから
再現可能。

## ステータス

データ取り込み・不変化前処理・窓化/分割・古典ML/NN モデル・共通評価・比較/レポート/
可視化・任意の early stopping まで実装済み（Phase 1〜8）。残りは手法A の本物の
expert features（#20 / Phase 9）とチーム間最終比較（#21 / Phase 10）。詳細は #1 の
ロードマップを参照。
