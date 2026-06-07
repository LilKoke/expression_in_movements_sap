# アーキテクチャ

本プロジェクトのコードアーキテクチャ — ファイル/依存構成、スクリプトの粒度、そして
モジュール性・config 駆動・再現性を保つための規約をまとめる。これは issue #2
（実装環境の整備）の成果物であり、モデリング作業自体は #1 にぶら下がる各 Phase issue に
分割されている。

## 目的

表情豊かな歩行動作（モーションキャプチャ **TRC**、41 マーカー × XYZ 時系列）を 4 つの
感情クラスに分類し、**2つのモデリング手法**を**同一の被験者グループ分割**の上で比較する:

- **手法A** — expert / 手作り特徴量 + 古典ML（RandomForest / SVM）
- **手法B** — 生の姿勢系列に対するニューラルネット（LSTM / 1D-CNN）

感情ラベルはファイル名の 3 文字コードに由来: `TRE=sad`, `COE=angry`, `NEE=neutral`,
`JOE=happy`。

> **手法A の現状に関する注記**: 現状の `random_forest` run は、正規化済み姿勢窓を flatten
> しただけの**ベースライン**であり、**まだ手作り expert features を使っていない**
> （`expr-featurize` / `features.build_feature_table` は未実装）。本物の expert feature 作業は
> #20（ロードマップ Phase 9）、チーム間の最終比較は #21（Phase 10）。

**関連ドキュメント**: 数値結果とモデルごとの学習方法の違いは [RESULTS.md](RESULTS.md)、
指標の定義（Macro-F1 と F1、accuracy と balanced accuracy）と現状の比較が実際に何を測って
いるかは [FAQ.md](FAQ.md)、gitignore された `data/` を持たない・GPU の無いメンバー向けの
Google Drive データ取得手順は [DATA_SETUP.md](DATA_SETUP.md) を参照。

## 設計原則

1. **モジュール化・差し替え可能**。各モデルは sklearn 風の `fit/predict` インターフェースを
   1つ実装し、registry を通じて config 名で選択される。よって手法A ↔ B は config の 1 行
   変更で済み、同じ学習/評価 harness が両方を駆動する。
2. **config 駆動**。全ハイパーパラメータは `configs/*.yaml` にあり、起動時に Pydantic で
   検証される。コード中にハードコードされたパラメータは無い。
3. **再現可能 — パラメータとモデル出力を一緒に**。各 run は**解決済み** config・モデル
   チェックポイント・指標・来歴メタデータを、config のハッシュをキーにした 1 つの不変
   ディレクトリに書き出す。
4. **処理済みデータは永続化し、再計算しない**。raw → interim → processed と進み、processed
   データセットはディスク（Parquet / npz / JSONL）に書いて再読み込みする。
5. **スクリプトは薄く、ライブラリは厚く**。CLI エントリポイントは引数解析と委譲だけ。
   ロジックはすべて `src/expr_movements/` に置く。

## リポジトリ構成

```
configs/                  実験 YAML（バージョン管理対象）
  experiment_rf.yaml         手法A
  experiment_lstm.yaml       手法B
data/                     GITIGNORE 対象（大きい / 再生成可能）
  raw/                       元の .trc — 不変
  interim/                   パース/クリーニング済みの clip 単位配列
  processed/                 manifest.jsonl + features.parquet + sequences.npz
outputs/                  GITIGNORE 対象 — run 単位の成果物（後述）
src/expr_movements/
  config.py                 Pydantic スキーマ; load_experiment()
  run.py                    run ディレクトリ + config↔成果物の紐付け
  splits.py                 被験者グループ分割（train/test）
  train.py / evaluate.py    オーケストレーション + A vs B 比較
  viz.py                    Phase 7 の図（latent PCA / 混同行列 / 棒グラフ）
  data/
    trc.py                  TRC パース + ファイル名メタデータ
    dataset.py              manifest (jsonl) + 系列テンソル (npz)
  features/__init__.py      expert features -> feature table (parquet)
  models/
    base.py                 BaseClassifier（sklearn インターフェース）
    registry.py             register() / build_model()
    classic.py              手法A: random_forest, svm
    neural.py               手法B: lstm, cnn1d
  cli/                      薄いエントリポイント（expr-parse/-build-dataset/
                            -featurize/-train/-evaluate/-report）
tests/
docs/ARCHITECTURE.md
```

`notebooks/` ディレクトリは存在しない: 比較/レポートの図は手作業の notebook ではなく
**再現可能なスクリプト（`expr-report`）**で生成する（後述のアンチパターン参照）。

## データフロー

```
data/raw/*.trc
   │  expr-parse        (data/trc.py)
   ▼
data/interim/*          clip 単位配列 + ファイル名メタデータ
   │  expr-build-dataset (data/dataset.py)
   ▼
data/processed/manifest.jsonl          人が読める clip/ラベル索引
data/processed/sequences.npz           手法B の入力  ──┐
   │  expr-featurize    (features/)                     │
   ▼                                                    │
data/processed/features.parquet        手法A の入力  ──┤
                                                        ▼
                              expr-train --config …  (train.py)
                                build_model(name, **params) を
                                被験者グループ分割 (splits.py) 上で実行
                                            │
                                            ▼
                              outputs/<name>_<config-hash>/
                                config.yaml  model.*  metrics.json  metadata.json
                                            │
                                            ▼
                              expr-evaluate --compare A B  (evaluate.py)
                                            │
                                            ▼
                              expr-report --run-a A --run-b B  (cli/report.py)
                                evaluate.py + viz.py → outputs/report/
                                comparison.md · protocol_comparison.md · figs/*.png
```

## データセット永続化（Phase 2, #4）

`expr-build-dataset`（`data/dataset.py`）はパース済み interim clip を `data/processed/` の
下の 2 つの永続成果物に変換する。これによりモデリングが生 TRC を再パースすることはない:

- **`manifest.jsonl`** — clip ごとに 1 行の JSON: interim パス、被験者、感情（コード + 名前）、
  テイク、生フレーム数、採用した `[trim_start, trim_stop)` 窓とその長さ。人が読める索引で、
  系列ビルドが消費する。
- **`sequences.npz`** — 手法B の入力（後述の*長さの扱い*参照）。

**トリミング（動作 onset/offset）**。各 clip には最初の一歩の前と最後の一歩の後に立ち
止まり/アイドルのフレームがある。`DataConfig.detect_onset`（既定 ON）では、活動歩行窓を
**マーカー速度**から求める: フレームごとにマーカー間のフレーム間変位平均（NaN 耐性、よって
オクルージョンされたマーカーは無視）を取り、それが clip のピークの `onset_speed_frac` を
超えたフレームを「動いている」とみなし、`onset_min_run` 連続で動いているフレームの最初/最後を
onset/offset とする。`trim_start_frames` / `trim_end_frames` はその窓の*内側*に追加の固定
マージンとして適用される。clip が空になるまでトリムされることはない（マージンで空になるなら
スキップ）。（同梱データではこれにより clip あたり中央値 ~50 の先頭/末尾静止フレームが除去
される。）検出パラメータは `DataConfig` にあり、ポリシーは config 駆動で解決済み config に
記録される。

**可変長の扱い（決定事項）**。clip の長さはばらつく（同梱データで約 73〜372 フレーム）。
固定フレーム数をデータセットに焼き込むのではなく、**正準（canonical）**ストアを*可変長*とする:
`sequences` は clip 単位の `(T_i, 41*3)` float32 配列（onset トリム済み）の object 配列で、
`lengths`・`labels`・`emotion_codes`・`subjects`・`clips`・`marker_names` を伴う。

理由: モデリングのフレーム数はスイープしたい*学習時*のハイパーパラメータ（さらにスライディング
窓による拡張もサポートしたい）。可変長配列から固定窓/長さを切り出すのは数十 MB のメモリ上
numpy 操作で、ミリ秒オーダー・ディスク読み込みより安い。よって `N` をビルド時に凍結する速度的
理由は無く、前もってゼロパディングすると全消費側が偽フレームで学習しないためのマスクを持つ
羽目になる。真の長さの系列を永続化することでその決定をデータセットの外に出す。

固定テンソルが欲しい呼び出し側のために、`--target-frames N` は密な `sequences_dense`
`(n_clips, N, 41*3)`（ゼロパディング / `N` で切り詰め）と、実フレームを示す bool `mask` を
追加で書き出す。これは正準ストアからいつでも再現できるので、`N` の凍結は「再実行」であって
不可逆な選択ではない。

**CSV 出力（#1）**。課題ブリーフは `(座標, クラス)` データセットを「CSV で」と求めている。
本プロジェクトはモデリング成果物を JSONL + npz として永続化し（npz は密/不規則な数値コンテナ
として自然で、clip あたり ~10⁶ の float を文字列化せずに済む）、CSV は*エクスポート*の関心事
として扱う。`manifest.jsonl` のフラット CSV（clip 1 行・ラベルとフレーム窓）は
`pandas.read_json(lines=True).to_csv(...)` で自明。long 形式の
`(clip, frame, marker, x, y, z, emotion)` CSV は提出物が必要なときに `sequences.npz` から
導出できる。作業ストアをバイナリに保ち CSV を必要時に生成することで、冗長で嵩張る 3 つ目の
コピーを持たずに済む。

## 不変化前処理（Phase 3, #5）

分類器は被験者が*どこで*歩いたか・*どちら向き*だったかではなく、*どう*動くかを手がかりに
すべき。そこで clip のトリム済み姿勢系列を `sequences.npz` に格納する前に、
`features/preprocess.py`（`normalize_sequence`）が**位置・向きに対して不変**にしつつ、
**歩行速度は明示的なチャネルとして意図的に残す**（速度は arousal を反映する強い感情手がかり。
本データの出典論文 Venture et al. 2014）。これは両モデリングチームが消費する**共通表現**で
ある（#1 ロードマップ v2 では Phase 3 はここで止まる — 派生 expert features はこのリポジトリ
では計算せず、それは別チームの手法A の作業）。

clip ごとに、順に:

1. **骨盤中心ローカル座標**。各フレームを、骨盤 4 マーカー（`LFWT/RFWT/LBWT/RBWT`）の重心が
   原点に来るよう平行移動 — 絶対歩行位置と部屋を横切る緩やかなドリフトを除去する。マーカーは
   被験者ごとの接頭辞（`NABA_LFWT`）を持ち、最後の `_` 以降の素のトークンで照合する。重心は
   NaN 耐性なので、オクルージョンされた骨盤マーカーが汚染しない。骨盤マーカーが*1つも無い*
   フレームは落とさず未平行移動のままにする。
2. **yaw 補正**（`yaw_align`、既定 ON）。各フレームを鉛直軸（`up_axis`、ここでは Y）まわりに
   回転させ、骨盤の左→右軸が固定方向を向くようにする — 向き（heading）を除去。**pitch と roll
   はそのまま残す**（体幹の傾き・頭部の向き）。これらは姿勢/感情を担うため。水平の向きだけを
   正規化で消す。
3. **速度チャネル**（`keep_speed`、既定 ON）。フレームごとの体の速度 — **ワールド**座標での
   骨盤重心の変位の大きさ（中心化*前*に計算。中心化すると 0 になる）— を末尾列として付加。
   よって格納される系列は `(T, 41*3 + 1)`: 123 の姿勢列の後に 1 の速度列。

4 つの挙動はすべて `DataConfig` のフィールド（`normalize`・`pelvis_markers`・`yaw_align`・
`keep_speed`・`up_axis`）であり、ポリシーは config 駆動で解決済み config に記録される。
`normalize=False` なら生の flatten された `(T, 41*3)` ワールド座標がそのまま格納される
（骨盤マーカーを持たない合成データセットのテストで使用）。`sequences.npz` は列の分割を
`feature_layout`（`"pose_local_yaw+speed"`）・`has_speed_channel`・`n_markers` で記録するので、
消費側は姿勢 vs 速度を推測ではなくメタデータで分割できる。

## run ディレクトリ（再現性の契約）

チェックポイントは、それを生成した正確な config を隣に置かずに保存されることはない:

```
outputs/rf_a1b2c3d4/
  config.yaml        # 解決済み ExperimentConfig
  metadata.json      # git commit, seed, データハッシュ, タイムスタンプ, 指標サマリ
  model.joblib       # (A) または model.pt (B)
  metrics.json       # accuracy, per-class F1, 混同行列
  predictions.jsonl
```

ディレクトリ名は `sha256(解決済み config)[:8]` を埋め込むので、同一 config は同じ
ディレクトリを再利用し、設定の再実行が検出可能になる。

## config 管理

Pydantic v2 で検証する素の YAML（`config.py`）。Hydra ではなくこれを選んだのは、プロジェクトが
小さく config グループ合成や CLI 乗っ取りを必要としないため — Pydantic はスキーマ検証
（YAML のタイポは `extra="forbid"` で明示的に失敗）と run ディレクトリ書き出し用の綺麗な
(逆)シリアライズを足す。スイープ実行が苦痛になったら Hydra を再検討する。

## データリーク回避（重要）

ランダム / フレーム単位の分割は同じ被験者を train と test の両方に入れ、精度を水増しする。
すべての分割は**被験者グループ単位**（`splits.py` の `GroupKFold` / `GroupShuffleSplit` /
leave-one-subject-out）であり、手法A と B は**同一の**分割を使うので比較が妥当になる。

## 比較 & レポート（Phase 7, #13）

2 つの手法は**同一の成果物契約の上で head-to-head 比較**される — どのチームが作った run でも
`metrics.json` を持つ `outputs/<name>_<hash>/` ディレクトリなので、比較は両方を同じ
`evaluate_run` で読む:

- **`evaluate.compare_runs(A, B)`** — A（expert features）vs B（NN）を*同一*の LOSO 分割・
  指標で比較。ラベル空間や分割プロトコルが異なる run は拒否するので、比較が黙って非互換な
  run を混ぜることはない。
- **`evaluate.compare_protocols(intra, loso)`** — 1 手法の intra-subject vs
  inter-subject(LOSO) Macro-F1 を出し、**ギャップ = 被験者依存性**を浮かび上がらせる（intra は
  Venture 2014 の >90% と直接比較可能、LOSO が本番タスク）。
- **`viz.py`** — 図: latent PCA（感情=色・被験者=マーカー）、混同行列ヒートマップ、A vs B 指標
  棒グラフ。matplotlib は遅延 import され、任意の `viz` extra（`uv sync --extra viz`）の背後に
  あるので、コアの評価面はこれを必要としない。

**`expr-report`**（`cli/report.py`）は run を一括成果物に繋ぐ薄いドライバで、1 つの再現可能な
コマンドで動く — notebook なし:

```
expr-report --run-a outputs/<rf> --run-b outputs/<cnn1d> \
            --run-b-intra outputs/<cnn1d_intra>
  → outputs/report/{comparison.md, protocol_comparison.md, figs/*.png}
```

latent の出所について: fold ごとの held-out モデルは永続化されない（保存されるのは全データの
最終モデルのみ）ため、PCA はその保存済みモデルで全 window を encode する — これは学習済み
latent の*可視化*であり、`metrics.json` の held-out 分離度**数値**が正直な汎化シグナルとして
残る。両者は併せて読む。

## early stopping（任意, Phase 8, #1）

既定では各 NN は **validation set 無し**で固定 `model.params.epochs` だけ学習する — これが
ベースライン経路で、手を加えない。実験 YAML に `validation` ブロック（`ValidationConfig`）を
足すと **early stopping** が有効になる:

```yaml
validation:
  enabled: true
  strategy: group_subject   # train 側から被験者を丸ごと 1 人 hold out
  val_size: 0.34            # LOSO-train の 3 人中 ~1 人
  patience: 10
  monitor: macro_f1         # accuracy | loss もサポート
  restore_best: true
```

各 fold で、harness は**その fold の train 側から validation set を切り出し**
（`splits.nested_validation_split`）、毎エポック validation 指標を見ながら NN を学習し、最良
エポックの重みを保持する。`strategy: group_subject`（既定）では validation set は**丸ごと
hold out された 1 被験者**なので、test fold と同様に early stopping のシグナルが*未知の人*から
来て、inter-subject タスクに対してモデル選択が正直に保たれる（停止基準への同一被験者リーク
なし）。`stratified_clip` は緩い「既知の人」版で、比較用に残してある。

重要な境界:

- **NN のみ**。古典ML（RandomForest/SVM）にはエポックが無いので、harness はこのブロックを
  無視して通常通り学習し、ES が適用されなかったことを記録する。
- **保存される成果物は影響を受けない**。early stopping は報告される held-out 指標を生む
  *fold ごと*のモデルを形作る。配布チェックポイント（`model.joblib`）は依然として全データで
  固定エポック数 refit される。
- **自己記述的な run**。`metrics.json` に `early_stopping` ブロック（平均実行エポック数・
  monitor・使用 fold）が加わり、各 fold には hold out した validation 被験者を示す
  `validation` エントリが入る。

これは固定エポック経路と*並行して*動くので、両者は直接比較できる: `experiment_cnn1d.yaml`
（固定 40 エポック）vs `experiment_cnn1d_es.yaml`（early stopping）。結果は `docs/RESULTS.md`。

## この構成が避けるアンチパターン

- 神スクリプト — ロジックは `src/` に、スクリプトは薄く。
- ハードコードされたパス/パラメータ — すべて `configs/` に。
- notebook 駆動の非再現性 — notebook は存在しない。比較と可視化は `expr-report`
  （`cli/report.py` が `evaluate.py` + `viz.py` を呼ぶ）を通すので、全図がコミット済み
  スクリプトから再生成される。
- 毎 run で処理済みデータを再計算 — 処理済み成果物は永続化される。
- 被験者リーク — グループ分割、手法間で共有。
- 固定されていない環境 — `uv.lock` をコミット済み。

## ステータス

データ取り込み（parse / build-dataset）・不変化前処理・窓化と分割・古典ML と NN マルチタスク
モデル・共通評価コード・比較/レポート/可視化・任意の early stopping まで実装済み（Phase 1〜8）。
残りは手法A の本物の expert features（#20 / Phase 9）とチーム間最終比較（#21 / Phase 10）で、
進捗は #1 のロードマップを参照。
