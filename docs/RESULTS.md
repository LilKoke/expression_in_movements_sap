# Results — 歩行モーションキャプチャからの感情認識

問題設定 → データ処理 → 学習方法 → 結果 → 考察 の流れでまとめる。全モデルを
Phase 6（#7）の共通評価コードで学習・評価し、両プロトコル（inter-subject = LOSO /
intra-subject）を**同一データ・同一窓化・同一評価コード**で回した。指標の定義は
[FAQ.md](FAQ.md)、算出は [evaluate.py](../src/expr_movements/evaluate.py) /
[train.py](../src/expr_movements/train.py) を参照。

---

## 1. 問題設定

- **タスク**: 全身モーションキャプチャ（TRC）の**歩行動作から感情4クラス**
  （angry / happy / neutral / sad）を分類する。
- **データ**: 81 clips / 4 被験者（EMLA・NABA・PAIB・SALE）× 4 感情、ほぼ均衡
  （angry20 / happy20 / neutral21 / sad20）。41 マーカー × XYZ、30fps、
  フレーム長 112〜345。出典論文 **Venture et al. 2014**（本データそのもの）。
- **本番の評価軸 = inter-subject（LOSO）**: テスト被験者は学習に未登場。現実の
  「未知の人の感情を当てる」設定。出典論文の intra-subject >90% はこの設定では
  そのまま比較対象にならない（下記 intra と区別）。
- **主指標 = Macro-F1**（クラスを均等に見る）。accuracy / balanced accuracy /
  per-class F1 / 混同行列を併記。NN は再構成誤差・latent 分離度も。

### 1.1 評価指標の意味（正確な定義）

| 指標 | 定義 | 何に頑健か / なぜ使うか |
|---|---|---|
| **per-class F1** | 1クラスの precision と recall の調和平均 `2PR/(P+R)`。P=予測したうち正しい割合、R=実サンプルのうち当てた割合 | クラス別の得手不得手。本タスクの「happy/neutral が難」を見る |
| **Macro-F1**（主指標） | 各クラス F1 の**均等平均**（angry/happy/neutral/sad の F1 を足して 4 で割る） | **クラス重みを均等**にするので少数クラスを多数クラスで薄めない。難クラスの取りこぼしを公平に評価 |
| **Accuracy** | 正解数 / 全サンプル数 | 直感的だが**多数クラス支配**（不均衡に弱い）。補助指標 |
| **Balanced accuracy** | 各クラスの **recall を均等平均**（precision は見ない） | クラス不均衡に頑健。「クラスを均等に見た正答率」。ランダム期待値 = 1/クラス数 = 0.25 |
| **混同行列** | 行=true, 列=pred のカウント表 | どのクラスをどのクラスと取り違えるか（誤分類の構造） |

- **Macro-F1 と Balanced accuracy の違い**: 前者は precision と recall の両方（= 誤検出 FP も
  罰する）、後者は recall だけの平均。Macro-F1 を主指標にしているのは、難クラスの取りこぼし
  （recall）と誤検出（precision）の両面を一つの数字で見たいため。
- **window 単位 と clip 単位**: 予測は窓ごとに出すが、clip（試行ファイル）単位の結論は**窓予測の
  多数決**で確定する。両レベルで全指標を報告する。
- **mean ± std（LOSO 4-fold 横断）**: 4 被験者それぞれを test にした 4 回の平均と標準偏差。
  **std の大きさ = 被験者によるブレ = 被験者依存性**そのもの。
- **NN 専用指標**: 再構成 MSE（held-out 動きをどれだけ復元できるか、低いほど良い）、latent
  分離度 = Silhouette（高いほどクラスが分離、-1〜1）/ Davies-Bouldin（低いほど分離）。
- 詳しい解説は [FAQ.md](FAQ.md) Q5・Q6。算出は [train.py](../src/expr_movements/train.py) の
  `_scores`（scikit-learn）。

> course 要件 "extracted features VS expert features"（手作り特徴 vs 学習特徴）の
> 本比較は**手法A（expert features, #20）が揃ってから** Phase 10（#21）で行う。
> 本ドキュメント現時点の `rf` は expert features ではない（§3 と [FAQ.md](FAQ.md) 参照）。

---

## 2. データ処理（両チーム共通の前処理 contract）

Phase 1〜4 で、生 TRC を**不変化前処理済みの共通 contract** に変換する。両手法は
この同じ入力を消費する（比較の妥当性の土台）。

1. **パース**（Phase 1, #3）: TRC → `(T, 41, 3)`、ファイル名から感情・被験者ID 抽出、
   欠損座標を NaN 化。
2. **クリーニング**（Phase 2, #4）: 歩行していない先頭・末尾区間を、全マーカー平均速度
   （NaN 耐性）の onset/offset 自動検出で除去（実データ中央値 ~50 frame 除去）。
3. **不変化前処理**（Phase 3, #5）: 位置・向きを正規化して**動きそのもの**を残す。
   - 骨盤中心ローカル座標（各フレームで骨盤4点重心を原点へ）
   - yaw 補正（進行方向基準で水平回転を揃える。pitch/roll は姿勢情報なので残す）
   - **歩行速度スカラー**を別チャネルで保持（velocity↔arousal が手がかり）
   - → `data/processed/sequences.npz`、`feature_layout = "pose_local_yaw+speed"`、
     `F = 124`（41×XYZ + speed 1）。
4. **窓化 + 分割**（Phase 4, #12）: スライディング窓 length=64 / stride=16。末尾必ず
   カバー・短 clip は pad（落とさない）。予測単位は窓、clip（試行）は窓予測の**多数決**。
   正規化統計（mean/std）は **train 窓のみ**から算出。

**2つの評価プロトコル**（両チーム同一 fold を消費）:
- **inter-subject = LOSO**（本番）: テスト被験者の窓は学習に一切混ぜない。4-fold。
- **intra-subject**: 被験者内で trial-level KFold（n_splits=5）→ pool。出典論文の
  intra >90% と直接比較するため。同一試行の窓が train/test に跨らない（窓間リーク防止）。

---

## 3. 学習方法

すべて**単一の harness**（[train.py](../src/expr_movements/train.py)）で、config だけ
差し替えて学習する。fold ごとに学習し held-out で評価 → 最後に全データで refit して
artifact 保存。**両手法が同じ windowing・同じ LOSO fold・同じ評価コード**を通る。

| モデル | 内容 | 入力 |
|---|---|---|
| `rf` | RandomForest（**Approach A の暫定ベースライン**） | 正規化窓を flatten した `length×F` ベクトル |
| `cnn1d` | 1D-CNN encoder のマルチタスク NN（**自チーム本命 / Approach B**） | 正規化窓 `(length, F)` 3D テンソル |
| `lstm` | LSTM encoder のマルチタスク NN（encoder 比較 baseline） | 同上 |

> **重要（よくある誤解）**: 現状の `rf` は **expert features を使っていない**。Phase 3 の
> 正規化済み姿勢窓を flatten して RandomForest に入れただけで、velocity 統計・関節角・
> gait cadence のような人が設計した特徴量ではない（[models/classic.py](../src/expr_movements/models/classic.py)
> の docstring 参照）。本物の expert features は未実装（#20 / Phase 9）。したがって現状の
> `rf` vs `cnn1d` 比較は **「同一の前処理済み特徴を古典MLに入れる vs NNに入れる」という
> モデル比較**であり、**(a) 不変化前処理の ablation ではない**（前処理は両者同一）し、
> **(b) expert-vs-extracted features 比較でもない**。詳細は [FAQ.md](FAQ.md) Q1〜Q3。

### 3.1 モデルごとの学習のさせ方の違い（明確化）

**共通部分**（全モデルで完全に同一 = 比較の公平性の根拠）:
- 同じ `sequences.npz`、同じ窓化（length=64/stride=16）、同じ LOSO/intra fold、
  同じ train-only 標準化、同じ評価コード・多数決・mean±std。

**分岐点はモデル本体だけ**。harness（[train.py](../src/expr_movements/train.py) の
`_model_X`）が `BaseClassifier.consumes` フラグで入力形状を出し分ける:

| | `rf`（Approach A 暫定） | `cnn1d` / `lstm`（Approach B = NN） |
|---|---|---|
| **入力形状** | 窓を flatten した 2D 表 `(n, length×F)` | 窓 3D テンソル `(n, length, F)` + マスク |
| **学習対象** | RandomForest（決定木アンサンブル） | encoder + decoder + 分類ヘッド（誤差逆伝播） |
| **損失** | なし（情報利得で木を分割） | `α·再構成MSE + β·分類CrossEntropy`（**マルチタスク**） |
| **エポック** | 概念なし（一括 fit） | 反復学習（既定 epochs=40, batch_size=32, Adam lr=1e-3） |
| **時系列の扱い** | フレーム順を平坦化＝**時間構造を陽に使わない** | 1D-CNN / RNN が**時間方向の局所/系列構造を学習** |
| **速度チャネル** | flatten ベクトルの一部 | pooled latent に concat（明示的に分離して扱う） |
| **追加で出る表現** | なし | **latent z**（PCA・分離度の対象）+ 再構成 |
| **early stopping** | 非該当（epoch なし → 自動スキップ） | opt-in 可（§5。validation 被験者で best epoch 復元） |

**マルチタスク NN（cnn1d / lstm）の構造**:
```
入力窓 → [encoder] → latent z ─┬→ [decoder] → 再構成   (loss: reconstruction MSE)
                                └→ [分類ヘッド] → 感情   (loss: cross-entropy)
total = α·reconstruction + β·classification
```
- 速度スカラーは pooled latent に concat。latent が「動きを保持（再構成可）」かつ「分類に
  効く」表現になるよう設計（→ PCA 可視化が意味を持つ）。
- `cnn1d` と `lstm` の違いは **encoder のみ**（1D 畳み込み vs 再帰）。decoder・分類ヘッド・
  損失・学習手順は共通。本スケール（~80 試行）では過学習耐性で 1D-CNN を本命に選定。
- `rf` と NN の本質的な違いは「**時間構造を陽にモデル化するか**」と「**表現を学習するか
  （latent を持つか）**」の2点。

- config: `configs/experiment_{rf,cnn1d,lstm}.yaml`（LOSO）/ `..._intra.yaml`（intra）。
- NN は小規模・少エポック（latent_dim=32, epochs=40）。ハイパラ最適化は未実施。
- 再現: `expr-train --config <cfg>` → `expr-evaluate --run outputs/<name>_<hash>`。
  `outputs/` は `.gitignore` 対象（成果物は config から再現可能）。
- Phase 8 で **early stopping**（opt-in）経路も追加（§5）。

---

## 4. 結果

### 4.1 主結果: 正答率と Macro-F1（mean ± std, fold 横断）

| モデル | プロトコル | clip Acc | window Acc | clip Macro-F1 |
|---|---|---|---|---|
| rf    | **inter (LOSO)** | 0.705 ± 0.054 | 0.769 ± 0.025 | 0.631 ± 0.038 |
| cnn1d | **inter (LOSO)** | 0.630 ± 0.241 | 0.677 ± 0.261 | 0.555 ± 0.287 |
| lstm  | **inter (LOSO)** | 0.601 ± 0.270 | 0.614 ± 0.275 | 0.548 ± 0.319 |
| rf    | intra            | 1.000 ± 0.000 | 0.971 ± 0.019 | 0.950 ± 0.100 |
| cnn1d | intra            | 0.976 ± 0.030 | 0.971 ± 0.039 | 0.921 ± 0.093 |
| lstm  | intra            | 1.000 ± 0.000 | 0.984 ± 0.009 | 0.950 ± 0.100 |

### 4.2 inter vs intra のギャップ（被験者依存性）

| モデル | inter clip Acc | intra clip Acc | ギャップ |
|---|---|---|---|
| rf    | 0.705 | 1.000 | **+0.30** |
| cnn1d | 0.630 | 0.976 | **+0.35** |
| lstm  | 0.601 | 1.000 | **+0.40** |

- intra-subject は全モデル **~98〜100%**（出典論文 intra >90% と一致）。
- inter-subject(LOSO) は **60〜70%** に低下、NN は **std が 0.24〜0.32** と非常に大きい。
- この **intra ≫ inter のギャップ自体が「未知の人への汎化の難しさ＝被験者依存性」** の知見。

### 4.3 被験者別（LOSO の per-fold clip 正答率）

| テスト被験者 | rf | cnn1d | lstm |
|---|---|---|---|
| EMLA | 0.750 | 0.900 | 0.750 |
| NABA | 0.750 | 0.750 | 0.450 |
| PAIB | 0.619 | 0.619 | 0.952 |
| SALE | 0.700 | 0.250 | 0.250 |

- 被験者によって **0.25〜0.95** と激しく振れる。NN は特に SALE で崩れる一方、RF は
  全 fold 0.6〜0.75 で**安定**。少データ（~80 試行 / 4 被験者）では NN の分散が大きい。

### 4.4 per-class F1（LOSO, clip）と混同の傾向

| モデル | angry | happy | neutral | sad |
|---|---|---|---|---|
| rf    | 0.688 | 0.472 | 0.394 | 0.972 |
| cnn1d | 0.583 | 0.366 | 0.520 | 0.750 |
| lstm  | 0.476 | 0.649 | 0.455 | 0.611 |

- **sad は全モデルで高精度**、**happy / neutral が難クラス**で相互に誤分類されやすい。
  出典論文の「Joy（happy）が難クラス」と方向一致。
- 混同行列（rf, LOSO, clip; 行=true, 列=pred, 順 angry/happy/neutral/sad）:

  |  | angry | happy | neutral | sad |
  |---|---|---|---|---|
  | angry   | 13 | 7  | 0  | 0  |
  | happy   | 0  | 15 | 5  | 0  |
  | neutral | 0  | 11 | 10 | 0  |
  | sad     | 0  | 0  | 1  | 19 |

  → neutral→happy の誤りが目立つ（11/21）。sad はほぼ完全に分離。

### 4.5 マルチタスク NN の追加指標（held-out）

| モデル | プロトコル | 再構成 MSE | Silhouette ↑ | Davies-Bouldin ↓ |
|---|---|---|---|---|
| cnn1d | inter (LOSO) | 13.16 ± 20.65 | 0.577 ± 0.131 | 0.783 ± 0.278 |
| lstm  | inter (LOSO) | 13.34 ± 20.85 | 0.448 ± 0.204 | 1.239 ± 0.676 |
| cnn1d | intra        |  0.57 ± 0.14  | 0.482 ± 0.059 | 0.823 ± 0.129 |
| lstm  | intra        |  0.55 ± 0.15  | 0.546 ± 0.031 | 0.713 ± 0.083 |

- **latent 分離度は LOSO で cnn1d > lstm**（Silhouette 高・DB 低）→ PCA 可視化の土台として
  1D-CNN の latent の方が綺麗に感情分離。本命を CNN にした判断と整合。
- 再構成 MSE は LOSO で大きく std も大（未知被験者の動きは再構成しづらい）。intra では桁が小さい。

---

## 5. 結果（補遺）: early stopping + validation-best モデル（Phase 8）

固定 epochs（baseline, validation なし）に対し、各 LOSO fold の train 側から**未知の1被験者を
validation に切り出して early stopping**（best epoch 復元）する経路を opt-in 追加。validation
分割は `group_subject`（早期停止シグナルも未知の人＝test と同性質、被験者リークなし）。

### 5.1 素朴な early stopping は単独では精度を下げる

| 構成 | clip Acc | clip Macro-F1 |
|---|---|---|
| cnn1d 固定40（baseline） | 0.630 ± 0.241 | 0.555 ± 0.287 |
| cnn1d ES のみ（patience30） | 0.542 ± 0.232 | 0.468 ± 0.281 |

理由: **被験者が4人**しかいないため validation に1人取ると **train が3人→2人**に減り、
モデルが弱くなる損失が ES の恩恵を上回る。

### 5.2 容量削減 + 正則化 + head BatchNorm を組み合わせて baseline 超え

**単独施策はどれも baseline 以下**（容量減のみ 0.613 / head BN のみ 0.592 /
weight_decay+dropout のみ 0.567）。**全部を組み合わせて初めて baseline を超えた**:

| 構成 (cnn1d) | clip Acc | clip Macro-F1 | epochs |
|---|---|---|---|
| 固定40（baseline） | 0.630 ± 0.241 | 0.555 ± 0.287 | 40 |
| **ES + reg combo** | **0.675 ± 0.297** | **0.606 ± 0.363** | 69 ± 12 |

reg combo = latent 32→16 / hidden 64→32 / dropout 0.2→0.4 / weight_decay 1e-3 /
head BatchNorm / patience 30。config: `experiment_cnn1d_es.yaml`。被験者別（cnn1d）:
固定 `{EMLA .90, NABA .75, PAIB .62, SALE .25}` → ES+reg `{EMLA .90, NABA .55, PAIB 1.00,
SALE .25}`。PAIB が大きく改善する一方 NABA は低下、SALE は両者 0.25 のまま（最難被験者）。

### 5.3 同じレシピは LSTM には効かない（転移しない）

| 構成 (lstm) | clip Acc | clip Macro-F1 |
|---|---|---|
| 固定40（baseline） | 0.601 ± 0.270 | 0.548 ± 0.319 |
| ES + reg combo（CNN と同設定） | 0.417 ± 0.177 | 0.316 ± 0.222 |

→ ハイパラはアーキ依存。「ES を入れれば必ず上がる」わけではないことの実証。

---

## 6. 考察

1. **本番 = LOSO では現状 RF が NN をやや上回る**（clip Acc 0.705 vs cnn1d 0.630）。
   ~80 試行という少データで NN が過学習・高分散。ただし**この RF は expert features では
   なく正規化窓 flatten**なので、これは「古典ML vs NN のモデル比較」であって course 要件の
   expert-vs-extracted features 比較ではない（§3 / [FAQ.md](FAQ.md)）。
2. **intra では全モデルが論文同等の ~98〜100%**。intra ≫ inter のギャップ（+0.30〜0.40）が
   被験者依存性の定量的知見。本タスクの難しさは**モデルよりデータ（4被験者・少試行）**に起因。
3. **happy/neutral が難クラス、sad が易クラス**は出典論文と一致。neutral→happy の混同が支配的。
4. **latent 分離度は cnn1d が最良** → 精度では RF だが、**解釈性・PCA 可視化では NN に優位性**。
5. **early stopping は単体では本データで不利**、強い正則化と組み合わせて CNN では baseline 超え
   （0.630→0.675）だが std 大・アーキ依存。少データの不安定性は解消しきれていない。

### 残課題（ロードマップ）
- **本物の expert features の実装**（#20 / Phase 9）: 手作り特徴量の設計・計算（**特徴量の
  中身は相手チームに一任**）。これが揃って初めて course 要件の比較になる。FYI として出典論文
  Venture 2014 は velocity / 体幹傾き / 頭部向きを手がかりにしている。
- **チーム間最終比較**（#21 / Phase 10）: 手法A(expert)・手法B(NN)を同一 LOSO で head-to-head、
  latent PCA vs feature PCA、解釈性 vs 精度トレードオフの考察。
- （任意）**不変化前処理の ablation**: yaw/骨盤ローカル/速度チャネルの有無で精度がどう変わるか
  （現状未実施。やるなら variant config を追加）。

---

## 付録: この結果を生成した run

| run dir | config | プロトコル |
|---|---|---|
| `outputs/rf_1f7822cf/`        | `experiment_rf.yaml`          | LOSO |
| `outputs/cnn1d_4359693b/`     | `experiment_cnn1d.yaml`       | LOSO |
| `outputs/lstm_566bd9d1/`      | `experiment_lstm.yaml`        | LOSO |
| `outputs/rf_intra_47958e45/`  | `experiment_rf_intra.yaml`    | intra |
| `outputs/cnn1d_intra_f52b073c/` | `experiment_cnn1d_intra.yaml` | intra |
| `outputs/lstm_intra_78d027f8/` | `experiment_lstm_intra.yaml`  | intra |
| `outputs/cnn1d_es_*/`         | `experiment_cnn1d_es.yaml`    | LOSO + early stopping |
| `outputs/lstm_es_*/`          | `experiment_lstm_es.yaml`     | LOSO + early stopping |

（run dir 名のハッシュは config 内容に依存。再学習で同一 config なら同じハッシュになる。
metrics の全数値は各 run の `metrics.json`、または `expr-evaluate --run <dir> --json` で参照。
比較バンドルは `expr-report --run-a <A> --run-b <B> --run-b-intra <B_intra>` で再生成。）
