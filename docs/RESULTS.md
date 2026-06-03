# Results — Phase 6 evaluation

Phase 6 (#7) の共通評価コードで全モデルを学習・評価した結果のまとめ。両プロトコル
（inter-subject = LOSO / intra-subject）を同一データ・同一窓化・同一評価コードで回し、
**正答率（accuracy）を中心に** Macro-F1・per-class F1・混同行列・（NN は）再構成誤差・
latent 分離度を記録した。指標の定義と算出は [evaluate.py](../src/expr_movements/evaluate.py) /
[train.py](../src/expr_movements/train.py) を参照。

## 実験設定

- **データ**: 81 clips / 4 被験者（EMLA・NABA・PAIB・SALE）× 4 感情（angry/happy/neutral/sad）、
  ほぼ均衡。骨盤中心ローカル座標 + yaw 補正 + 速度チャネル（Phase 3 の不変化前処理済み）。
- **窓化**: length=64, stride=16。予測単位は窓、clip（試行）単位は窓予測の**多数決**で確定。
- **モデル**:
  - `rf` — expert features + RandomForest（別チーム想定の代表 / Approach A）
  - `cnn1d` — 1D-CNN encoder のマルチタスク NN（自チーム本命 / Approach B）
  - `lstm` — LSTM encoder のマルチタスク NN（encoder 比較用 baseline）
- **プロトコル**:
  - **inter-subject = LOSO** — テスト被験者は学習に未登場。4 fold。**本タスク本番**。
  - **intra-subject** — 被験者内で trial-level KFold（n_splits=5）、pool。出典論文
    Venture et al. 2014 の intra-subject >90% と直接比較するため。
- config: `configs/experiment_{rf,cnn1d,lstm}.yaml`（LOSO）/ `..._intra.yaml`（intra）。
- 再現: `expr-train --config <cfg>` → `expr-evaluate --run outputs/<name>_<hash>`。
  `outputs/` は `.gitignore` 対象（成果物は config から再現可能）。

> 注: NN は小規模・少エポック設定（latent_dim=32, epochs=40）での結果。ハイパラ最適化は未実施。

## 正答率（accuracy）— 主結果

clip 単位（多数決後）と窓単位、fold 横断の **mean ± std**。

| モデル | プロトコル | clip Acc | window Acc | clip Macro-F1 |
|---|---|---|---|---|
| rf    | **inter (LOSO)** | 0.705 ± 0.054 | 0.769 ± 0.025 | 0.631 ± 0.038 |
| cnn1d | **inter (LOSO)** | 0.630 ± 0.241 | 0.677 ± 0.261 | 0.555 ± 0.287 |
| lstm  | **inter (LOSO)** | 0.601 ± 0.270 | 0.614 ± 0.275 | 0.548 ± 0.319 |
| rf    | intra            | 1.000 ± 0.000 | 0.971 ± 0.019 | 0.950 ± 0.100 |
| cnn1d | intra            | 0.976 ± 0.030 | 0.971 ± 0.039 | 0.921 ± 0.093 |
| lstm  | intra            | 1.000 ± 0.000 | 0.984 ± 0.009 | 0.950 ± 0.100 |

### inter vs intra のギャップ（被験者依存性）

| モデル | inter clip Acc | intra clip Acc | ギャップ |
|---|---|---|---|
| rf    | 0.705 | 1.000 | **+0.30** |
| cnn1d | 0.630 | 0.976 | **+0.35** |
| lstm  | 0.601 | 1.000 | **+0.40** |

- intra-subject は全モデル **~98〜100%**。出典論文 Venture 2014 の **intra >90%** と一致。
- inter-subject(LOSO) は **60〜70%** に低下し、NN は **std が 0.24〜0.32** と非常に大きい。
- この **intra ≫ inter のギャップ自体が「未知の人への汎化の難しさ＝被験者依存性」** の知見。
  本タスク本番は LOSO（inter）側であり、論文の >90% はそのまま比較対象にならない。

## 被験者別（LOSO の per-fold clip 正答率）

被験者依存性が最も鮮明に出る数字。各 fold = その被験者をテストに回したときの正答率。

| テスト被験者 | rf | cnn1d | lstm |
|---|---|---|---|
| EMLA | 0.750 | 0.900 | 0.750 |
| NABA | 0.750 | 0.750 | 0.450 |
| PAIB | 0.619 | 0.619 | 0.952 |
| SALE | 0.700 | 0.250 | 0.250 |

- 被験者によって正答率が **0.25〜0.95** と激しく振れる。NN は特に SALE で大きく崩れる
  （cnn1d 0.250 / lstm 0.250）一方、RF は全 fold 0.6〜0.75 で**安定**。
- 少データ（~80 試行 / 4 被験者）では NN の分散が大きく、平均では RF が NN を上回る。

## per-class F1（LOSO, clip）と混同の傾向

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

## マルチタスク NN の追加指標（held-out）

再構成誤差（overall MSE）と latent 分離度（Silhouette↑ / Davies-Bouldin↓）。
classic ML（rf）には該当指標なし。

| モデル | プロトコル | 再構成 MSE | Silhouette ↑ | Davies-Bouldin ↓ |
|---|---|---|---|---|
| cnn1d | inter (LOSO) | 13.16 ± 20.65 | 0.577 ± 0.131 | 0.783 ± 0.278 |
| lstm  | inter (LOSO) | 13.34 ± 20.85 | 0.448 ± 0.204 | 1.239 ± 0.676 |
| cnn1d | intra        |  0.57 ± 0.14  | 0.482 ± 0.059 | 0.823 ± 0.129 |
| lstm  | intra        |  0.55 ± 0.15  | 0.546 ± 0.031 | 0.713 ± 0.083 |

- **latent 分離度は LOSO で cnn1d > lstm**（Silhouette 高・DB 低）。PCA 可視化（Phase 7）の
  土台として **1D-CNN の latent の方が綺麗に感情分離**できており、本命を CNN にした判断と整合。
- 再構成 MSE は LOSO で非常に大きく std も大（未知被験者の動きは再構成しづらい）。intra では
  桁が小さい（同一被験者の動きは再構成しやすい）。

## まとめ・含意

1. **本タスク本番 = LOSO では現状 RF（expert features）が NN をやや上回る**
   （clip Acc 0.705 vs cnn1d 0.630）。~80 試行という少データで NN が過学習・高分散。
2. **intra では全モデルが論文同等の ~98〜100%**。intra ≫ inter のギャップ（+0.30〜0.40）が
   被験者依存性の定量的な知見。
3. **happy/neutral が難クラス、sad が易クラス**は論文と一致。
4. **latent 分離度は cnn1d が最良** → 解釈性・PCA 可視化の観点では NN に優位性。

→ Phase 7 では、この LOSO 数字で **NN系 vs expert features系** を head-to-head 比較し、
精度（RF 優位）と解釈性・latent 可視化（CNN 優位）のトレードオフを言語化する。

## Phase 8: early stopping + validation-best モデル

固定 epochs（baseline, validation なし）に対し、各 LOSO fold の train 側から
**未知の1被験者を validation に切り出して early stopping**（best epoch の重みを復元）
する経路を追加し、同一 LOSO で head-to-head 比較した。validation 分割は
`group_subject`（早期停止シグナルも未知の人＝test と同性質、被験者リークなし）。

### 素朴な early stopping は単独では精度を下げる

最初に「ES だけ（正則化なし）」を試すと **LOSO 精度が悪化**した：

| 構成 | clip Acc | clip Macro-F1 |
|---|---|---|
| cnn1d 固定40（baseline） | 0.630 ± 0.241 | 0.555 ± 0.287 |
| cnn1d ES のみ（patience30） | 0.542 ± 0.232 | 0.468 ± 0.281 |

理由は明快で、**被験者が4人しかいない**ため validation に1人取ると **train が3人→2人**に
減り、モデルが弱くなる損失が ES の恩恵を上回る。

### 改善: 容量削減 + 正則化 + head BatchNorm を組み合わせて baseline 超え

@LilKoke 提案（パラメータ数・patience・batch normalization）を受けてスイープした。
**単独施策はどれも baseline 以下**（容量減のみ 0.613 / head BN のみ 0.592 /
weight_decay+dropout のみ 0.567）。**全部を組み合わせて初めて baseline を超えた**：

| 構成 (cnn1d) | clip Acc | clip Macro-F1 | epochs |
|---|---|---|---|
| 固定40（baseline） | 0.630 ± 0.241 | 0.555 ± 0.287 | 40 |
| **ES + reg combo** | **0.675 ± 0.297** | **0.606 ± 0.363** | 69 ± 12 |

reg combo = latent 32→16 / hidden 64→32 / dropout 0.2→0.4 / weight_decay 1e-3 /
head BatchNorm / patience 30。config: `experiment_cnn1d_es.yaml`。

被験者別（cnn1d）: 固定 `{EMLA .90, NABA .75, PAIB .62, SALE .25}` →
ES+reg `{EMLA .90, NABA .55, PAIB 1.00, SALE .25}`。PAIB が大きく改善する一方
NABA は低下、SALE は両者とも 0.25 のまま（最難被験者）。std はむしろ拡大しており、
**4被験者の少データでは依然不安定**。

### 同じレシピは LSTM には効かない（転移しない）

CNN で当たったレシピを LSTM にそのまま適用すると**悪化**した：

| 構成 (lstm) | clip Acc | clip Macro-F1 |
|---|---|---|
| 固定40（baseline） | 0.601 ± 0.270 | 0.548 ± 0.319 |
| ES + reg combo（CNN と同設定） | 0.417 ± 0.177 | 0.316 ± 0.222 |

→ ハイパラはアーキ依存で、LSTM には別途チューニングが必要。**「ES を入れれば必ず上がる」
わけではない**ことの実証でもある。

### 含意

- early stopping は**それ単体では本データ（4被験者）でむしろ不利**。validation 用に
  被験者を取られる損失が大きい。
- **強い正則化 + 容量削減と組み合わせると CNN では baseline を上回る**（0.630→0.675）が、
  std は大きく、効果はアーキ・ハイパラに敏感。
- 既存の固定 epochs パスは baseline として残してあり（後方互換）、ES は opt-in。
  最終配布モデル（`model.joblib`）は両経路とも全データ固定 epoch で refit され不変。

> 注: ES 経路の詳細指標（per-fold validation 被験者・停止 epoch 等）は各 run の
> `metrics.json` の `early_stopping` / 各 fold の `validation`、または
> `expr-evaluate --run <dir>` の "Early stopping" 節で確認できる。

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
metrics の全数値は各 run の `metrics.json`、または `expr-evaluate --run <dir> --json` で参照。）
