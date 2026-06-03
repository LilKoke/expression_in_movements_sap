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

## 付録: この結果を生成した run

| run dir | config | プロトコル |
|---|---|---|
| `outputs/rf_1f7822cf/`        | `experiment_rf.yaml`          | LOSO |
| `outputs/cnn1d_4359693b/`     | `experiment_cnn1d.yaml`       | LOSO |
| `outputs/lstm_566bd9d1/`      | `experiment_lstm.yaml`        | LOSO |
| `outputs/rf_intra_47958e45/`  | `experiment_rf_intra.yaml`    | intra |
| `outputs/cnn1d_intra_f52b073c/` | `experiment_cnn1d_intra.yaml` | intra |
| `outputs/lstm_intra_78d027f8/` | `experiment_lstm_intra.yaml`  | intra |

（run dir 名のハッシュは config 内容に依存。再学習で同一 config なら同じハッシュになる。
metrics の全数値は各 run の `metrics.json`、または `expr-evaluate --run <dir> --json` で参照。）
