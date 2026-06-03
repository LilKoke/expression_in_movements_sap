# FAQ — 用語・指標・現状の比較の意味

このプロジェクトで頻出する疑問（指標の定義、現状の `rf` ベースラインが何なのか、
PCA 図の出所、図が何を比較しているのか）をまとめる。アーキ全体は
[ARCHITECTURE.md](ARCHITECTURE.md)、数値結果は [RESULTS.md](RESULTS.md) 参照。

---

## Q1. `rf`（RandomForest）はどんな特徴量を使っている？ expert features なの？

**現状の `rf` は expert features を使っていない。** Phase 3 の不変化前処理済み姿勢系列
（骨盤中心ローカル座標 + yaw 補正 + 速度チャネル、`F=124` = 41 marker×XYZ + speed）の
**窓 `(length, F)` を flatten して `length×F` 次元のベクトルにしただけ**を RandomForest に
入れている。

- 実装: [models/classic.py](../src/expr_movements/models/classic.py) の docstring が明言
  （"this baseline runs classic ML directly on the standardised window features"）。
  flatten は [data/windows.py](../src/expr_movements/data/windows.py) の `flatten_windows`。
- つまり velocity 統計・関節角レンジ・gait cadence・体幹傾きのような**人が設計した特徴量
  ではなく、生の正規化座標を並べただけ**。
- 本物の expert features（`expr-featurize` / `features.build_feature_table`）は両方
  `NotImplementedError` で**未実装**。設計・計算の手順は #20、本物の手法A実装は
  ロードマップ #1 の Phase 9。

**含意**: 現状の `rf` は course 要件 "expert features" を満たしていない。あくまで
**共通 harness（windowing / LOSO fold / 評価コード）の疎通用・古典MLベースライン**。

---

## Q2. `rf` は expert features を NN の入力に使っただけ、では？

ほぼその通り。正確には「**手法B(NN)と同じ共通 contract（正規化窓）を flatten して
古典ML(RF)に食わせた**」もの。NN と古典MLが**同じ入力**を見ている。違うのは
**モデル（RF か 1D-CNN か）だけ**。

---

## Q3. `compare_clip_level.png` / `compare_window_level.png` は何の比較？ 不変化前処理の ablation？

**ablation ではない。** これは `rf` vs `cnn1d` の比較だが、**両者とも同じ不変化前処理済み
入力（`sequences.npz`, `pose_local_yaw+speed`）を使っている**。したがって変えているのは
**モデル（RandomForest ↔ 1D-CNN マルチタスク）だけ**で、前処理は両者で同一。

→ 正しくは **「同一前処理・同一窓・同一 LOSO fold の上でのモデル比較（古典ML vs NN）」**。

- **前処理の ablation（不変化のあり/なしの比較）は現状やっていない。** やるなら
  yaw 補正なし / 骨盤ローカルなし / 速度チャネルなし などの variant を別 config で学習し
  比較する必要がある（未実施。やるなら別 Phase）。
- さらに注意: 現状の `rf` は expert features ではない（Q1）ので、この図は course 要件の
  **「extracted features VS expert features」そのものでもない**。本来のその比較は手法A
  （expert features, #20）が揃ってから Phase 10（#21）で行う。

> まとめると、現状の `compare_*.png` は **「同じ特徴量を古典MLに入れる vs NNに入れる」の
> モデル比較**であって、前処理 ablation でも expert-vs-extracted features 比較でもない。

---

## Q4. `latent_pca.png` はどの NN から出した？

**`outputs/cnn1d_4359693b`** = 本命の **1D-CNN マルチタスクモデル**（encoder=cnn1d,
LOSO, window length=64 / stride=16, latent_dim=32, 固定40 epochs）の latent。

- latent の出所: fold 毎の held-out モデルは保存していない（保存されるのは全データで
  refit した最終モデルのみ）ため、**保存済み最終モデルで全 window を encode** して PCA に
  かけている＝**学習済み latent の可視化**であって held-out 汎化推定ではない。
  held-out の純度数値（Silhouette / Davies-Bouldin）は `metrics.json` 側が担保し、
  両者を併読する設計。詳細は [viz.py](../src/expr_movements/viz.py) の docstring。

---

## Q5. Macro-F1 と F1 は何が違う？

**F1 スコア**は、ある1クラスに対する適合率(precision)と再現率(recall)の調和平均：

```
F1 = 2 · precision · recall / (precision + recall)
```

- precision = そのクラスと予測したうち実際に正しかった割合（= TP / (TP+FP)）
- recall    = そのクラスの実サンプルのうち当てられた割合（= TP / (TP+FN)）

F1 は **1クラスごとに1つ**決まる（本タスクなら angry/happy/neutral/sad の4つ。これが
RESULTS.md の **per-class F1**）。

**Macro-F1** は、その**各クラスの F1 を単純平均**したもの：

```
Macro-F1 = (F1_angry + F1_happy + F1_neutral + F1_sad) / 4
```

- 「Macro」＝ クラスごとに計算してから**クラス重みを均等にして平均**する方式。
  クラスの**サンプル数に関わらず各クラスを等価に扱う**（少数クラスも多数クラスと同じ重み）。
- 対比される **Micro-F1** は全クラスの TP/FP/FN を合算してから1つの F1 を出す方式で、
  こちらは**サンプル数の多いクラスに引っ張られる**（多クラス単一ラベルでは Micro-F1 =
  accuracy になる）。本プロジェクトでは使っていない。
- 本タスクはクラスがほぼ均衡だが、**難クラス(happy/neutral)の取りこぼしを多数クラスで
  薄めず公平に見たい**ので Macro-F1 を**主指標**にしている。

要するに **F1 = 1クラスの指標**、**Macro-F1 = 全クラスの F1 を均等平均した全体指標**。

---

## Q6. Accuracy と Balanced accuracy は何が違う？

**Accuracy（正答率）** は、全サンプルのうち正解した割合：

```
Accuracy = 正解数 / 全サンプル数
```

- **サンプル単位**で数えるので、**多いクラスの正解が支配的**になる。極端な例: 90% が
  クラスAなら「常にAと答える」だけで Accuracy 0.90 になり、少数クラスを完全に外しても
  高く見える。

**Balanced accuracy（バランス正答率）** は、**各クラスの recall（再現率）を計算してから
平均**したもの：

```
Balanced accuracy = (recall_angry + recall_happy + recall_neutral + recall_sad) / 4
```

- recall = そのクラスの実サンプルのうち当てられた割合。これをクラスごとに出して**均等
  平均**するので、**クラス不均衡の影響を受けにくい**。「常に多数クラス」戦略は
  少数クラスの recall が 0 になるため balanced accuracy では低く出る。
- ランダム予測の期待値は **1/クラス数**（4クラスなら 0.25）。

**Macro-F1 との違い**: balanced accuracy は **recall だけ**の平均（precision を見ない）、
Macro-F1 は **precision と recall の両方**（F1）の平均。誤検出（FP）を罰したいかどうかが差。

本プロジェクトは4感情がほぼ均衡なので Accuracy と Balanced accuracy は近い値になるが、
**fold によってクラス比が崩れる**こと、混同がクラス偏在している（happy/neutral が難）ことを
考慮し、両方を併記している。**主指標は Macro-F1**、Accuracy は直感的な補助、Balanced
accuracy は「クラスを均等に見た正答率」として並べている。

---

## 指標の早見表

| 指標 | 計算 | クラス不均衡への頑健さ | 本プロジェクトでの位置づけ |
|---|---|---|---|
| F1（per-class） | 1クラスの precision と recall の調和平均 | （単一クラスの指標） | クラス別の得手不得手を見る |
| **Macro-F1** | 各クラス F1 の均等平均 | 強い（均等重み） | **主指標** |
| Accuracy | 正解数 / 全数 | 弱い（多数クラス支配） | 直感的な補助指標 |
| Balanced accuracy | 各クラス recall の均等平均 | 強い（均等重み） | クラス均等の正答率として併記 |

すべて [train.py](../src/expr_movements/train.py) の `_scores` で scikit-learn を使って
算出し、window / clip 両レベル・LOSO 4-fold の mean ± std で
[metrics.json](../src/expr_movements/evaluate.py) に永続化している。
