# Team A — 発表台本（Approach A: Expert Features + RandomForest）

歩行モーションからの感情分類（4クラス: happy / angry / neutral / sad）。
**手法A = 人が設計した歩行特徴量 + RandomForest**。Aチーム単独・**英語で約2分**・5スライド。

- スライド本番ファイル: [`slides.html`](slides.html)（ブラウザで開き `F` で全画面、`→` で送る、`S` で台本表示）
- 図・動画: [`img/`](img/)（再生成: `uv run python docs/presentation/A-Team/make_assets.py`）
- 数値の出典: [`../../METHOD_A_RESULTS.md`](../../METHOD_A_RESULTS.md) ・ [`../../../reports/issue24/`](../../../reports/issue24/)（すべて clip-level、手法A = `random_forest_expert`）

## 時間配分

| # | スライド | 役割 | 使用する図 | 目安 |
|---|---|---|---|---:|
| 1 | Title & hook | 掴み・立ち位置 | `walk_compare.mp4` | ~15s |
| 2 | The four features | 手法の核 | `gait_features.png` | ~30s |
| 3 | Evaluation setup | 公平性の前提 | `protocols.png` | ~20s |
| 4 | Results | 主結果 | `results_intra_vs_loso.png` + `perclass_f1_loso.png` | ~35s |
| 5 | Discussion vs baseline | 持ち帰り | `expert_vs_flatten.png` | ~25s |

---

## Slide 1 — Title & hook（~15s・図: `walk_compare.mp4`）

**要点（日本語）**
- 課題: モーションキャプチャの歩行データから4感情（happy / angry / neutral / sad）を分類。
- 立ち位置: B班は生データをNNに入れる。**A班は「解釈できる少数の手設計特徴量」でどこまでいけるか**を検証。
- 掴み: 同一人物(NABA)の happy vs sad の歩行動画で「歩き方に感情が出る」を視覚的に提示。

**台本（English）**
> "We're Team A. The task: recognise four emotions — happy, angry, neutral, sad — from motion-capture walking. As you can see, the same person walks very differently when happy versus sad. Team B feeds the raw motion to a neural network; we take the hand-crafted route — a few interpretable gait features."

---

## Slide 2 — The four expert features（~30s・図: `gait_features.png`）

**要点（日本語）**
- 使う特徴量は4つだけ: `walking_speed`（歩く速さ）/ `stride_length_proxy`（歩幅）/ `arm_swing_mean`（腕振り）/ `head_vertical_range`（頭の上下動）。
- 直感に対応: 元気だと速く大きく歩く・落ち込むと頭が下がる。
- 前処理で**人・歩く向きに不変**: 骨盤中心ローカル座標 + yaw整列（+速度チャネル）。
- 1クリップ → この**4次元ベクトル**に圧縮し、RandomForest で分類。

**台本（English）**
> "We describe each walk with just four features: walking speed, a stride-length proxy, average arm swing, and the vertical range of head motion. They match intuition — energetic people walk fast with big arm swings; sad people slow down and drop the head. Before measuring them, we normalise every pose to pelvis-local coordinates and align the facing direction, so different people and walking directions stay comparable. Each clip becomes this single four-dimensional vector, classified by a RandomForest."

---

## Slide 3 — Evaluation: two protocols（~20s・図: `protocols.png`）

**要点（日本語）**
- 評価は2種類。**intra-subject**（同一人物を train/test 両方に）= 先行研究 Venture et al. 2014（>90%）と比較可能な設定。
- **LOSO**（leave-one-subject-out, 1人を丸ごとテスト）= 未知の人で評価する**本番相当の難しい設定**。
- どちらも**被験者単位で分割**＝同じ人が train/test にまたがらない（リーク無し）。

**台本（English）**
> "We evaluate two ways. Intra-subject keeps the same people in training and test — comparable to Venture and colleagues' prior work at over ninety percent. Leave-one-subject-out tests on a person never seen in training — the realistic, harder task. Both split strictly by subject, so nobody appears in both sides."

---

## Slide 4 — Results（~35s・図: `results_intra_vs_loso.png` + `perclass_f1_loso.png`）

**要点（日本語）**
- clip-level の主結果: **intra = Macro-F1 0.79 / Acc 0.84**、**LOSO = Macro-F1 0.53 / Acc 0.58**。
- **被験者依存ギャップ ≈ +0.26（Macro-F1）** が一番の発見＝感情の出方には個人差が大きい。
- クラス別（LOSO）: **sad 0.90（得意）**↔ **happy 0.22（苦手）**、angry 0.54 / neutral 0.47。happyは中立・怒りと混同。
- 主指標は Macro-F1。Accuracy は補助。

**台本（English）**
> "At the clip level, the same-person setting reaches Macro-F1 0.79 and 84 percent accuracy. On unseen people it drops to 0.53 and 58 percent — a 0.26 gap in Macro-F1. That gap is the headline: emotional expression is highly individual. Per class, sadness is easy at F1 0.90 — slow, collapsed posture is distinctive — while happiness is hardest at just 0.22, confused with neutral and angry."

---

## Slide 5 — Discussion: vs a naive flatten baseline（~25s・図: `expert_vs_flatten.png`）

**要点（日本語）**
- 比較対象 = **同じ前処理・分割・モデルで、特徴量だけ違う**対照実験。我々の4次元 vs 姿勢を素朴にflattenした **~7,936次元**。
- flatten RF が勝つ（LOSO 0.63 > 0.53、intra 0.95 > 0.79）→ **4特徴量は情報を捨てている**。
- ただし flatten は**解釈不可**、かつ intra で **Acc=1.00 → 人を丸暗記**（未知被験者で 0.95→0.63 と急落）。
- 我々の特徴量は**人に不変**な設計＝頭打ちは低いが**落ち方が正直**。
- 締め: **解釈性 vs 精度・汎化のトレードオフ** → 手法B（学習特徴量）の比較へ橋渡し。
- （口頭の補足可）CNN1D は LOSO 0.59 / intra 0.95。詳細比較は合同パートで。

**台本（English）**
> "Finally, the honest lesson. We compared against a naive baseline: same preprocessing, same splits, same RandomForest, but instead of four features it just flattens the full pose — about eight thousand numbers. It beats us — 0.63 versus 0.53 on unseen people — so four features discard too much. Yet that baseline is uninterpretable, and on seen people it hits perfect accuracy: a sign it memorises the individual, collapsing to 0.63 on unseen ones. Our features are person-invariant by design — a lower ceiling, but a more honest drop. The real takeaway from Approach A is the trade-off between interpretability and accuracy — which sets up the comparison with Team B's learned features."

---

## バックアップ図（Q&A・差し替え用）

| 図 | 用途 |
|---|---|
| `img/feature_pca.png` | 4特徴量のPCA（実データ）。sad/angryは明確に分離、happy/neutralは重なり気味 → 「4特徴量で感情が分かれるか」をQ&Aで提示 |
| `img/pipeline.png` | 手法全体の流れ図。Slide 1 の代替ビジュアル |
| `img/walk_compare.gif` | 動画が自動再生できない環境向けフォールバック（PPT/HTMLで自動ループ） |

## メモ
- 台本は5枚合計**約125秒**で2分に収まる配分。
- 最大の聞かせどころは Slide 5 の「flatten RFにも負ける（0.53 < 0.63）」を**正直に出す**こと。課題要件「expert features vs 学習特徴量」の比較に自然につながる。
