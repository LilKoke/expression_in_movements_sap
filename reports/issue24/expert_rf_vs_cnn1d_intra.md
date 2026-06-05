# Approach comparison — rf_expert_intra_eaf0d6ee vs cnn1d_intra_a0fdb7b6

- Split protocol: **intra_subject** (same for both runs)
- Classes: angry, happy, neutral, sad
- A = `rf_expert_intra_eaf0d6ee`  ·  B = `cnn1d_intra_a0fdb7b6`

## Trial/clip level (majority vote)

| Metric | A | B |
|---|---|---|
| Macro-F1 (primary) | 0.7940 ± 0.1226 | 0.9500 ± 0.1000 |
| Accuracy | 0.8405 ± 0.0999 | 1.0000 ± 0.0000 |
| Balanced accuracy | 0.8656 ± 0.0848 | 1.0000 ± 0.0000 |

Per-class F1:

| Class | A | B |
|---|---|---|
| angry | 0.621 | 0.800 |
| happy | 0.729 | 1.000 |
| neutral | 0.843 | 1.000 |
| sad | 0.983 | 1.000 |

## Window level

| Metric | A | B |
|---|---|---|
| Macro-F1 (primary) | 0.7872 ± 0.1261 | 0.9382 ± 0.1016 |
| Accuracy | 0.8904 ± 0.0663 | 0.9878 ± 0.0191 |
| Balanced accuracy | 0.8470 ± 0.1008 | 0.9924 ± 0.0117 |

Per-class F1:

| Class | A | B |
|---|---|---|
| angry | 0.589 | 0.783 |
| happy | 0.707 | 0.987 |
| neutral | 0.864 | 0.989 |
| sad | 0.989 | 0.993 |

## Latent separability (held-out, NN only)

| Metric | A | B |
|---|---|---|
| Silhouette (↑) | — | 0.4811 |
| Davies-Bouldin (↓) | — | 0.8119 |
