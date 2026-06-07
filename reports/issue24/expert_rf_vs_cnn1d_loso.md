# Approach comparison — rf_expert_ff1c9e75 vs cnn1d_06548d90

- Split protocol: **leave_one_subject_out** (same for both runs)
- Classes: angry, happy, neutral, sad
- A = `rf_expert_ff1c9e75`  ·  B = `cnn1d_06548d90`

## Trial/clip level (majority vote)

| Metric | A | B |
|---|---|---|
| Macro-F1 (primary) | 0.5324 ± 0.2372 | 0.5911 ± 0.2995 |
| Accuracy | 0.5762 ± 0.2375 | 0.6542 ± 0.2476 |
| Balanced accuracy | 0.5750 ± 0.2358 | 0.6625 ± 0.2484 |

Per-class F1:

| Class | A | B |
|---|---|---|
| angry | 0.543 | 0.637 |
| happy | 0.221 | 0.458 |
| neutral | 0.470 | 0.520 |
| sad | 0.896 | 0.750 |

## Window level

| Metric | A | B |
|---|---|---|
| Macro-F1 (primary) | 0.5272 ± 0.2402 | 0.5812 ± 0.2973 |
| Accuracy | 0.6898 ± 0.2171 | 0.6950 ± 0.2674 |
| Balanced accuracy | 0.5731 ± 0.2372 | 0.6481 ± 0.2413 |

Per-class F1:

| Class | A | B |
|---|---|---|
| angry | 0.496 | 0.599 |
| happy | 0.215 | 0.462 |
| neutral | 0.474 | 0.518 |
| sad | 0.924 | 0.745 |

## Latent separability (held-out, NN only)

| Metric | A | B |
|---|---|---|
| Silhouette (↑) | — | 0.5693 |
| Davies-Bouldin (↓) | — | 0.7943 |
