# Approach comparison — rf_expert_ff1c9e75 vs rf_712a3a44

- Split protocol: **leave_one_subject_out** (same for both runs)
- Classes: angry, happy, neutral, sad
- A = `rf_expert_ff1c9e75`  ·  B = `rf_712a3a44`

## Trial/clip level (majority vote)

| Metric | A | B |
|---|---|---|
| Macro-F1 (primary) | 0.5324 ± 0.2372 | 0.6315 ± 0.0379 |
| Accuracy | 0.5762 ± 0.2375 | 0.7048 ± 0.0535 |
| Balanced accuracy | 0.5750 ± 0.2358 | 0.7125 ± 0.0415 |

Per-class F1:

| Class | A | B |
|---|---|---|
| angry | 0.543 | 0.688 |
| happy | 0.221 | 0.472 |
| neutral | 0.470 | 0.394 |
| sad | 0.896 | 0.972 |

## Window level

| Metric | A | B |
|---|---|---|
| Macro-F1 (primary) | 0.5272 ± 0.2402 | 0.6041 ± 0.0165 |
| Accuracy | 0.6898 ± 0.2171 | 0.7692 ± 0.0246 |
| Balanced accuracy | 0.5731 ± 0.2372 | 0.6839 ± 0.0052 |

Per-class F1:

| Class | A | B |
|---|---|---|
| angry | 0.496 | 0.643 |
| happy | 0.215 | 0.418 |
| neutral | 0.474 | 0.392 |
| sad | 0.924 | 0.964 |
