# Approach comparison — rf_expert_intra_eaf0d6ee vs rf_intra_ef53d1ea

- Split protocol: **intra_subject** (same for both runs)
- Classes: angry, happy, neutral, sad
- A = `rf_expert_intra_eaf0d6ee`  ·  B = `rf_intra_ef53d1ea`

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
| Macro-F1 (primary) | 0.7872 ± 0.1261 | 0.8982 ± 0.0835 |
| Accuracy | 0.8904 ± 0.0663 | 0.9707 ± 0.0188 |
| Balanced accuracy | 0.8470 ± 0.1008 | 0.9722 ± 0.0094 |

Per-class F1:

| Class | A | B |
|---|---|---|
| angry | 0.589 | 0.779 |
| happy | 0.707 | 0.911 |
| neutral | 0.864 | 0.925 |
| sad | 0.989 | 0.977 |
