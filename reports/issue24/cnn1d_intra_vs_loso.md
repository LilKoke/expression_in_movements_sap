# Intra-subject vs inter-subject (LOSO)

- Intra run: `cnn1d_intra_a0fdb7b6` (intra_subject)
- LOSO run: `cnn1d_06548d90` (leave_one_subject_out)
- Intra is directly comparable to Venture 2014 (>90%); LOSO is the real (unseen-subject) task. The gap = subject dependence.

| Macro-F1 | Intra-subject | Inter-subject (LOSO) | Gap (intra − LOSO) |
|---|---|---|---|
| Trial/clip | 0.9500 ± 0.1000 | 0.5911 ± 0.2995 | +0.3589 |
| Window | 0.9382 ± 0.1016 | 0.5812 ± 0.2973 | +0.3570 |
