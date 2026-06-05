# Intra-subject vs inter-subject (LOSO)

- Intra run: `rf_expert_intra_eaf0d6ee` (intra_subject)
- LOSO run: `rf_expert_ff1c9e75` (leave_one_subject_out)
- Intra is directly comparable to Venture 2014 (>90%); LOSO is the real (unseen-subject) task. The gap = subject dependence.

| Macro-F1 | Intra-subject | Inter-subject (LOSO) | Gap (intra − LOSO) |
|---|---|---|---|
| Trial/clip | 0.7940 ± 0.1226 | 0.5324 ± 0.2372 | +0.2616 |
| Window | 0.7872 ± 0.1261 | 0.5272 ± 0.2402 | +0.2599 |
