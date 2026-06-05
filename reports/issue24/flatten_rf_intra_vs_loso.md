# Intra-subject vs inter-subject (LOSO)

- Intra run: `rf_intra_ef53d1ea` (intra_subject)
- LOSO run: `rf_712a3a44` (leave_one_subject_out)
- Intra is directly comparable to Venture 2014 (>90%); LOSO is the real (unseen-subject) task. The gap = subject dependence.

| Macro-F1 | Intra-subject | Inter-subject (LOSO) | Gap (intra − LOSO) |
|---|---|---|---|
| Trial/clip | 0.9500 ± 0.1000 | 0.6315 ± 0.0379 | +0.3185 |
| Window | 0.8982 ± 0.0835 | 0.6041 ± 0.0165 | +0.2942 |
