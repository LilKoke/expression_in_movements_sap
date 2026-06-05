# Issue #24 Evaluation Summary

## Run directories

| Method | Protocol | Run dir |
|---|---|---|
| expert RF | LOSO | `outputs/rf_expert_ff1c9e75` |
| expert RF | intra-subject | `outputs/rf_expert_intra_eaf0d6ee` |
| flatten RF | LOSO | `outputs/rf_712a3a44` |
| flatten RF | intra-subject | `outputs/rf_intra_ef53d1ea` |
| CNN1D | LOSO | `outputs/cnn1d_06548d90` |
| CNN1D | intra-subject | `outputs/cnn1d_intra_a0fdb7b6` |

## Main comparison: expert RF vs CNN1D

### LOSO

| Metric | expert RF | CNN1D |
|---|---:|---:|
| Clip macro-F1 | 0.5324 | 0.5911 |
| Clip accuracy | 0.5762 | 0.6542 |
| Clip balanced accuracy | 0.5750 | 0.6625 |
| Window macro-F1 | 0.5272 | 0.5812 |

### intra-subject

| Metric | expert RF | CNN1D |
|---|---:|---:|
| Clip macro-F1 | 0.7940 | 0.9500 |
| Clip accuracy | 0.8405 | 1.0000 |
| Clip balanced accuracy | 0.8656 | 1.0000 |
| Window macro-F1 | 0.7872 | 0.9382 |

## Baseline comparison: expert RF vs flatten RF

### LOSO

| Metric | expert RF | flatten RF |
|---|---:|---:|
| Clip macro-F1 | 0.5324 | 0.6315 |
| Clip accuracy | 0.5762 | 0.7048 |
| Clip balanced accuracy | 0.5750 | 0.7125 |
| Window macro-F1 | 0.5272 | 0.6041 |

### intra-subject

| Metric | expert RF | flatten RF |
|---|---:|---:|
| Clip macro-F1 | 0.7940 | 0.9500 |
| Clip accuracy | 0.8405 | 1.0000 |
| Clip balanced accuracy | 0.8656 | 1.0000 |
| Window macro-F1 | 0.7872 | 0.8982 |

## Protocol gap: intra-subject vs LOSO

| Method | Clip macro-F1 gap | Window macro-F1 gap |
|---|---:|---:|
| expert RF | +0.2616 | +0.2599 |
| CNN1D | +0.3589 | +0.3570 |
| flatten RF | +0.3185 | +0.2942 |

## Observations

- All methods perform better in intra-subject evaluation than in LOSO.
- This indicates a clear subject-dependence gap.
- CNN1D outperforms expert RF in both LOSO and intra-subject settings.
- flatten RF also outperforms expert RF in this run.
- expert RF still runs through the same train/evaluate harness and produces comparable metrics, so it can be used as the expert-feature baseline.

## Generated comparison reports

- `reports/issue24/expert_rf_vs_cnn1d_loso.md`
- `reports/issue24/expert_rf_vs_cnn1d_intra.md`
- `reports/issue24/expert_rf_vs_flatten_rf_loso.md`
- `reports/issue24/expert_rf_vs_flatten_rf_intra.md`
- `reports/issue24/expert_rf_intra_vs_loso.md`
- `reports/issue24/cnn1d_intra_vs_loso.md`
- `reports/issue24/flatten_rf_intra_vs_loso.md`
