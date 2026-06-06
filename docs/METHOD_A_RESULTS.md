# Method A Results — Expert Features + RandomForest

This document summarizes the Method A results only.
Method A uses clip-level expert features and RandomForest.

This document does not include detailed discussion, slide preparation, or final interpretation.

---

## 1. Method A overview

Method A uses a clip-level expert feature table.

Input feature table:

- data/processed/features.parquet

Classifier:

- random_forest_expert

Evaluation protocols:

- LOSO: leave-one-subject-out evaluation
- intra-subject: within-subject evaluation

---

## 2. Expert features

The following four expert features are used.

| Feature | Description |
|---|---|
| walking_speed | Walking speed feature |
| stride_length_proxy | Proxy feature for stride length |
| arm_swing_mean | Mean arm swing magnitude |
| head_vertical_range | Vertical range of head motion |

Each clip is represented by these four expert features.

---

## 3. Run directories

| Protocol | Run directory |
|---|---|
| LOSO | outputs/rf_expert_ff1c9e75 |
| intra-subject | outputs/rf_expert_intra_eaf0d6ee |

Each run directory contains metrics.json, predictions.jsonl, config.yaml, model.joblib, and metadata.json.

---

## 4. Main results

### Clip-level results

| Protocol | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| LOSO | 0.5762 | 0.5324 | 0.5750 |
| intra-subject | 0.8405 | 0.7940 | 0.8656 |

### Window-level results

| Protocol | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| LOSO | 0.6898 | 0.5272 | 0.5731 |
| intra-subject | 0.8904 | 0.7872 | 0.8470 |

---

## 5. Generated figures

The following figures can be generated locally for presentation or reference.

### Expert-feature PCA

Expected output path:

- reports/issue25/expert_feature_pca.png

Meaning:

- One point represents one clip.
- Color represents the emotion label.
- Marker shape represents the subject.
- The four expert features are standardized and projected to 2D by PCA.

### Confusion matrices

| Protocol | Figure path |
|---|---|
| LOSO | reports/confusion/rf_expert_ff1c9e75_clip_confusion_matrix.png |
| intra-subject | reports/confusion/rf_expert_intra_eaf0d6ee_clip_confusion_matrix.png |

Meaning:

- Rows are true labels.
- Columns are predicted labels.
- Diagonal cells are correct predictions.
- Off-diagonal cells are misclassifications.

---

## 6. Regenerate figures

### Regenerate expert-feature PCA

Run from the repository root:

    mkdir -p reports/issue25

    uv run python - <<'PY'
    import pandas as pd
    from expr_movements.viz import feature_pca

    df = pd.read_parquet("data/processed/features.parquet")
    out = feature_pca(
        df,
        "reports/issue25/expert_feature_pca.png",
        title="Expert-feature PCA"
    )

    print(out)
    print("exists:", out.exists())
    print("size:", out.stat().st_size)
    PY

### Regenerate Method A confusion matrices

Run from the repository root:

    mkdir -p reports/confusion

    uv run python scripts/plot_clip_confusion.py --run outputs/rf_expert_ff1c9e75
    uv run python scripts/plot_clip_confusion.py --run outputs/rf_expert_intra_eaf0d6ee

Expected output files:

- reports/confusion/rf_expert_ff1c9e75_clip_confusion_matrix.png
- reports/confusion/rf_expert_intra_eaf0d6ee_clip_confusion_matrix.png

---

## 7. Notes for downstream users

- Generated PNG files are not committed by default.
- Use the commands above to regenerate figures locally.
- Detailed interpretation and slide preparation are outside the scope of this document.
