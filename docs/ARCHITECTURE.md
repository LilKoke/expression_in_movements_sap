# Architecture

Code architecture for this project — the file/dependency structure, script
granularity, and the conventions that keep it modular, config-driven and
reproducible. This is the deliverable of issue #2 (実装環境の整備); the modeling
work itself is split into the phase issues linked from #1.

## Goal

Classify expressive walking motion (motion-capture **TRC**, 41 markers × XYZ
time series) into 4 emotion classes, and **compare two modeling approaches** on
the *same* subject-grouped split:

- **Approach A** — expert / hand-crafted features + classic ML (RandomForest / SVM)
- **Approach B** — a neural network (LSTM / 1D-CNN) on raw pose sequences

Emotion labels come from the 3-letter filename code: `TRE=sad`, `COE=angry`,
`NEE=neutral`, `JOE=happy`.

## Design principles

1. **Modular & swappable.** Every model implements one sklearn-style
   `fit/predict` interface and is selected by config name through a registry, so
   approach A ↔ B is a one-line config change and the same training/eval harness
   drives both.
2. **Config-driven.** All hyperparameters live in `configs/*.yaml`, validated by
   Pydantic at startup. No hardcoded params in code.
3. **Reproducible — params + model output together.** Each run writes the
   *resolved* config, model checkpoint, metrics and provenance metadata into one
   immutable directory keyed by a hash of the config.
4. **Persist processed data, don't recompute.** raw → interim → processed, with
   processed datasets written to disk (Parquet / npz / JSONL) and reloaded.
5. **Thin scripts, fat library.** CLI entry points only parse args and delegate;
   all logic lives in `src/expr_movements/`.

## Repository layout

```
configs/                  experiment YAMLs (versioned)
  experiment_rf.yaml         approach A
  experiment_lstm.yaml       approach B
data/                     GITIGNORED (large / regenerable)
  raw/                       original .trc — immutable
  interim/                   parsed/cleaned per-clip arrays
  processed/                 manifest.jsonl + features.parquet + sequences.npz
outputs/                  GITIGNORED — per-run artifacts (see below)
src/expr_movements/
  config.py                 Pydantic schemas; load_experiment()
  run.py                    run dir + config<->artifact binding  [implemented]
  splits.py                 subject-grouped train/test splitting
  train.py / evaluate.py    orchestration + the A-vs-B comparison
  data/
    trc.py                  TRC parsing + filename metadata
    dataset.py              manifest (jsonl) + sequence tensor (npz)
  features/__init__.py      expert features -> feature table (parquet)
  models/
    base.py                 BaseClassifier (sklearn interface)  [implemented]
    registry.py             register() / build_model()          [implemented]
    classic.py              approach A: random_forest, svm
    neural.py               approach B: lstm, cnn1d
  cli/                      thin entry points (expr-parse/-build-dataset/
                            -featurize/-train/-evaluate)
tests/
docs/ARCHITECTURE.md
```

## Data flow

```
data/raw/*.trc
   │  expr-parse        (data/trc.py)
   ▼
data/interim/*          per-clip arrays + filename metadata
   │  expr-build-dataset (data/dataset.py)
   ▼
data/processed/manifest.jsonl          human-inspectable clip/label index
data/processed/sequences.npz           approach B input  ──┐
   │  expr-featurize    (features/)                         │
   ▼                                                        │
data/processed/features.parquet        approach A input  ──┤
                                                            ▼
                              expr-train --config …  (train.py)
                                build_model(name, **params) over
                                subject-grouped splits (splits.py)
                                            │
                                            ▼
                              outputs/<name>_<config-hash>/
                                config.yaml  model.*  metrics.json  metadata.json
                                            │
                                            ▼
                              expr-evaluate --compare A B  (evaluate.py)
```

## Run directory (reproducibility contract)

A checkpoint is never saved without the exact config that produced it beside it:

```
outputs/rf_a1b2c3d4/
  config.yaml        # resolved ExperimentConfig
  metadata.json      # git commit, seed, data hashes, timestamp, metrics summary
  model.joblib       # (A) or model.pt (B)
  metrics.json       # accuracy, per-class F1, confusion matrix
  predictions.parquet
```

The directory name embeds `sha256(resolved config)[:8]`, so an identical config
re-uses the same directory and re-running a configuration is detectable.

## Config management

Plain YAML validated by Pydantic v2 (`config.py`). Chosen over Hydra because the
project is small and doesn't need config-group composition or a CLI takeover —
Pydantic adds schema validation (typos in YAML fail loudly via `extra="forbid"`)
and clean (de)serialization for the run-dir dump. Revisit Hydra only if sweep
running becomes painful.

## Avoiding data leakage (critical)

Random / frame-level splits put the same subject in both train and test and
inflate accuracy. All splitting is **subject-grouped** (`GroupKFold` /
`GroupShuffleSplit` / leave-one-subject-out via `splits.py`), and approach A and
B use the **identical** split so the comparison is valid.

## Anti-patterns this layout avoids

- God scripts — logic stays in `src/`, scripts are thin.
- Hardcoded paths/params — everything in `configs/`.
- Notebook-driven non-reproducibility — `notebooks/` is EDA only.
- Recomputing processed data every run — processed artifacts are persisted.
- Subject leakage — grouped splits, shared across approaches.
- Unpinned environment — `uv.lock` is committed.

## Status

Implemented in this PR: `config.py`, `run.py`, `models/base.py`,
`models/registry.py`, model/CLI skeletons, config YAMLs, tests for the above.
The remaining modules raise `NotImplementedError` and are filled in by the phase
issues linked from #1.
