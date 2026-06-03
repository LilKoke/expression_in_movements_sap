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

## Dataset persistence (Phase 2, #4)

`expr-build-dataset` (`data/dataset.py`) turns the parsed interim clips into two
persisted artifacts under `data/processed/`, so modeling never re-parses raw TRC:

- **`manifest.jsonl`** — one JSON line per clip: interim path, subject, emotion
  (code + name), take, raw frame count, and the kept `[trim_start, trim_stop)`
  window with its length. Human-inspectable; it is the index the sequence build
  consumes.
- **`sequences.npz`** — the approach-B input (see *length handling* below).

**Trimming (motion onset/offset).** Each clip has standing/idle frames before
the first step and after the last. With `DataConfig.detect_onset` (default on)
the active walking window is found from **marker speed**: per frame, the mean
inter-frame displacement over markers (NaN-robust, so occluded markers are
ignored); a frame is "moving" once that exceeds `onset_speed_frac` of the clip's
peak, and onset/offset are the first/last runs of `onset_min_run` consecutive
moving frames. `trim_start_frames` / `trim_end_frames` then apply as an extra
fixed margin *inside* that window. A clip is never trimmed to empty — if the
margins would do so they are skipped. (On the bundled data this drops a median
of ~50 leading/trailing static frames per clip.) Detection params live in
`DataConfig`, so the policy is config-driven and recorded in the resolved config.

**Variable-length handling (decision).** Clip lengths vary (≈73–372 frames on
the bundled data). Rather than bake a fixed frame count into the dataset, the
**canonical** store is *variable-length*: `sequences` is an object array of
per-clip `(T_i, 41*3)` float32 arrays (already onset-trimmed), carried with
`lengths`, `labels`, `emotion_codes`, `subjects`, `clips` and `marker_names`.

Rationale: the modeling frame count is a *training-time* hyperparameter we want
to sweep (and to support sliding-window augmentation). Slicing a fixed window /
length out of the variable-length arrays is an in-memory numpy operation on a
few tens of MB — milliseconds, cheaper than the disk read — so there is no
speed reason to freeze `N` at build time, and zero-padding up front would force
every consumer to carry a mask to avoid training on fake frames. Persisting the
true-length sequences keeps that decision out of the dataset.

For callers that *do* want a ready-made fixed tensor, `--target-frames N`
additionally writes a dense `sequences_dense` `(n_clips, N, 41*3)` (pad with
zeros / truncate to `N`) plus a boolean `mask` marking the real frames. This is
reproducible from the canonical store at any time, so freezing an `N` is a
re-run, not a one-way door.

**CSV output (#1).** The course brief asks for a `(coordinates, class)` dataset
"as CSV". We persist the modeling artifacts as JSONL + npz instead (npz is the
natural dense/ragged numeric container and avoids stringifying ~10⁶ floats per
clip), and treat CSV as an *export* concern rather than the working format. A
flat CSV of `manifest.jsonl` (one row per clip, the label and frame window) is a
trivial `pandas.read_json(lines=True).to_csv(...)`; a long-format
`(clip, frame, marker, x, y, z, emotion)` CSV is derivable from `sequences.npz`
when an assignment deliverable needs it. Keeping the working store binary and
generating CSV on demand avoids carrying a redundant, bulky third copy.

## Invariant pre-processing (Phase 3, #5)

The classifier must key on *how* a subject moves, not *where* in the capture
volume they walked or *which way* they faced. So before a clip's trimmed pose
sequence is stored in `sequences.npz`, `features/preprocess.py`
(`normalize_sequence`) makes it **position- and heading-invariant**, while
deliberately **keeping walking speed** as an explicit channel (speed tracks
arousal — a strong emotion cue, per Venture et al. 2014, the source paper for
this data). This is the **common representation** consumed by both modeling
teams (per the #1 roadmap v2, Phase 3 stops here — no derived expert features
are computed in this repo; that is the other team's approach-A work).

Per clip, in order:

1. **Pelvis-centred local coordinates.** Each frame is translated so the
   centroid of the four pelvis markers (`LFWT/RFWT/LBWT/RBWT`) sits at the
   origin — removing absolute walking position and the slow drift across the
   room. Markers carry a per-subject prefix (`NABA_LFWT`); they are matched by
   the bare token after the last `_`. The centroid is NaN-robust, so an occluded
   pelvis marker doesn't poison it; a frame with *no* pelvis markers is left
   unshifted rather than dropped.
2. **Yaw alignment** (`yaw_align`, default on). Each frame is rotated about the
   vertical (`up_axis`, Y here) so the pelvis left→right axis points in a fixed
   direction — removing heading. **Pitch and roll are left intact** (trunk lean,
   head orientation) because those carry posture/emotion; only the horizontal
   heading is normalized away.
3. **Speed channel** (`keep_speed`, default on). The per-frame body speed —
   the pelvis-centroid displacement magnitude in **world** coordinates, computed
   *before* centring (centring would zero it) — is appended as a trailing
   column. So a stored sequence is `(T, 41*3 + 1)`: 123 pose columns then 1
   speed column.

All four behaviours are `DataConfig` fields (`normalize`, `pelvis_markers`,
`yaw_align`, `keep_speed`, `up_axis`), so the policy is config-driven and
recorded in the resolved config. With `normalize=False` the raw flattened
`(T, 41*3)` world coordinates are stored unchanged (used by the synthetic
dataset tests, which have no pelvis markers). `sequences.npz` records the column
split via `feature_layout` (`"pose_local_yaw+speed"`), `has_speed_channel` and
`n_markers`, so consumers split pose vs. speed by metadata rather than guessing.

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

## Early stopping (optional, Phase 8, #1)

By default every NN trains for a flat `model.params.epochs` with **no validation
set** — that is the baseline path and stays untouched. Adding a `validation`
block to an experiment YAML (`ValidationConfig`) turns on **early stopping**:

```yaml
validation:
  enabled: true
  strategy: group_subject   # hold out a whole subject from the train side
  val_size: 0.34            # ~1 of 3 LOSO-train subjects
  patience: 10
  monitor: macro_f1         # accuracy | loss also supported
  restore_best: true
```

For each fold, the harness carves a **validation set out of that fold's train
side** (`splits.nested_validation_split`), trains the NN with the validation
metric watched each epoch, and keeps the weights from the best epoch. With
`strategy: group_subject` (default) the validation set is a **whole held-out
subject** — so, like the test fold, the early-stopping signal comes from an
*unseen person* and model selection stays honest with respect to the
inter-subject task (no same-subject leakage into the stopping criterion).
`stratified_clip` is the looser seen-person alternative, kept for comparison.

Key boundaries:

- **NN-only.** Classic ML (RandomForest/SVM) has no epochs; the harness ignores
  the block and trains normally, recording that ES did not apply.
- **The saved artifact is unaffected.** Early stopping shapes the *per-fold*
  models that produce the reported held-out metrics. The distributed checkpoint
  (`model.joblib`) is still refit on **all** data for a fixed epoch count.
- **Self-describing runs.** `metrics.json` gains an `early_stopping` block (mean
  epochs run, monitor, folds used) and each fold a `validation` entry naming the
  held-out validation subject.

This runs *parallel to* the fixed-epochs path so the two are directly
comparable: `experiment_cnn1d.yaml` (fixed 40 epochs) vs
`experiment_cnn1d_es.yaml` (early stopping). Results: see `docs/RESULTS.md`.

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
