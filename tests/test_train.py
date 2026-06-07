"""End-to-end training test (Phase 4, #6).

Builds a tiny synthetic ``sequences.npz`` with a learnable signal, runs
``run_training`` under LOSO, and asserts the run dir holds the resolved config,
model, metrics and predictions — i.e. the issue-#6 requirement that params and
the corresponding model are output together, on the shared A-vs-B harness.
"""

from __future__ import annotations

import json

import joblib
import numpy as np

from expr_movements.config import ExperimentConfig, ModelConfig, SplitConfig, WindowConfig
from expr_movements.train import run_training


def _write_synthetic_npz(path, n_per_subject=6, f=6, length=40):
    """4 subjects x 2 classes, each class a different constant offset (learnable)."""
    subjects = ["A", "B", "C", "D"]
    classes = {"calm": 0.0, "excited": 5.0}
    seqs, labels, subs, clips = [], [], [], []
    rng = np.random.default_rng(0)
    i = 0
    for s in subjects:
        for cls, offset in classes.items():
            for _ in range(n_per_subject):
                base = offset + rng.normal(0, 0.1, size=(length, f)).astype(np.float32)
                seqs.append(base)
                labels.append(cls)
                subs.append(s)
                clips.append(f"{s}_{cls}_{i}.npz")
                i += 1
    obj = np.empty(len(seqs), dtype=object)
    obj[:] = seqs
    np.savez_compressed(
        path,
        sequences=obj,
        labels=np.asarray(labels, dtype=object),
        subjects=np.asarray(subs, dtype=object),
        clips=np.asarray(clips, dtype=object),
        emotion_codes=np.asarray(labels, dtype=object),
        marker_names=np.asarray([f"m{j}" for j in range(f // 3 or 1)], dtype=object),
        lengths=np.asarray([length] * len(seqs)),
        feature_layout=np.asarray("synthetic"),
        n_markers=np.asarray(f // 3 or 1),
        has_speed_channel=np.asarray(False),
    )


def _make_cfg(processed_dir, strategy="leave_one_subject_out"):
    return ExperimentConfig(
        name="rf_test",
        data={"processed_dir": str(processed_dir)},
        window=WindowConfig(length=20, stride=10),
        split=SplitConfig(strategy=strategy, n_splits=4),
        model=ModelConfig(name="random_forest", params={"n_estimators": 30, "random_state": 0}),
    )


def test_run_training_writes_full_run_dir(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_synthetic_npz(processed / "sequences.npz")

    cfg = _make_cfg(processed)
    run_dir = run_training(cfg, outputs_root=tmp_path / "outputs")

    # All four artifacts land together in the immutable run dir.
    for name in (
        "config.yaml",
        "model.joblib",
        "metrics.json",
        "metadata.json",
        "predictions.jsonl",
    ):
        assert (run_dir / name).exists(), name

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert len(metrics["folds"]) == 4  # LOSO over 4 subjects
    # Signal is trivially learnable -> strong inter-subject score.
    assert metrics["clip_level"]["macro_f1_mean"] > 0.9

    bundle = joblib.load(run_dir / "model.joblib")
    assert {"model", "scaler_mean", "scaler_std", "labels"} <= set(bundle)
    assert sorted(bundle["labels"]) == ["calm", "excited"]


def test_early_stopping_path_runs_and_records(tmp_path):
    """An NN run with validation.enabled early-stops per LOSO fold and records it.

    The metrics carry an ``early_stopping`` block and each fold a ``validation``
    entry naming the held-out validation subject (an unseen person, disjoint from
    the test subject). The split protocol/folds are otherwise unchanged.
    """
    from expr_movements.config import ValidationConfig

    processed = tmp_path / "processed"
    processed.mkdir()
    _write_synthetic_npz(processed / "sequences.npz")

    cfg = ExperimentConfig(
        name="cnn1d_es_test",
        data={"processed_dir": str(processed)},
        window=WindowConfig(length=20, stride=10),
        split=SplitConfig(strategy="leave_one_subject_out", n_splits=4),
        validation=ValidationConfig(
            enabled=True, strategy="group_subject", val_size=0.34, patience=3, monitor="macro_f1"
        ),
        model=ModelConfig(
            name="cnn1d", params={"epochs": 50, "latent_dim": 8, "hidden_size": 16, "batch_size": 16}
        ),
    )
    run_dir = run_training(cfg, outputs_root=tmp_path / "outputs")
    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert "early_stopping" in metrics
    assert metrics["early_stopping"]["enabled"] is True
    assert metrics["early_stopping"]["folds_used"] >= 1

    # Each fold records its validation subject, disjoint from the test subject.
    for fold in metrics["folds"]:
        v = fold.get("validation")
        assert v is not None and v["used"] is True
        assert set(v["val_subjects"]).isdisjoint(set(fold["test_subjects"]))

    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["early_stopping"] is True


def test_early_stopping_ignored_for_classic_ml(tmp_path):
    """validation.enabled is a no-op for RandomForest (no epochs); it still runs."""
    from expr_movements.config import ValidationConfig

    processed = tmp_path / "processed"
    processed.mkdir()
    _write_synthetic_npz(processed / "sequences.npz")

    cfg = _make_cfg(processed)
    cfg = cfg.model_copy(update={"validation": ValidationConfig(enabled=True)})
    run_dir = run_training(cfg, outputs_root=tmp_path / "outputs")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    # The run completes; folds note that ES did not apply to classic ML.
    assert metrics["folds"]
    for fold in metrics["folds"]:
        assert fold["validation"]["used"] is False


def test_intra_subject_protocol_runs(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_synthetic_npz(processed / "sequences.npz")

    cfg = _make_cfg(processed, strategy="intra_subject")
    cfg = cfg.model_copy(update={"split": cfg.split.model_copy(update={"n_splits": 3})})
    run_dir = run_training(cfg, outputs_root=tmp_path / "outputs")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["folds"]  # produced folds
    # metadata records the protocol so a run is self-describing
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["split_strategy"] == "intra_subject"

def test_expert_model_broadcasts_clip_features_to_windows(tmp_path):
    """Expert models use clip-level features repeated for each source window."""

    import pandas as pd

    import expr_movements.models.classic  # noqa: F401
    from expr_movements.data.windows import SequenceDataset, make_windows
    from expr_movements.models.registry import build_model
    from expr_movements.train import EXPERT_FEATURE_COLUMNS, _model_X

    processed = tmp_path / "processed"
    processed.mkdir()
    _write_synthetic_npz(processed / "sequences.npz", n_per_subject=1, length=25)

    ds = SequenceDataset.load(processed / "sequences.npz")
    features = pd.DataFrame(
        {
            "clip_idx": np.arange(len(ds)),
            "clip": ds.clips.astype(str),
            "subject": ds.subjects.astype(str),
            "label": ds.labels.astype(str),
            "walking_speed": np.arange(len(ds), dtype=np.float32),
            "stride_length_proxy": np.arange(len(ds), dtype=np.float32) + 100,
            "arm_swing_mean": np.arange(len(ds), dtype=np.float32) + 200,
            "head_vertical_range": np.arange(len(ds), dtype=np.float32) + 300,
        }
    )
    features_path = processed / "features.parquet"
    features.to_parquet(features_path)

    model = build_model(
        "random_forest_expert",
        n_estimators=10,
        random_state=0,
        features_path=str(features_path),
    )

    ws = make_windows(ds, np.array([0, 1]), length=20, stride=10)
    x = _model_X(model, ws)

    expected = (
        features.set_index("clip_idx")
        .loc[ws.clip_idx, list(EXPERT_FEATURE_COLUMNS)]
        .to_numpy(dtype=np.float32)
    )

    assert x.shape == (len(ws.clip_idx), 4)
    np.testing.assert_allclose(x, expected)
