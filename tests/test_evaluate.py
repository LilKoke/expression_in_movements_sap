"""Phase 6 (#7): evaluation metrics + the run read-out.

Asserts the checklist items that landed in this phase:

* balanced accuracy is reported at window and clip level (mean ± std), alongside
  Macro-F1 / accuracy that already existed;
* the multi-task NN run records held-out reconstruction MSE (overall + per-frame
  + per-joint) and latent separability (Silhouette / Davies-Bouldin), and a
  classic-ML run does not (the metrics are NN-only);
* ``evaluate_run`` loads a real run dir and folds its metrics into the Phase 6
  summary, with the split protocol carried through so intra-subject reads as such.

Networks stay tiny (few epochs) — correctness of the metric plumbing is the
target, not model accuracy.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from expr_movements.config import ExperimentConfig, ModelConfig, SplitConfig, WindowConfig
from expr_movements.evaluate import compare_runs, evaluate_run, format_report
from expr_movements.train import run_training


def _write_synthetic_npz(path, n_per_subject=4, n_pose=6, length=30, with_speed=True):
    """4 subjects x 2 classes, per-class offset (+ optional speed channel)."""
    subjects = ["A", "B", "C", "D"]
    classes = {"calm": 0.0, "excited": 4.0}
    rng = np.random.default_rng(0)
    seqs, labels, subs, clips = [], [], [], []
    i = 0
    f = n_pose + (1 if with_speed else 0)
    for s in subjects:
        for cls, off in classes.items():
            for _ in range(n_per_subject):
                seqs.append((off + rng.normal(0, 0.1, size=(length, f))).astype(np.float32))
                labels.append(cls)
                subs.append(s)
                clips.append(f"{s}_{cls}_{i}.trc")
                i += 1
    obj = np.empty(len(seqs), dtype=object)
    obj[:] = seqs
    np.savez_compressed(
        path,
        sequences=obj,
        labels=np.asarray(labels, dtype=object),
        subjects=np.asarray(subs, dtype=object),
        clips=np.asarray(clips, dtype=object),
        has_speed_channel=np.asarray(with_speed),
    )


def _cfg(processed, *, name, model_name, params, strategy="leave_one_subject_out", n_splits=4):
    return ExperimentConfig(
        name=name,
        data={"processed_dir": str(processed)},
        window=WindowConfig(length=20, stride=10),
        split=SplitConfig(strategy=strategy, n_splits=n_splits),
        model=ModelConfig(name=model_name, params=params),
    )


def _run_rf(tmp_path, **kw):
    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    _write_synthetic_npz(processed / "sequences.npz")
    cfg = _cfg(
        processed,
        name="rf_test",
        model_name="random_forest",
        params={"n_estimators": 20, "random_state": 0},
        **kw,
    )
    return run_training(cfg, outputs_root=tmp_path / "outputs")


def _run_cnn(tmp_path, **kw):
    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    _write_synthetic_npz(processed / "sequences.npz")
    cfg = _cfg(
        processed,
        name="cnn1d_test",
        model_name="cnn1d",
        params={"epochs": 6, "latent_dim": 8, "hidden_size": 16, "batch_size": 16},
        **kw,
    )
    return run_training(cfg, outputs_root=tmp_path / "outputs")


# -- balanced accuracy (both teams) -------------------------------------------


def test_balanced_accuracy_reported_at_both_levels(tmp_path):
    run_dir = _run_rf(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    for level in ("window_level", "clip_level"):
        assert "balanced_accuracy_mean" in metrics[level]
        assert "balanced_accuracy_std" in metrics[level]
        ba = metrics[level]["balanced_accuracy_mean"]
        assert 0.0 <= ba <= 1.0
    # And per fold, next to macro_f1 / accuracy.
    for fold in metrics["folds"]:
        for level in ("window", "clip"):
            assert "balanced_accuracy" in fold[level]


# -- NN-only metrics: reconstruction + separability ---------------------------


def test_nn_run_records_reconstruction_and_separability(tmp_path):
    run_dir = _run_cnn(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert "reconstruction" in metrics
    assert metrics["reconstruction"]["overall_mse_mean"] >= 0.0
    assert "overall_mse_std" in metrics["reconstruction"]

    assert "separability" in metrics
    assert "silhouette" in metrics["separability"]
    assert "davies_bouldin" in metrics["separability"]

    # Per-fold reconstruction carries the per-frame and per-joint breakdowns.
    fold0 = metrics["folds"][0]
    rec = fold0["reconstruction"]
    assert len(rec["per_frame_mse"]) == 20  # window length
    assert "per_joint_mse" in rec  # 6 pose channels -> 2 joints
    assert len(rec["per_joint_mse"]) == 2
    assert "speed_channel_mse" in rec  # speed channel present
    sep = fold0["separability"]
    assert sep["n_classes"] == 2


def test_classic_run_omits_nn_only_metrics(tmp_path):
    run_dir = _run_rf(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "reconstruction" not in metrics
    assert "separability" not in metrics
    for fold in metrics["folds"]:
        assert "reconstruction" not in fold
        assert "separability" not in fold


# -- evaluate_run read-out ----------------------------------------------------


def test_evaluate_run_summary_shape(tmp_path):
    run_dir = _run_cnn(tmp_path)
    summary = evaluate_run(run_dir)

    assert summary["split_strategy"] == "leave_one_subject_out"
    assert summary["n_folds"] == 4
    assert sorted(summary["labels"]) == ["calm", "excited"]
    # Aggregated scores at both levels.
    for level in ("window_level", "clip_level"):
        assert "macro_f1_mean" in summary[level]
        assert "balanced_accuracy_mean" in summary[level]
    # Per-class F1 and a summed confusion matrix at both levels.
    assert set(summary["per_class_f1"]["clip"]) == {"calm", "excited"}
    cm = summary["confusion_matrix"]["clip"]
    assert np.asarray(cm).shape == (2, 2)
    # NN extras passed through.
    assert "reconstruction" in summary
    assert "separability" in summary

    # The markdown report renders without error and mentions the headline metric.
    report = format_report(summary)
    assert "Macro-F1" in report
    assert "leave_one_subject_out" in report


def test_evaluate_run_intra_subject_protocol_carried_through(tmp_path):
    run_dir = _run_rf(tmp_path, strategy="intra_subject", n_splits=3)
    summary = evaluate_run(run_dir)
    assert summary["split_strategy"] == "intra_subject"
    assert summary["n_folds"] >= 1
    # Classic ML -> no NN-only blocks in the summary.
    assert "reconstruction" not in summary
    assert "separability" not in summary


def test_evaluate_run_missing_metrics_raises(tmp_path):
    empty = tmp_path / "empty_run"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        evaluate_run(empty)


def test_compare_runs_is_phase7_stub(tmp_path):
    with pytest.raises(NotImplementedError):
        compare_runs("a", "b", tmp_path / "cmp.md")
