"""Phase 7 (#13): the report figures (latent PCA, confusion, comparison bars).

These assert the plumbing — a NN run's saved model can be reloaded and encoded
into a per-window latent, the PCA scatter and the charts write a PNG, and the
classic-ML run (which has no latent) is rejected for PCA. matplotlib is an
optional ``viz`` extra, so the whole module skips when it's not installed.

The networks stay tiny; correctness of the figure plumbing is the target, not
the look of the plot.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

from expr_movements.cli.report import generate_report
from expr_movements.config import ExperimentConfig, ModelConfig, SplitConfig, WindowConfig
from expr_movements.evaluate import compare_runs
from expr_movements.train import run_training
from expr_movements.viz import (
    latent_pca,
    latent_windows,
    plot_confusion_matrix,
    plot_metric_comparison,
)


def _write_synthetic_npz(path, n_per_subject=4, n_pose=6, length=30, with_speed=True):
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


def _cfg(processed, *, name, model_name, params):
    return ExperimentConfig(
        name=name,
        data={"processed_dir": str(processed)},
        window=WindowConfig(length=20, stride=10),
        split=SplitConfig(strategy="leave_one_subject_out", n_splits=4),
        model=ModelConfig(name=model_name, params=params),
    )


def _run_cnn(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    _write_synthetic_npz(processed / "sequences.npz")
    cfg = _cfg(
        processed,
        name="cnn1d_test",
        model_name="cnn1d",
        params={"epochs": 6, "latent_dim": 8, "hidden_size": 16, "batch_size": 16},
    )
    return run_training(cfg, outputs_root=tmp_path / "outputs")


def _run_rf(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    _write_synthetic_npz(processed / "sequences.npz")
    cfg = _cfg(
        processed,
        name="rf_test",
        model_name="random_forest",
        params={"n_estimators": 20, "random_state": 0},
    )
    return run_training(cfg, outputs_root=tmp_path / "outputs")


def test_latent_windows_shapes_align(tmp_path):
    run_dir = _run_cnn(tmp_path)
    data = latent_windows(run_dir)
    n = len(data["z"])
    assert data["z"].ndim == 2 and n > 0
    # Every per-window tag is aligned to z.
    assert len(data["labels"]) == n
    assert len(data["subjects"]) == n
    assert len(data["clips"]) == n
    assert set(data["labels"]) == {"calm", "excited"}
    assert set(data["subjects"]) == {"A", "B", "C", "D"}


def test_latent_windows_rejects_classic_ml(tmp_path):
    run_dir = _run_rf(tmp_path)
    with pytest.raises(ValueError, match="no latent"):
        latent_windows(run_dir)


def test_latent_pca_writes_png(tmp_path):
    run_dir = _run_cnn(tmp_path)
    out = tmp_path / "figs" / "pca.png"
    path = latent_pca(run_dir, out)
    assert path == out
    assert out.exists() and out.stat().st_size > 0


def test_plot_confusion_matrix_writes_png(tmp_path):
    out = tmp_path / "cm.png"
    cm = [[5, 1], [2, 4]]
    plot_confusion_matrix(cm, ["calm", "excited"], out, title="t")
    assert out.exists() and out.stat().st_size > 0


def test_plot_metric_comparison_writes_png(tmp_path):
    run_a = _run_rf(tmp_path / "a")
    run_b = _run_cnn(tmp_path / "b")
    summary = compare_runs(run_a, run_b)
    out = tmp_path / "bars.png"
    plot_metric_comparison(summary, out, level="clip_level")
    assert out.exists() and out.stat().st_size > 0


def test_generate_report_writes_full_bundle(tmp_path):
    """expr-report's driver writes the reports + every figure into out_dir."""
    run_a = _run_rf(tmp_path / "a")
    run_b = _run_cnn(tmp_path / "b")
    out_dir = generate_report(run_a, run_b, out_dir=tmp_path / "report")

    assert (out_dir / "comparison.md").exists()
    for fig in (
        "compare_clip_level.png",
        "compare_window_level.png",
        "latent_pca.png",
        "confusion_A_expert.png",
        "confusion_B_nn.png",
    ):
        p = out_dir / "figs" / fig
        assert p.exists() and p.stat().st_size > 0, fig
    # No intra run given -> the protocol section is skipped, not errored.
    assert not (out_dir / "protocol_comparison.md").exists()


def test_generate_report_includes_protocol_when_intra_given(tmp_path):
    run_a = _run_rf(tmp_path / "a")
    run_b = _run_cnn(tmp_path / "b")
    # An intra-subject NN run for the gap section.
    processed = tmp_path / "intra"
    processed.mkdir(parents=True, exist_ok=True)
    _write_synthetic_npz(processed / "sequences.npz")
    cfg = _cfg(
        processed,
        name="cnn1d_intra",
        model_name="cnn1d",
        params={"epochs": 6, "latent_dim": 8, "hidden_size": 16, "batch_size": 16},
    )
    cfg = cfg.model_copy(
        update={"split": cfg.split.model_copy(update={"strategy": "intra_subject"})}
    )
    run_b_intra = run_training(cfg, outputs_root=tmp_path / "outputs_intra")

    out_dir = generate_report(run_a, run_b, out_dir=tmp_path / "report", run_b_intra=run_b_intra)
    assert (out_dir / "protocol_comparison.md").exists()
