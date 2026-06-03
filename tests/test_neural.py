"""Tests for the multi-task NN (Phase 5, #14).

Covers the checklist's three asks at the unit level — forward/backward, latent
extraction, and end-to-end LOSO training through the shared harness — plus the
speed-concat and encoder-swap requirements. Networks are kept tiny (few epochs,
small latent) so the suite stays fast; correctness, not accuracy, is asserted.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from expr_movements.config import ExperimentConfig, ModelConfig, SplitConfig, WindowConfig
from expr_movements.data.windows import WindowSet
from expr_movements.models.registry import build_model, registered_names
from expr_movements.train import run_training


def _learnable_windows(n_per_class=12, length=20, n_pose=6, with_speed=True, seed=0):
    """Two classes as constant offsets (trivially learnable) + a speed channel.

    Returns a :class:`WindowSet` with ``has_speed`` set, mirroring what the
    harness builds from ``sequences.npz``. The speed column differs by class too,
    so the model can use the concatenated speed feature.
    """
    rng = np.random.default_rng(seed)
    f = n_pose + (1 if with_speed else 0)
    xs, ys = [], []
    for cls, off in (("calm", 0.0), ("excited", 4.0)):
        for _ in range(n_per_class):
            w = off + rng.normal(0, 0.1, size=(length, f)).astype(np.float32)
            xs.append(w)
            ys.append(cls)
    X = np.stack(xs)
    y = np.asarray(ys, dtype=object)
    mask = np.ones((len(X), length), dtype=bool)
    return WindowSet(X=X, mask=mask, y=y, clip_idx=np.arange(len(X)), has_speed=with_speed)


@pytest.mark.parametrize("encoder", ["cnn1d", "lstm", "gru"])
def test_encoders_registered_and_swappable(encoder):
    """Each encoder is reachable via the registry (CNN headline + RNN baselines)."""
    assert encoder in registered_names()
    model = build_model(encoder, epochs=1, latent_dim=4, hidden_size=8, batch_size=8)
    assert model.encoder == encoder
    assert model.consumes == "window"


def test_fit_predict_learns_and_backprops():
    """forward + backward: a few epochs on a separable signal -> perfect train acc.

    A non-trivial fit (acc > chance) proves gradients flowed (backward worked) and
    the multi-task loss didn't NaN out.
    """
    ws = _learnable_windows()
    model = build_model("cnn1d", epochs=30, latent_dim=8, hidden_size=16, batch_size=8)
    model.fit(ws, ws.y)
    pred = model.predict(ws)
    assert set(pred) <= {"calm", "excited"}
    assert (pred == ws.y).mean() == 1.0  # trivially separable


def test_latent_extraction_shape_and_speed_concat():
    """``transform``/``encode`` return ``(n, latent_dim + 1)`` when speed is present.

    The +1 is the concatenated mean-speed channel — the checklist's "speed into
    the pooled latent" requirement. ``encode`` is an alias for ``transform``.
    """
    ws = _learnable_windows(with_speed=True)
    latent_dim = 8
    model = build_model("cnn1d", epochs=2, latent_dim=latent_dim, hidden_size=16, batch_size=8)
    model.fit(ws, ws.y)

    z = model.transform(ws)
    assert z.shape == (len(ws.X), latent_dim + 1)  # +1 speed channel
    assert np.isfinite(z).all()
    # encode is the same callable -> identical output.
    np.testing.assert_array_equal(z, model.encode(ws))


def test_no_speed_channel_latent_dim():
    """Without a speed channel the latent is exactly ``latent_dim`` (no concat)."""
    ws = _learnable_windows(with_speed=False)
    latent_dim = 8
    model = build_model("lstm", epochs=2, latent_dim=latent_dim, hidden_size=16, batch_size=8)
    model.fit(ws, ws.y)
    assert model.transform(ws).shape == (len(ws.X), latent_dim)


def test_fit_early_stopping_stops_and_records(monkeypatch):
    """fit(..., validation_data, patience) early-stops and records the stop epoch.

    With a high epoch cap, a small patience, and a validation set, training must
    stop before the cap and expose ``stopped_epoch_`` / ``best_score_``. The
    restored model still predicts the learnable classes.
    """
    train = _learnable_windows(n_per_class=12, seed=0)
    val = _learnable_windows(n_per_class=4, seed=1)
    model = build_model("cnn1d", epochs=200, latent_dim=8, hidden_size=16, batch_size=8)
    model.fit(
        train, train.y,
        validation_data=(val, val.y),
        patience=3, monitor="macro_f1",
    )
    assert model.stopped_epoch_ <= 200  # early stop (or hit cap) was tracked
    assert model.stopped_epoch_ < 200  # separable signal converges well before cap
    assert model.best_score_ is not None
    pred = model.predict(val)
    assert set(pred) <= {"calm", "excited"}


def test_fit_without_validation_runs_all_epochs():
    """Backward compat: fit(X, y) with no validation runs the full epoch count."""
    ws = _learnable_windows()
    model = build_model("cnn1d", epochs=5, latent_dim=4, hidden_size=8, batch_size=8)
    model.fit(ws, ws.y)
    assert model.stopped_epoch_ == 5  # no early stopping -> ran all epochs
    assert model.best_score_ is None  # no validation tracked


def test_fit_early_stopping_monitor_loss():
    """monitor='loss' is accepted (lower-is-better path) and trains/predicts."""
    train = _learnable_windows(n_per_class=10, seed=0)
    val = _learnable_windows(n_per_class=4, seed=2)
    model = build_model("lstm", epochs=30, latent_dim=8, hidden_size=16, batch_size=8)
    model.fit(train, train.y, validation_data=(val, val.y), patience=3, monitor="loss")
    assert model.best_score_ is not None
    assert model.predict(val).shape[0] == len(val.X)


def test_alpha_beta_are_configurable():
    """Reconstruction/classification weights are plain hyperparameters on the model."""
    model = build_model("cnn1d", alpha=0.3, beta=2.0, epochs=1)
    assert model.alpha == 0.3 and model.beta == 2.0
    # Pure-reconstruction (beta=0) still fits without error.
    ws = _learnable_windows()
    build_model("cnn1d", alpha=1.0, beta=0.0, epochs=1, latent_dim=4, hidden_size=8).fit(ws, ws.y)


def _write_synthetic_npz(path, n_per_subject=4, n_pose=6, length=30):
    """4 subjects x 2 classes with a per-class offset + speed channel -> learnable LOSO."""
    subjects = ["A", "B", "C", "D"]
    classes = {"calm": 0.0, "excited": 4.0}
    rng = np.random.default_rng(0)
    seqs, labels, subs, clips = [], [], [], []
    i = 0
    f = n_pose + 1
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
        has_speed_channel=np.asarray(True),
    )


def test_end_to_end_loso_training_through_harness(tmp_path):
    """The NN runs through the same harness as classic ML under LOSO and writes a run dir.

    Asserts the `expr-train ... lstm` requirement: 4 LOSO folds, a picklable saved
    model (the closure-class pickling bug would surface here), and metrics.
    """
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_synthetic_npz(processed / "sequences.npz")

    cfg = ExperimentConfig(
        name="cnn1d_test",
        data={"processed_dir": str(processed)},
        window=WindowConfig(length=20, stride=10),
        split=SplitConfig(strategy="leave_one_subject_out", n_splits=4),
        model=ModelConfig(
            name="cnn1d",
            params={"epochs": 8, "latent_dim": 8, "hidden_size": 16, "batch_size": 16},
        ),
    )
    run_dir = run_training(cfg, outputs_root=tmp_path / "outputs")

    for name in ("config.yaml", "model.joblib", "metrics.json", "metadata.json", "predictions.jsonl"):
        assert (run_dir / name).exists(), name

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert len(metrics["folds"]) == 4

    # Saved NN artifact reloads (module-level torch classes -> picklable).
    bundle = joblib.load(run_dir / "model.joblib")
    assert sorted(bundle["labels"]) == ["calm", "excited"]
    # And the reloaded model still exposes the latent API for downstream PCA.
    ws = _learnable_windows(with_speed=True)
    assert bundle["model"].transform(ws).shape[0] == len(ws.X)
