"""Tests for sliding-window expansion + train-only scaling (Phase 4, #6)."""

from __future__ import annotations

import numpy as np

from expr_movements.data.windows import (
    SequenceDataset,
    apply_scaler,
    fit_scaler,
    flatten_windows,
    make_windows,
)


def _toy_ds(lengths, f=4):
    seqs = np.empty(len(lengths), dtype=object)
    for i, t in enumerate(lengths):
        seqs[i] = (np.arange(t * f).reshape(t, f) + i).astype(np.float32)
    return SequenceDataset(
        sequences=seqs,
        labels=np.array(["a", "b", "a", "b"][: len(lengths)], dtype=object),
        subjects=np.array(["S0", "S1", "S0", "S1"][: len(lengths)], dtype=object),
        clips=np.array([f"c{i}" for i in range(len(lengths))], dtype=object),
        feature_dim=f,
        has_speed_channel=False,
    )


def test_window_shapes_and_clip_tagging():
    ds = _toy_ds([100, 50], f=4)
    ws = make_windows(ds, np.array([0, 1]), length=32, stride=16)
    assert ws.X.ndim == 3 and ws.X.shape[1:] == (32, 4)
    assert ws.X.shape[0] == ws.mask.shape[0] == ws.y.shape[0] == ws.clip_idx.shape[0]
    # clip 1 (T=50) and clip 0 (T=100) both produce windows tagged by clip idx
    assert set(ws.clip_idx.tolist()) == {0, 1}


def test_short_clip_is_padded_not_dropped():
    ds = _toy_ds([10], f=4)  # shorter than window length
    ws = make_windows(ds, np.array([0]), length=32, stride=16)
    assert ws.X.shape == (1, 32, 4)  # exactly one padded window
    assert ws.mask[0, :10].all() and not ws.mask[0, 10:].any()  # real frames masked True


def test_tail_is_always_covered():
    # T=100, length=32, stride=16 -> starts 0,16,32,48,64 and a tail at 68.
    ds = _toy_ds([100], f=4)
    ws = make_windows(ds, np.array([0]), length=32, stride=16)
    # last window must end exactly at the clip end (frame 99 included)
    last = ws.X[-1]
    assert np.array_equal(last, ds.sequences[0][68:100])


def test_scaler_fit_on_train_only_standardises():
    ds = _toy_ds([64, 64], f=4)
    train = make_windows(ds, np.array([0]), length=32, stride=16)
    mean, std = fit_scaler(train)
    scaled = apply_scaler(train, mean, std)
    real = scaled.X[scaled.mask]
    assert np.allclose(real.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(real.std(axis=0), 1, atol=1e-4)
    # padded frames are zeroed
    assert (scaled.X[~scaled.mask] == 0).all()


def test_flatten_windows_dims():
    ds = _toy_ds([64], f=4)
    ws = make_windows(ds, np.array([0]), length=32, stride=16)
    flat = flatten_windows(ws)
    assert flat.shape == (ws.X.shape[0], 32 * 4)
