"""Tests for dataset build: manifest (JSONL), sequence npz, and onset trimming.

Small synthetic interim ``.npz`` clips (matching ``expr-parse`` output) exercise
manifest fields, the variable-length + dense sequence stores, and the
speed-based active-window detection — without depending on the bundled motion
data.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from expr_movements.config import DataConfig
from expr_movements.data.dataset import (
    active_window,
    build_manifest,
    build_sequence_dataset,
    trim_clip,
)


def _write_interim(path, frames, *, subject="SUBA", code="COE", emotion="angry", take=1):
    """Write a synthetic interim clip npz like ``expr-parse`` produces."""
    n_markers = frames.shape[1]
    np.savez_compressed(
        path,
        frames=frames.astype(np.float64),
        times=np.arange(frames.shape[0], dtype=np.float64) / 60.0,
        marker_names=np.asarray([f"M{i}" for i in range(n_markers)], dtype=object),
        frame_rate=60.0,
        subject=subject,
        emotion_code=code,
        emotion=emotion,
        take=take,
    )


def _moving_clip(n_static_pre, n_move, n_static_post, n_markers=2):
    """A clip: static, then linearly translating (walking), then static again."""
    static_pre = np.zeros((n_static_pre, n_markers, 3))
    # Move: each frame shifts by a constant step so inter-frame speed is high.
    steps = np.arange(1, n_move + 1).reshape(-1, 1, 1)
    move = steps * np.ones((1, n_markers, 3)) * 10.0
    last = move[-1] if n_move else np.zeros((n_markers, 3))
    static_post = np.broadcast_to(last, (n_static_post, n_markers, 3))
    return np.concatenate([static_pre, move, static_post], axis=0)


@pytest.fixture
def interim_dir(tmp_path):
    d = tmp_path / "interim"
    d.mkdir()
    # Two clips, different lengths, with leading/trailing static frames.
    _write_interim(
        d / "SUBACOE01.4.npz", _moving_clip(4, 6, 5), subject="SUBA", code="COE", emotion="angry"
    )
    _write_interim(
        d / "SUBBJOE02.4.npz",
        _moving_clip(2, 10, 3),
        subject="SUBB",
        code="JOE",
        emotion="happy",
        take=2,
    )
    return d


def test_active_window_drops_static_ends():
    frames = _moving_clip(4, 6, 5)  # static[0:4], move[4:10], static[10:15]
    start, stop = active_window(frames, speed_frac=0.1, min_run=2)
    # Onset is where movement begins; offset just past the last moving frame.
    assert 3 <= start <= 5
    assert 9 <= stop <= 11
    assert stop - start < frames.shape[0]  # something was trimmed


def test_active_window_static_clip_keeps_all():
    frames = np.zeros((10, 2, 3))
    assert active_window(frames, speed_frac=0.1, min_run=3) == (0, 10)


def test_trim_clip_applies_fixed_margin_inside_window():
    frames = _moving_clip(4, 8, 4)
    cfg = DataConfig(detect_onset=True, onset_min_run=2, trim_start_frames=1, trim_end_frames=1)
    start, stop = trim_clip(frames, cfg)
    base_start, base_stop = active_window(frames, cfg.onset_speed_frac, cfg.onset_min_run)
    assert start == base_start + 1
    assert stop == base_stop - 1


def test_trim_clip_never_empties_short_clip():
    frames = _moving_clip(0, 3, 0)
    cfg = DataConfig(detect_onset=True, onset_min_run=2, trim_start_frames=10, trim_end_frames=10)
    start, stop = trim_clip(frames, cfg)
    assert stop > start  # margins skipped rather than emptying the clip


def test_build_manifest_fields(interim_dir, tmp_path):
    out = tmp_path / "processed" / "manifest.jsonl"
    cfg = DataConfig(onset_min_run=2)
    build_manifest(interim_dir, out, cfg=cfg)

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 2
    rec = next(r for r in records if r["clip"] == "SUBACOE01.4.npz")
    assert rec["subject"] == "SUBA"
    assert rec["emotion"] == "angry"
    assert rec["emotion_code"] == "COE"
    assert rec["take"] == 1
    assert rec["n_frames_raw"] == 15
    assert rec["n_frames"] == rec["trim_stop"] - rec["trim_start"]
    assert 0 < rec["n_frames"] <= rec["n_frames_raw"]


def test_build_manifest_raises_on_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="run expr-parse"):
        build_manifest(empty, tmp_path / "m.jsonl")


def test_build_sequence_dataset_variable_length(interim_dir, tmp_path):
    out_dir = tmp_path / "processed"
    manifest = out_dir / "manifest.jsonl"
    build_manifest(interim_dir, manifest, cfg=DataConfig(onset_min_run=2))
    npz = build_sequence_dataset(manifest, out_dir / "sequences.npz")

    data = np.load(npz, allow_pickle=True)
    seqs = data["sequences"]
    assert seqs.shape == (2,)
    # Variable length: each clip's own (T_i, 41*3-equivalent) array; 2 markers -> 6 cols.
    assert seqs[0].shape[1] == 6
    assert all(s.shape[0] == length for s, length in zip(seqs, data["lengths"]))
    assert set(data["labels"]) == {"angry", "happy"}
    assert set(data["subjects"]) == {"SUBA", "SUBB"}
    assert "sequences_dense" not in data.files  # dense only on request


def test_build_sequence_dataset_dense(interim_dir, tmp_path):
    out_dir = tmp_path / "processed"
    manifest = out_dir / "manifest.jsonl"
    build_manifest(interim_dir, manifest, cfg=DataConfig(onset_min_run=2))
    npz = build_sequence_dataset(manifest, out_dir / "sequences.npz", target_frames=8)

    data = np.load(npz, allow_pickle=True)
    dense = data["sequences_dense"]
    mask = data["mask"]
    assert dense.shape == (2, 8, 6)
    assert mask.shape == (2, 8)
    # Mask marks exactly the real frames per clip (capped at target_frames).
    for i, length in enumerate(data["lengths"]):
        assert mask[i].sum() == min(int(length), 8)
    # Padded frames are zero where the mask is False.
    assert np.all(dense[~mask] == 0)
