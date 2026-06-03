"""Tests for Phase-3 invariant pre-processing (``features/preprocess.py``).

The contract: pelvis-local + yaw-aligned pose is invariant to where the subject
walked (translation) and which way they faced (yaw rotation), while the speed
channel still reflects real walking speed. Synthetic clips with known pelvis
geometry exercise each property without depending on the bundled data.
"""

from __future__ import annotations

import numpy as np
import pytest

from expr_movements.config import DataConfig
from expr_movements.features.preprocess import (
    body_speed,
    normalize_sequence,
    pelvis_indices,
)

# Marker order: 4 pelvis markers (prefixed, like the real data) + one extra
# "head" marker so we can check non-pelvis markers move with the body frame.
MARKERS = ["SUB_LFWT", "SUB_RFWT", "SUB_LBWT", "SUB_RBWT", "SUB_HEAD"]
UP = 1  # Y-up, matching the bundled TRC


def _frame(center, *, heading=0.0, head_height=2.0):
    """One frame: a unit-square pelvis around ``center`` facing ``heading`` (yaw).

    Pelvis layout in the X-Z (horizontal) plane, Y up:
      LFWT(-0.5,+0.5) RFWT(+0.5,+0.5) LBWT(-0.5,-0.5) RBWT(+0.5,-0.5)
    so right-left axis is +X and front-back is +Z before rotation. ``heading``
    rotates the body about the up axis. HEAD sits above the centroid.
    """
    base = np.array(
        [
            [-0.5, 0.0, 0.5],  # LFWT
            [0.5, 0.0, 0.5],  # RFWT
            [-0.5, 0.0, -0.5],  # LBWT
            [0.5, 0.0, -0.5],  # RBWT
            [0.0, head_height, 0.0],  # HEAD
        ]
    )
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])  # yaw about Y
    return base @ rot.T + np.asarray(center)


def _clip(centers, headings):
    return np.stack([_frame(c, heading=h) for c, h in zip(centers, headings)])


def test_pelvis_indices_match_by_bare_token():
    idx = pelvis_indices(MARKERS, ("LFWT", "RFWT", "LBWT", "RBWT"))
    assert idx == [0, 1, 2, 3]  # the HEAD marker is excluded


def test_pelvis_indices_raises_when_absent():
    with pytest.raises(ValueError, match="no pelvis markers"):
        pelvis_indices(["SUB_HEAD", "SUB_NOSE"], ("LFWT", "RFWT", "LBWT", "RBWT"))


def test_translation_invariance():
    """Two clips identical but for an absolute position offset normalize equal."""
    cfg = DataConfig(yaw_align=True, keep_speed=False)
    headings = [0.0, 0.0, 0.0]
    a = _clip([(0, 0, 0), (0, 0, 1), (0, 0, 2)], headings)
    b = _clip([(10, 5, -3), (10, 5, -2), (10, 5, -1)], headings)  # same motion, shifted
    na = normalize_sequence(a, MARKERS, cfg)
    nb = normalize_sequence(b, MARKERS, cfg)
    assert np.allclose(na, nb, atol=1e-5)


def test_yaw_invariance():
    """A clip and a yaw-rotated copy normalize to the same pose."""
    cfg = DataConfig(yaw_align=True, keep_speed=False)
    centers = [(0, 0, 0), (0, 0, 1)]
    a = _clip(centers, [0.0, 0.0])
    b = _clip(centers, [0.9, 0.9])  # whole body faces a different heading
    na = normalize_sequence(a, MARKERS, cfg)
    nb = normalize_sequence(b, MARKERS, cfg)
    assert np.allclose(na, nb, atol=1e-5)


def test_pitch_is_preserved():
    """Yaw alignment must NOT flatten trunk lean (pitch) — it carries posture."""
    cfg = DataConfig(yaw_align=True, keep_speed=False)
    upright = _clip([(0, 0, 0)], [0.0])
    # Tilt the head forward (in Z) — a vertical-plane change yaw must keep.
    leaned = upright.copy()
    leaned[0, 4] = [0.0, 1.5, 1.0]  # HEAD pushed forward and lower
    n_up = normalize_sequence(upright, MARKERS, cfg)
    n_lean = normalize_sequence(leaned, MARKERS, cfg)
    assert not np.allclose(n_up, n_lean)  # the lean survives normalization


def test_speed_channel_appended_and_matches_world_speed():
    cfg = DataConfig(yaw_align=True, keep_speed=True)
    # Centroid moves 0, 1, 1 units between frames (constant step of 1 in Z).
    clip = _clip([(0, 0, 0), (0, 0, 1), (0, 0, 2)], [0.0, 0.0, 0.0])
    out = normalize_sequence(clip, MARKERS, cfg)
    # 5 markers * 3 + 1 speed column.
    assert out.shape == (3, 5 * 3 + 1)
    speed = out[:, -1]
    assert np.allclose(speed, [0.0, 1.0, 1.0], atol=1e-5)


def test_speed_invariant_to_translation_offset():
    """Speed reads the trajectory, so an absolute offset doesn't change it."""
    cfg_idx = pelvis_indices(MARKERS, DataConfig().pelvis_markers)
    a = _clip([(0, 0, 0), (0, 0, 1), (0, 0, 3)], [0.0, 0.0, 0.0])
    b = _clip([(100, 0, 0), (100, 0, 1), (100, 0, 3)], [0.0, 0.0, 0.0])
    assert np.allclose(body_speed(a, cfg_idx), body_speed(b, cfg_idx))
    assert np.allclose(body_speed(a, cfg_idx), [0.0, 1.0, 2.0])


def test_normalize_off_returns_raw_flattened():
    cfg = DataConfig(normalize=False)
    clip = _clip([(3, 0, 0), (3, 0, 1)], [0.0, 0.0])
    out = normalize_sequence(clip, MARKERS, cfg)
    assert out.shape == (2, 5 * 3)  # no speed column when normalize is off
    assert np.allclose(out, clip.reshape(2, -1))  # untouched world coordinates


def test_occluded_pelvis_marker_still_normalizes():
    """A NaN on one pelvis marker must not poison the centroid (NaN-robust mean)."""
    cfg = DataConfig(yaw_align=True, keep_speed=True)
    clip = _clip([(5, 0, 0), (5, 0, 1)], [0.0, 0.0])
    clip[0, 0] = np.nan  # occlude LFWT on frame 0
    out = normalize_sequence(clip, MARKERS, cfg)
    # HEAD (index 4) is fully observed -> its normalized coords stay finite.
    head_cols = out[:, 4 * 3 : 4 * 3 + 3]
    assert np.all(np.isfinite(head_cols))
