"""Tests for the approach-A expert features (``features/expert.py``).

The features run on the *already normalized* flattened sequence layout from
``sequences.npz``: marker ``m`` occupies flat columns ``[3m, 3m+1, 3m+2]`` for
X/Y/Z, with an optional trailing world-speed column. Synthetic clips with known
coordinate ranges pin each feature's value exactly, independent of the bundled
data. The bundled TRC is Y-up, so ``up_axis=1`` and the forward/walking axis is
Z (index 2) — see :func:`forward_axis`.
"""

from __future__ import annotations

import numpy as np
import pytest

from expr_movements.features.expert import (
    FEATURE_NAMES,
    arm_swing_mean,
    compute_features,
    forward_axis,
    head_vertical_range,
    stride_length_proxy,
    token_index,
    walking_speed,
)

# Marker order (prefixed like the real data) covering every token the features
# touch: heels, wrists, the four head markers, plus one unused filler.
MARKERS = [
    "S_LHEE",
    "S_RHEE",
    "S_LWRA",
    "S_RWRA",
    "S_LFHD",
    "S_RFHD",
    "S_LBHD",
    "S_RBHD",
    "S_STRN",  # unused filler
]
N = len(MARKERS)
UP = 1  # Y-up, matching the bundled TRC
FWD = 2  # forward/walking axis for Y-up


def _set(seq: np.ndarray, idx: int, axis: int, values: np.ndarray) -> None:
    """Write a per-frame series into marker ``idx``'s ``axis`` flat column."""
    seq[:, 3 * idx + axis] = values


def _blank(t: int, *, with_speed: bool = True) -> np.ndarray:
    """Zeroed ``(t, N*3[+1])`` sequence in the flattened layout."""
    f = N * 3 + (1 if with_speed else 0)
    return np.zeros((t, f), dtype=np.float64)


def test_forward_axis_is_second_horizontal():
    assert forward_axis(1) == 2  # Y-up -> forward is Z
    assert forward_axis(2) == 1  # Z-up -> forward is Y
    assert forward_axis(0) == 2  # X-up -> forward is Z


def test_token_index_strips_prefix():
    idx = token_index(MARKERS)
    assert idx["LHEE"] == 0
    assert idx["RBHD"] == 7


def test_walking_speed_is_mean_of_speed_channel():
    seq = _blank(5)
    seq[:, -1] = np.array([0.0, 2.0, 4.0, 6.0, 8.0])  # mean 4.0
    assert walking_speed(seq, has_speed=True) == pytest.approx(4.0)


def test_walking_speed_requires_speed_channel():
    seq = _blank(5, with_speed=False)
    with pytest.raises(ValueError, match="speed"):
        walking_speed(seq, has_speed=False)


def test_stride_length_proxy_is_mean_heel_forward_range():
    idx = token_index(MARKERS)
    seq = _blank(4)
    # LHEE swings 0..10 along forward (range 10); RHEE 0..20 (range 20).
    _set(seq, idx["LHEE"], FWD, np.array([0.0, 5.0, 10.0, 5.0]))
    _set(seq, idx["RHEE"], FWD, np.array([0.0, 10.0, 20.0, 10.0]))
    assert stride_length_proxy(seq, idx, UP) == pytest.approx(15.0)  # (10+20)/2


def test_arm_swing_mean_is_mean_wrist_forward_range():
    idx = token_index(MARKERS)
    seq = _blank(3)
    _set(seq, idx["LWRA"], FWD, np.array([0.0, 4.0, 0.0]))  # range 4
    _set(seq, idx["RWRA"], FWD, np.array([0.0, 6.0, 0.0]))  # range 6
    assert arm_swing_mean(seq, idx, UP) == pytest.approx(5.0)


def test_head_vertical_range_uses_centroid_height():
    idx = token_index(MARKERS)
    seq = _blank(3)
    # Give each head marker the same vertical series so the centroid range == 3;
    # horizontal coords are irrelevant.
    for tok in ("LFHD", "RFHD", "LBHD", "RBHD"):
        _set(seq, idx[tok], UP, np.array([1.0, 4.0, 2.0]))  # range 3
    assert head_vertical_range(seq, idx, UP) == pytest.approx(3.0)


def test_forward_axis_ignored_by_head_feature():
    # Motion only along the forward axis must NOT register as head bob.
    idx = token_index(MARKERS)
    seq = _blank(3)
    for tok in ("LFHD", "RFHD", "LBHD", "RBHD"):
        _set(seq, idx[tok], FWD, np.array([0.0, 100.0, 0.0]))
    assert head_vertical_range(seq, idx, UP) == pytest.approx(0.0)


def test_fully_occluded_marker_yields_nan_not_crash():
    idx = token_index(MARKERS)
    seq = _blank(4)
    _set(seq, idx["LHEE"], FWD, np.full(4, np.nan))
    _set(seq, idx["RHEE"], FWD, np.full(4, np.nan))
    assert np.isnan(stride_length_proxy(seq, idx, UP))


def test_partial_occlusion_is_nan_robust():
    # One heel occluded, the other present -> mean over the finite one.
    idx = token_index(MARKERS)
    seq = _blank(4)
    _set(seq, idx["LHEE"], FWD, np.full(4, np.nan))
    _set(seq, idx["RHEE"], FWD, np.array([0.0, 8.0, 4.0, 0.0]))  # range 8
    assert stride_length_proxy(seq, idx, UP) == pytest.approx(8.0)


def test_compute_features_returns_all_named_features():
    seq = _blank(5)
    seq[:, -1] = 3.0  # constant speed -> mean 3.0
    feats = compute_features(seq, MARKERS, has_speed=True, up_axis=UP)
    assert set(feats) == set(FEATURE_NAMES)
    assert feats["walking_speed"] == pytest.approx(3.0)
    # Static markers -> zero ranges for the other three.
    assert feats["stride_length_proxy"] == pytest.approx(0.0)
    assert feats["arm_swing_mean"] == pytest.approx(0.0)
    assert feats["head_vertical_range"] == pytest.approx(0.0)
