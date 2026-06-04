"""Tests for the approach-A expert features (``features/expert.py``).

The features run on the *already normalized* flattened sequence layout from
``sequences.npz``: marker ``m`` occupies flat columns ``[3m, 3m+1, 3m+2]`` for
X/Y/Z, with an optional trailing world-speed column. Synthetic clips with known
coordinate ranges pin each feature's value exactly, independent of the bundled
data. The bundled TRC is Y-up, so ``up_axis=1`` and the forward/walking axis is
Z (index 2) — see :func:`forward_axis`.

Feature definitions (after the PR #27 integration):
* stride = range of the fore-aft separation between the two ankles ``LANK-RANK``;
* arm swing = mean L/R range of (mean wrist ``WRA/WRB`` minus same-side shoulder
  ``SHO``) along the forward axis.
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
# touch: ankles, wrists, shoulders, the four head markers, plus one unused filler.
MARKERS = [
    "S_LANK",
    "S_RANK",
    "S_LWRA",
    "S_LWRB",
    "S_RWRA",
    "S_RWRB",
    "S_LSHO",
    "S_RSHO",
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
    assert idx["LANK"] == 0
    assert idx["RBHD"] == 11


def test_walking_speed_is_mean_of_speed_channel():
    seq = _blank(5)
    seq[:, -1] = np.array([0.0, 2.0, 4.0, 6.0, 8.0])  # mean 4.0
    assert walking_speed(seq, has_speed=True) == pytest.approx(4.0)


def test_walking_speed_requires_speed_channel():
    seq = _blank(5, with_speed=False)
    with pytest.raises(ValueError, match="speed"):
        walking_speed(seq, has_speed=False)


def test_stride_length_proxy_is_range_of_ankle_separation():
    idx = token_index(MARKERS)
    seq = _blank(3)
    # L_fwd - R_fwd over frames: [0, 10, 20] -> range 20.
    _set(seq, idx["LANK"], FWD, np.array([0.0, 5.0, 10.0]))
    _set(seq, idx["RANK"], FWD, np.array([0.0, -5.0, -10.0]))
    assert stride_length_proxy(seq, idx, UP) == pytest.approx(20.0)


def test_stride_is_drift_robust_common_mode_cancels():
    # Both ankles drifting forward together must NOT inflate stride.
    idx = token_index(MARKERS)
    seq = _blank(3)
    drift = np.array([0.0, 50.0, 100.0])
    _set(seq, idx["LANK"], FWD, drift + np.array([0.0, 2.0, 0.0]))
    _set(seq, idx["RANK"], FWD, drift)  # difference range = 2, not 100
    assert stride_length_proxy(seq, idx, UP) == pytest.approx(2.0)


def test_arm_swing_is_shoulder_relative_mean():
    idx = token_index(MARKERS)
    seq = _blank(3)
    # Left wrist (mean WRA/WRB) - shoulder range 4; right range 6 -> mean 5.
    _set(seq, idx["LWRA"], FWD, np.array([0.0, 4.0, 0.0]))
    _set(seq, idx["LWRB"], FWD, np.array([0.0, 4.0, 0.0]))  # mean = [0,4,0]
    _set(seq, idx["RWRA"], FWD, np.array([0.0, 6.0, 0.0]))
    _set(seq, idx["RWRB"], FWD, np.array([0.0, 6.0, 0.0]))  # mean = [0,6,0]
    # shoulders stay at 0 -> swing = wrist range
    assert arm_swing_mean(seq, idx, UP) == pytest.approx(5.0)


def test_arm_swing_subtracts_shoulder_motion():
    # Wrist moving in lockstep with the shoulder -> zero relative swing.
    idx = token_index(MARKERS)
    seq = _blank(3)
    common = np.array([0.0, 10.0, 0.0])
    for tok in ("LWRA", "LWRB", "LSHO", "RWRA", "RWRB", "RSHO"):
        _set(seq, idx[tok], FWD, common)
    assert arm_swing_mean(seq, idx, UP) == pytest.approx(0.0)


def test_head_vertical_range_uses_centroid_height():
    idx = token_index(MARKERS)
    seq = _blank(3)
    # Same vertical series on each head marker -> centroid range == 3.
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
    _set(seq, idx["LANK"], FWD, np.full(4, np.nan))
    _set(seq, idx["RANK"], FWD, np.full(4, np.nan))
    assert np.isnan(stride_length_proxy(seq, idx, UP))


def test_partial_occlusion_is_nan_robust():
    # Some frames of one ankle occluded -> range from the finite frames only.
    idx = token_index(MARKERS)
    seq = _blank(4)
    _set(seq, idx["LANK"], FWD, np.array([0.0, np.nan, 4.0, 8.0]))
    _set(seq, idx["RANK"], FWD, np.zeros(4))  # diff = [0, nan, 4, 8] -> range 8
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
