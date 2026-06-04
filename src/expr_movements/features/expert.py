"""Expert / hand-crafted features for approach A (Phase 9, #20).

These are interpretable gait descriptors computed per clip from the **already
invariant** pose sequences in ``sequences.npz`` (pelvis-local + yaw-aligned, with
a trailing world-space speed channel — see :mod:`features.preprocess`). That
store is the common contract the expert-feature team consumes, so nothing here
re-reads raw TRC or re-normalizes.

Four features, matching the design doc's mathematical definitions:

``walking_speed``
    Mean of the per-frame body speed ``s(t)`` (the trailing speed channel),
    i.e. the average pelvis-centroid translation speed in world units/frame.
    Tracks arousal.

``stride_length_proxy``
    Mean of the left/right heel marker (``LHEE``/``RHEE``) travel **range along
    the walking direction**. After yaw alignment the heading is fixed, so the
    forward axis is a stable coordinate; the per-clip max-minus-min of a heel's
    forward coordinate approximates how far it swings each stride.

``arm_swing_mean``
    Same range statistic for the wrist markers (``LWRA``/``RWRA``) along the
    walking direction — captures arm-swing amplitude (suppressed when sad,
    exaggerated when happy/angry).

``head_vertical_range``
    Vertical (up-axis) travel range of the head-marker centroid
    (``LFHD/RFHD/LBHD/RBHD``) — head bob amplitude, an energy/arousal cue. In
    pelvis-local coordinates this is the head's vertical motion *relative to the
    pelvis*, not absolute height.

Axis convention. The bundled TRC is Y-up (``up_axis=1``). Yaw alignment fixes the
pelvis left->right axis to one horizontal axis, leaving the *other* horizontal
axis as the walking/forward direction. :func:`forward_axis` derives it from the
up axis (for Y-up: vertical=1, forward=2).

All reductions are NaN-robust (occluded markers stay NaN through preprocessing);
a fully occluded marker yields NaN for its feature rather than crashing.
"""

from __future__ import annotations

import numpy as np

# Bare marker tokens (the suffix after the last "_"; real names carry a
# per-subject prefix like ``NABA_LHEE``) used by the features.
HEEL_MARKERS: tuple[str, ...] = ("LHEE", "RHEE")
WRIST_MARKERS: tuple[str, ...] = ("LWRA", "RWRA")
HEAD_MARKERS: tuple[str, ...] = ("LFHD", "RFHD", "LBHD", "RBHD")

# Column order in the parquet table (metadata columns are prepended elsewhere).
FEATURE_NAMES: tuple[str, ...] = (
    "walking_speed",
    "stride_length_proxy",
    "arm_swing_mean",
    "head_vertical_range",
)


def _bare(name: str) -> str:
    """Marker token without its subject prefix: ``"NABA_LHEE" -> "LHEE"``."""
    return name.rsplit("_", 1)[-1]


def token_index(marker_names: list[str]) -> dict[str, int]:
    """Map each bare marker token to its index in ``marker_names``.

    Tokens are unique in this rig, so the map is unambiguous. Used to locate a
    marker's XYZ columns in the flattened ``(T, n_markers*3[+1])`` sequence.
    """
    return {_bare(n): i for i, n in enumerate(marker_names)}


def forward_axis(up_axis: int) -> int:
    """Walking-direction axis index, given the vertical axis.

    Yaw alignment pins the pelvis left->right axis to the *first* horizontal axis
    and leaves the *second* horizontal axis as the forward/walking direction.
    For Y-up (``up_axis=1``): horizontal axes are (0, 2), forward is 2.
    """
    plane = [a for a in (0, 1, 2) if a != up_axis]
    return plane[1]


def _marker_axis(seq: np.ndarray, idx: int, axis: int) -> np.ndarray:
    """The ``axis`` coordinate of marker ``idx`` across frames: ``(T,)``.

    The sequence is flattened so marker ``m`` occupies columns
    ``[3m, 3m+1, 3m+2]`` for X, Y, Z.
    """
    return seq[:, 3 * idx + axis]


def _range(values: np.ndarray) -> float:
    """NaN-robust max-minus-min of a 1-D series; NaN if every entry is NaN."""
    if not np.any(np.isfinite(values)):
        return float("nan")
    with np.errstate(invalid="ignore"):
        return float(np.nanmax(values) - np.nanmin(values))


def walking_speed(seq: np.ndarray, has_speed: bool) -> float:
    """Mean per-frame body speed ``mean_t s(t)`` from the trailing speed channel.

    Requires the sequence to carry the speed channel (``has_speed_channel`` in
    ``sequences.npz``); pelvis-local coordinates alone cannot recover world speed
    (centring zeroes it). Raises if the channel is absent so the failure is loud.
    """
    if not has_speed:
        raise ValueError(
            "walking_speed needs the body-speed channel; rebuild sequences with "
            "keep_speed=True (has_speed_channel must be True)"
        )
    speed = seq[:, -1]
    return float(np.nanmean(speed)) if np.any(np.isfinite(speed)) else float("nan")


def _markers_axis_range_mean(
    seq: np.ndarray, tok2idx: dict[str, int], tokens: tuple[str, ...], axis: int
) -> float:
    """Mean over ``tokens`` of each marker's travel range along ``axis``."""
    ranges = [_range(_marker_axis(seq, tok2idx[t], axis)) for t in tokens]
    return float(np.nanmean(ranges)) if np.any(np.isfinite(ranges)) else float("nan")


def stride_length_proxy(seq: np.ndarray, tok2idx: dict[str, int], up_axis: int) -> float:
    """Mean L/R heel travel range along the walking direction."""
    return _markers_axis_range_mean(seq, tok2idx, HEEL_MARKERS, forward_axis(up_axis))


def arm_swing_mean(seq: np.ndarray, tok2idx: dict[str, int], up_axis: int) -> float:
    """Mean L/R wrist travel range along the walking direction."""
    return _markers_axis_range_mean(seq, tok2idx, WRIST_MARKERS, forward_axis(up_axis))


def head_vertical_range(seq: np.ndarray, tok2idx: dict[str, int], up_axis: int) -> float:
    """Vertical travel range of the head-marker centroid (head bob amplitude)."""
    cols = [3 * tok2idx[t] + up_axis for t in HEAD_MARKERS]
    with np.errstate(invalid="ignore"):
        head_y = np.nanmean(seq[:, cols], axis=1)  # (T,) per-frame head centroid height
    return _range(head_y)


def compute_features(
    seq: np.ndarray,
    marker_names: list[str],
    *,
    has_speed: bool,
    up_axis: int = 1,
) -> dict[str, float]:
    """All expert features for one clip's ``(T, F)`` sequence, keyed by name.

    ``seq`` is the normalized, flattened per-clip array from ``sequences.npz``
    (markers in ``marker_names`` order, optional trailing speed channel).
    """
    tok2idx = token_index(marker_names)
    return {
        "walking_speed": walking_speed(seq, has_speed),
        "stride_length_proxy": stride_length_proxy(seq, tok2idx, up_axis),
        "arm_swing_mean": arm_swing_mean(seq, tok2idx, up_axis),
        "head_vertical_range": head_vertical_range(seq, tok2idx, up_axis),
    }
