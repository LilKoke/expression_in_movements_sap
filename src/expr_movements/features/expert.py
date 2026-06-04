"""Expert / hand-crafted features for approach A (Phase 9, #20).

These are interpretable gait descriptors computed per clip from the **already
invariant** pose sequences in ``sequences.npz`` (pelvis-local + yaw-aligned, with
a trailing world-space speed channel — see :mod:`features.preprocess`). That
store is the common contract the expert-feature team consumes, so nothing here
re-reads raw TRC or re-normalizes.

Four features, matching the design doc's definitions (the stride and arm-swing
definitions follow the refinements from PR #27):

``walking_speed``
    Mean of the per-frame body speed ``s(t)`` (the trailing speed channel),
    i.e. the average pelvis-centroid translation speed in world units/frame.
    Tracks arousal.

``stride_length_proxy``
    Range over the clip of the **fore-aft separation between the two ankles**
    (``LANK``/``RANK``) along the walking direction. At mid-stance the ankles are
    together (separation ~0); at full stride one is forward and the other back,
    so the max-minus-min of ``L_forward - R_forward`` tracks stride length. Using
    the left-right *difference* makes it robust to any residual common-mode
    drift.

``arm_swing_mean``
    Mean of the left/right arm-swing amplitudes, each measured as the range of
    the wrist's fore-aft position **relative to the same-side shoulder**. Wrists
    are the mean of ``WRA``/``WRB``; subtracting the shoulder (``SHO``) removes
    trunk sway so the value isolates the pendulum motion of the arm (suppressed
    when sad, exaggerated when happy/angry).

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

# --- Marker tokens (bare; real names carry a per-subject prefix like ``NABA_LANK``). ---
# stride: fore-aft separation between the two ankles.
ANKLE_LEFT = "LANK"
ANKLE_RIGHT = "RANK"
# arm swing: per-side wrist (mean of WRA/WRB) relative to the same-side shoulder.
ARM_LEFT_WRISTS: tuple[str, ...] = ("LWRA", "LWRB")
ARM_RIGHT_WRISTS: tuple[str, ...] = ("RWRA", "RWRB")
ARM_LEFT_SHOULDER = "LSHO"
ARM_RIGHT_SHOULDER = "RSHO"
# head bob: centroid of the four head markers.
HEAD_MARKERS: tuple[str, ...] = ("LFHD", "RFHD", "LBHD", "RBHD")

# Column order in the parquet table (identity columns are prepended elsewhere).
FEATURE_NAMES: tuple[str, ...] = (
    "walking_speed",
    "stride_length_proxy",
    "arm_swing_mean",
    "head_vertical_range",
)


def _bare(name: str) -> str:
    """Marker token without its subject prefix: ``"NABA_LANK" -> "LANK"``."""
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


def _markers_mean_axis(seq: np.ndarray, idxs: list[int], axis: int) -> np.ndarray:
    """NaN-robust mean over markers of their ``axis`` series: ``(T,)``."""
    stacked = np.stack([_marker_axis(seq, i, axis) for i in idxs], axis=0)  # (k, T)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0)


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


def stride_length_proxy(seq: np.ndarray, tok2idx: dict[str, int], up_axis: int) -> float:
    """Range of the fore-aft separation between the two ankles (PR #27 definition)."""
    fwd = forward_axis(up_axis)
    left = _marker_axis(seq, tok2idx[ANKLE_LEFT], fwd)
    right = _marker_axis(seq, tok2idx[ANKLE_RIGHT], fwd)
    return _range(left - right)


def _side_arm_swing(
    seq: np.ndarray, tok2idx: dict[str, int], wrists: tuple[str, ...], shoulder: str, fwd: int
) -> float:
    """One side's arm-swing amplitude: range of (wrist - shoulder) along forward."""
    wrist = _markers_mean_axis(seq, [tok2idx[t] for t in wrists], fwd)
    shoulder_series = _marker_axis(seq, tok2idx[shoulder], fwd)
    return _range(wrist - shoulder_series)


def arm_swing_mean(seq: np.ndarray, tok2idx: dict[str, int], up_axis: int) -> float:
    """Mean of L/R shoulder-relative wrist fore-aft swing (PR #27 definition)."""
    fwd = forward_axis(up_axis)
    left = _side_arm_swing(seq, tok2idx, ARM_LEFT_WRISTS, ARM_LEFT_SHOULDER, fwd)
    right = _side_arm_swing(seq, tok2idx, ARM_RIGHT_WRISTS, ARM_RIGHT_SHOULDER, fwd)
    sides = np.array([left, right])
    return float(np.nanmean(sides)) if np.any(np.isfinite(sides)) else float("nan")


def head_vertical_range(seq: np.ndarray, tok2idx: dict[str, int], up_axis: int) -> float:
    """Vertical travel range of the head-marker centroid (head bob amplitude)."""
    head_y = _markers_mean_axis(seq, [tok2idx[t] for t in HEAD_MARKERS], up_axis)
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
