"""Invariant pre-processing of pose sequences (Phase 3, #5).

The course requires the classifier to key on *how* someone moves, not *where*
in the capture volume they walked or which way they faced. This module turns a
raw per-clip marker sequence into a position- and heading-invariant pose
sequence, while deliberately **keeping walking speed** as an explicit channel
(speed tracks arousal — a strong emotion cue, per Venture et al. 2014).

Two invariances, in order, per clip:

1. **Pelvis-centred local coordinates.** Every frame is translated so the
   centroid of the four pelvis markers (``LFWT/RFWT/LBWT/RBWT``) sits at the
   origin. This removes the absolute walking position and the slow forward
   drift across the room.
2. **Yaw alignment.** Every frame is rotated about the vertical axis so the
   pelvis left->right axis points in a fixed direction. This removes the
   subject's heading (which way they walked) while leaving pitch and roll —
   trunk lean, head orientation — intact, since those carry posture/emotion.

Speed is computed from the pelvis-centroid trajectory in **world** coordinates
*before* centring (centring would zero it out), so it reflects real walking
speed in the capture's length units per frame.

Marker names in this dataset carry a per-subject prefix (e.g. ``NABA_LFWT``);
pelvis markers are matched by the bare token after the last ``_``. Occluded
markers are ``NaN`` and are handled NaN-robustly throughout — a frame missing
some pelvis markers still gets a centroid from those present; only a frame with
*no* pelvis markers falls back to no translation for that frame.

The public entry point is :func:`normalize_sequence`. ``data/dataset.py`` calls
it per clip while building ``sequences.npz`` so the stored sequences — the
common contract consumed by both the NN team and the expert-feature team — are
already invariant.
"""

from __future__ import annotations

import numpy as np

from expr_movements.config import DataConfig


def _bare(name: str) -> str:
    """Marker token without its subject prefix: ``"NABA_LFWT" -> "LFWT"``."""
    return name.rsplit("_", 1)[-1]


def pelvis_indices(marker_names: list[str], pelvis: tuple[str, ...]) -> list[int]:
    """Indices of the pelvis markers within ``marker_names`` (by bare token).

    Raises if none are found, since every invariance here hinges on the pelvis
    frame — a clip without it cannot be normalized and should fail loudly rather
    than silently skip alignment.
    """
    bare = [_bare(n) for n in marker_names]
    idx = [i for i, b in enumerate(bare) if b in pelvis]
    if not idx:
        raise ValueError(f"no pelvis markers {pelvis} found among markers {marker_names[:6]}...")
    return idx


def _pelvis_centroid(frames: np.ndarray, pelvis_idx: list[int]) -> np.ndarray:
    """Per-frame pelvis centroid, NaN-robust. ``frames`` (T, M, 3) -> (T, 3).

    A frame with every pelvis marker occluded yields NaN here; callers replace
    that with a no-op translation so the frame is passed through unshifted.
    """
    pelvis = frames[:, pelvis_idx, :]  # (T, P, 3)
    with np.errstate(invalid="ignore"):
        return np.nanmean(pelvis, axis=1)  # (T, 3)


def _yaw_rotation(
    frames: np.ndarray,
    pelvis_idx: list[int],
    up_axis: int,
    marker_names: list[str],
    pelvis: tuple[str, ...],
) -> np.ndarray:
    """Per-frame rotation (T, 3, 3) that aligns the pelvis L->R axis to +X-of-plane.

    Works in the horizontal plane (the two axes other than ``up_axis``): it finds
    the pelvis left->right direction (right-front + right-back minus the left
    pair, projected to the plane) and rotates each frame so that direction is
    constant. Frames where the left->right vector is undefined (occlusion, or
    degenerate) get the identity rotation.
    """
    bare = [_bare(n) for n in marker_names]
    plane = [a for a in range(3) if a != up_axis]  # the two horizontal axes

    def _mean_of(tokens: set[str]) -> np.ndarray:
        cols = [i for i in pelvis_idx if bare[i] in tokens]
        with np.errstate(invalid="ignore"):
            return np.nanmean(frames[:, cols, :], axis=1)  # (T, 3)

    # Left->right axis = mean(right markers) - mean(left markers).
    right = {t for t in pelvis if t.startswith("R")}
    left = {t for t in pelvis if t.startswith("L")}
    lr = _mean_of(right) - _mean_of(left)  # (T, 3)

    # Project to the horizontal plane and build the angle that rotates it to +plane[0].
    u = lr[:, plane[0]]
    v = lr[:, plane[1]]
    norm = np.hypot(u, v)
    angle = np.arctan2(v, u)  # heading of the L->R axis in the plane
    cos = np.cos(-angle)
    sin = np.sin(-angle)

    t = frames.shape[0]
    rot = np.broadcast_to(np.eye(3), (t, 3, 3)).copy()
    # In-plane rotation by -angle; identity where the L->R axis is undefined.
    valid = norm > 0
    a0, a1 = plane
    rot[valid, a0, a0] = cos[valid]
    rot[valid, a0, a1] = -sin[valid]
    rot[valid, a1, a0] = sin[valid]
    rot[valid, a1, a1] = cos[valid]
    return rot


def body_speed(frames: np.ndarray, pelvis_idx: list[int]) -> np.ndarray:
    """Per-frame walking speed from the world-space pelvis centroid. (T,) -> units/frame.

    Speed at frame ``t`` is the centroid displacement magnitude ``|c_t - c_{t-1}|``;
    ``speed[0] = 0`` (no previous frame). NaN displacements (missing centroid on
    either frame) become 0 so the channel is always finite.
    """
    centroid = _pelvis_centroid(frames, pelvis_idx)  # (T, 3), world space
    if centroid.shape[0] < 2:
        return np.zeros(centroid.shape[0], dtype=np.float64)
    disp = np.diff(centroid, axis=0)  # (T-1, 3)
    dist = np.sqrt(np.sum(disp**2, axis=1))  # (T-1,), NaN where centroid missing
    dist = np.nan_to_num(dist, nan=0.0)
    return np.concatenate([[0.0], dist])


def normalize_sequence(frames: np.ndarray, marker_names: list[str], cfg: DataConfig) -> np.ndarray:
    """Make one clip position/heading-invariant; return a flat ``(T, F)`` array.

    ``frames`` is ``(T, M, 3)`` (world coordinates, possibly with NaN occlusions).
    With ``cfg.normalize`` off, returns the raw flattened ``(T, M*3)`` unchanged.
    With it on:

    * translate each frame so the pelvis centroid is the origin;
    * (if ``cfg.yaw_align``) rotate each frame about the vertical so heading is
      fixed;
    * flatten markers to ``(T, M*3)``;
    * (if ``cfg.keep_speed``) append the world-space body speed as a final
      column, giving ``(T, M*3 + 1)``.

    Marker NaNs are preserved through the transform (translation/rotation of NaN
    stays NaN) so downstream interpolation still sees the occluded coordinates.
    """
    frames = np.asarray(frames, dtype=np.float64)
    if not cfg.normalize:
        return frames.reshape(frames.shape[0], -1).astype(np.float32)

    pelvis_idx = pelvis_indices(marker_names, cfg.pelvis_markers)

    # Speed must be read from the world-space trajectory before we centre it.
    speed = body_speed(frames, pelvis_idx) if cfg.keep_speed else None

    centroid = _pelvis_centroid(frames, pelvis_idx)  # (T, 3)
    # A frame with no pelvis markers -> NaN centroid -> translate by 0 (no-op).
    centroid = np.nan_to_num(centroid, nan=0.0)
    local = frames - centroid[:, None, :]  # (T, M, 3)

    if cfg.yaw_align:
        rot = _yaw_rotation(
            frames, pelvis_idx, cfg.up_axis, marker_names, cfg.pelvis_markers
        )  # (T, 3, 3)
        # Rotate every marker of each frame: p' = R @ p  ==  (p @ R^T).
        local = np.einsum("tmj,tkj->tmk", local, rot)

    flat = local.reshape(local.shape[0], -1).astype(np.float32)  # (T, M*3)
    if speed is not None:
        flat = np.concatenate([flat, speed[:, None].astype(np.float32)], axis=1)
    return flat
