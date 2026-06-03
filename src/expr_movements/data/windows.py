"""Sliding-window expansion + the common data contract (Phase 4, #6).

``sequences.npz`` (built in Phase 2/3) is the *clip-level* common contract: one
position/heading-invariant ``(T_i, F)`` sequence per clip, plus per-clip
``labels``, ``subjects`` and ``clips``. This module turns it into the
*window-level* samples both approaches actually train on, while keeping the
split honest:

* :class:`SequenceDataset` loads the npz and exposes the per-clip arrays
  (``subjects`` / ``trials`` / ``labels``) that ``splits.iter_splits`` folds on.
* :func:`make_windows` slides a fixed ``length``-frame window every ``stride``
  frames over the clips in a *given set of clip indices*, returning one row per
  window tagged with its source clip. Because windowing happens after the fold
  is chosen, every window of a clip stays on that clip's side — no window-level
  leakage.
* :func:`fit_scaler` / :func:`apply_scaler` compute the normalisation statistics
  (mean/std) from **train windows only** and apply them to any split, so test
  statistics never bleed into training.

Approach A (classic ML) flattens each window to a feature vector; approach B
(NN) keeps the ``(length, F)`` shape. :func:`flatten_windows` does the former.
The windowing, splitting and scaling are shared so the A-vs-B comparison runs on
identical samples and folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SequenceDataset:
    """The clip-level common contract loaded from ``sequences.npz``.

    ``sequences`` is a length-``n_clips`` object array of ``(T_i, F)`` float32
    arrays. ``subjects`` / ``labels`` / ``clips`` are per-clip. ``trials`` is a
    per-clip trial id used by intra-subject splitting; here each clip is its own
    trial (one file == one take), so it equals the clip index.
    """

    sequences: np.ndarray  # object array of (T_i, F) float32
    labels: np.ndarray  # (n_clips,) str
    subjects: np.ndarray  # (n_clips,) str
    clips: np.ndarray  # (n_clips,) str clip filenames
    feature_dim: int
    has_speed_channel: bool

    @property
    def trials(self) -> np.ndarray:
        """Per-clip trial id (one file == one trial) for intra-subject splits."""
        return np.arange(len(self.sequences))

    @classmethod
    def load(cls, path: str | Path) -> "SequenceDataset":
        d = np.load(Path(path), allow_pickle=True)
        seqs = d["sequences"]
        feature_dim = int(seqs[0].shape[1]) if len(seqs) else 0
        # ``has_speed_channel`` / ``feature_layout`` were added with Phase 3; an
        # older npz lacks them — infer conservatively (no speed channel).
        has_speed = bool(d["has_speed_channel"]) if "has_speed_channel" in d else False
        return cls(
            sequences=seqs,
            labels=np.asarray(d["labels"]),
            subjects=np.asarray(d["subjects"]),
            clips=np.asarray(d["clips"]),
            feature_dim=feature_dim,
            has_speed_channel=has_speed,
        )

    def __len__(self) -> int:
        return len(self.sequences)


@dataclass(frozen=True)
class WindowSet:
    """Windowed samples for one split.

    ``X`` is ``(n_windows, length, F)`` float32; ``mask`` ``(n_windows, length)``
    bool (False on padded frames of short clips). ``y`` is the per-window label;
    ``clip_idx`` the source clip index per window (for clip-level majority vote).
    ``has_speed`` mirrors the dataset's ``has_speed_channel`` (whether the last
    feature column is the walking-speed channel) so a model that treats speed
    specially — e.g. the NN concatenating it onto the latent — knows where it is.
    """

    X: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    clip_idx: np.ndarray
    has_speed: bool = False


def _clip_windows(seq: np.ndarray, length: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Slide ``length``-frame windows over one ``(T, F)`` clip every ``stride``.

    Returns ``(windows, masks)`` with windows ``(k, length, F)`` and masks
    ``(k, length)``. A clip shorter than ``length`` yields a single zero-padded
    window with its real frames masked True. Otherwise the last window is
    anchored to the clip end so the final frames are always covered even when
    ``(T - length)`` is not a multiple of ``stride``.
    """
    t, f = seq.shape
    if t <= length:
        win = np.zeros((1, length, f), dtype=np.float32)
        msk = np.zeros((1, length), dtype=bool)
        win[0, :t] = seq
        msk[0, :t] = True
        return win, msk

    starts = list(range(0, t - length + 1, stride))
    if starts[-1] != t - length:
        starts.append(t - length)  # cover the tail
    win = np.stack([seq[s : s + length] for s in starts]).astype(np.float32)
    msk = np.ones((len(starts), length), dtype=bool)
    return win, msk


def make_windows(
    ds: SequenceDataset, clip_indices: np.ndarray, length: int, stride: int
) -> WindowSet:
    """Expand the given clips into a :class:`WindowSet` of fixed-length windows.

    ``clip_indices`` selects which clips (typically one side of a fold) to
    window. Every window is tagged with its source clip index so predictions can
    be majority-voted back to a clip-level decision downstream.
    """
    xs, masks, ys, idxs = [], [], [], []
    for ci in np.asarray(clip_indices):
        win, msk = _clip_windows(ds.sequences[ci], length, stride)
        xs.append(win)
        masks.append(msk)
        ys.append(np.full(len(win), ds.labels[ci], dtype=object))
        idxs.append(np.full(len(win), ci, dtype=np.int64))
    if not xs:
        f = ds.feature_dim
        return WindowSet(
            X=np.zeros((0, length, f), np.float32),
            mask=np.zeros((0, length), bool),
            y=np.array([], dtype=object),
            clip_idx=np.array([], dtype=np.int64),
            has_speed=ds.has_speed_channel,
        )
    return WindowSet(
        X=np.concatenate(xs),
        mask=np.concatenate(masks),
        y=np.concatenate(ys),
        clip_idx=np.concatenate(idxs),
        has_speed=ds.has_speed_channel,
    )


def fit_scaler(ws: WindowSet) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over the real (unmasked) frames of ``ws`` windows.

    Computed from **train windows only**; the returned ``(mean, std)`` are then
    applied to val/test via :func:`apply_scaler`. ``std`` is floored away from
    zero so constant features don't divide by zero. NaNs (residual occlusions)
    are ignored in the statistics.
    """
    flat = ws.X[ws.mask]  # (n_real_frames, F)
    if flat.size == 0:
        f = ws.X.shape[2]
        return np.zeros(f, np.float32), np.ones(f, np.float32)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
    mean = np.nan_to_num(mean, nan=0.0).astype(np.float32)
    std = np.nan_to_num(std, nan=1.0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_scaler(ws: WindowSet, mean: np.ndarray, std: np.ndarray) -> WindowSet:
    """Standardise ``ws.X`` with train ``mean``/``std``; zero out padded frames.

    Residual NaNs become 0 after centring so downstream models see finite input.
    Padded frames (``mask`` False) are forced to 0 so they carry no signal.
    """
    x = (ws.X - mean) / std
    x = np.nan_to_num(x, nan=0.0).astype(np.float32)
    x[~ws.mask] = 0.0
    return WindowSet(X=x, mask=ws.mask, y=ws.y, clip_idx=ws.clip_idx, has_speed=ws.has_speed)


def flatten_windows(ws: WindowSet) -> np.ndarray:
    """Flatten ``(n, length, F)`` windows to ``(n, length*F)`` for classic ML."""
    n = ws.X.shape[0]
    return ws.X.reshape(n, -1)
