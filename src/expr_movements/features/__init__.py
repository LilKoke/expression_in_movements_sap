"""Feature/representation building.

This module builds approach-A expert features from the processed sequence
dataset. The input is the common ``sequences.npz`` contract produced by the
pre-processing pipeline. Each clip is converted into one row of interpretable
gait features and written as a Parquet table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


VERTICAL_AXIS = 1
FORWARD_AXIS = 2


def _bare(name: str) -> str:
    """Return marker name without subject prefix: 'NABA_LFHD' -> 'LFHD'."""
    return str(name).rsplit("_", 1)[-1]


def _safe_range(x: np.ndarray) -> float:
    """NaN-robust range. Return NaN if all values are missing."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))


def _safe_mean(x: np.ndarray) -> float:
    """NaN-robust mean. Return NaN if all values are missing."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def _resolve_npz_path(path: str | Path) -> Path:
    """Accept either sequences.npz, manifest.jsonl, or a processed directory."""
    p = Path(path)

    if p.suffix == ".npz":
        return p

    if p.suffix == ".jsonl":
        return p.parent / "sequences.npz"

    if p.is_dir():
        return p / "sequences.npz"

    return p


def _extract_features_from_sequence(
    seq: np.ndarray,
    *,
    marker_names: list[str],
    n_markers: int,
    has_speed_channel: bool,
) -> dict[str, float]:
    """Compute four interpretable gait features from one processed clip."""
    pose = seq[:, : n_markers * 3].reshape(seq.shape[0], n_markers, 3)

    if not has_speed_channel:
        raise ValueError("Expected a speed channel in the last feature column.")

    speed = seq[:, -1]

    bare_names = [_bare(name) for name in marker_names]
    name_to_idx = {name: i for i, name in enumerate(bare_names)}

    def marker(name: str) -> np.ndarray:
        if name not in name_to_idx:
            raise KeyError(f"Marker {name} not found in marker_names.")
        return pose[:, name_to_idx[name], :]

    def avg_markers(names: list[str]) -> np.ndarray:
        arrays = [marker(name) for name in names]
        with np.errstate(invalid="ignore"):
            return np.nanmean(np.stack(arrays, axis=0), axis=0)

    # 1. Walking speed
    walking_speed = _safe_mean(speed)

    # 2. Stride length proxy
    left_ankle = marker("LANK")
    right_ankle = marker("RANK")
    stride_length_proxy = _safe_range(
        left_ankle[:, FORWARD_AXIS] - right_ankle[:, FORWARD_AXIS]
    )

    # 3. Arm swing mean
    left_wrist = avg_markers(["LWRA", "LWRB"])
    right_wrist = avg_markers(["RWRA", "RWRB"])
    left_shoulder = marker("LSHO")
    right_shoulder = marker("RSHO")

    left_arm_swing = _safe_range((left_wrist - left_shoulder)[:, FORWARD_AXIS])
    right_arm_swing = _safe_range((right_wrist - right_shoulder)[:, FORWARD_AXIS])
    arm_swing_mean = _safe_mean(np.array([left_arm_swing, right_arm_swing]))

    # 4. Head vertical range
    head = avg_markers(["LFHD", "RFHD", "LBHD", "RBHD"])
    head_vertical_range = _safe_range(head[:, VERTICAL_AXIS])

    return {
        "walking_speed": walking_speed,
        "stride_length_proxy": stride_length_proxy,
        "arm_swing_mean": arm_swing_mean,
        "head_vertical_range": head_vertical_range,
    }


def build_feature_table(manifest_path: str | Path, out_path: str | Path) -> Path:
    """Write the expert-feature table for approach A as Parquet.

    Parameters
    ----------
    manifest_path:
        Path to ``sequences.npz``, ``manifest.jsonl``, or the processed data
        directory. When ``manifest.jsonl`` is given, ``sequences.npz`` is read
        from the same directory.
    out_path:
        Destination path for the Parquet feature table.

    Returns
    -------
    Path
        Path to the written Parquet file.
    """
    npz_path = _resolve_npz_path(manifest_path)
    out = Path(out_path)

    data = np.load(npz_path, allow_pickle=True)

    sequences = data["sequences"]
    clips = data["clips"]
    subjects = data["subjects"]
    labels = data["labels"]
    marker_names = list(data["marker_names"])
    n_markers = int(data["n_markers"])
    has_speed_channel = bool(data["has_speed_channel"])

    rows: list[dict[str, object]] = []

    for i, seq in enumerate(sequences):
        features = _extract_features_from_sequence(
            seq,
            marker_names=marker_names,
            n_markers=n_markers,
            has_speed_channel=has_speed_channel,
        )

        row: dict[str, object] = {
            "clip_idx": i,
            "clip": str(clips[i]),
            "subject": str(subjects[i]),
            "label": str(labels[i]),
        }
        row.update(features)
        rows.append(row)

    df = pd.DataFrame(rows)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    return out
