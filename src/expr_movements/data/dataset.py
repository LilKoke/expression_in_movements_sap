"""Build modeling-ready datasets from parsed TRC, and persist them.

Per the roadmap (#1): produce, per clip, the marker coordinates from the start
of motion to its end, paired with the clip's emotion class. Two artifacts are
written to ``data/processed/``:

``manifest.jsonl``
    One JSON object per clip — its interim path, subject, emotion (code + name),
    take, and frame counts (raw and after trimming). Human-inspectable and the
    index that :func:`build_sequence_dataset` consumes.

``sequences.npz``
    The approach-B input. Clip lengths vary, so the canonical store is a
    **variable-length** object array ``sequences`` of ``(T_i, 41*3)`` float32
    arrays (one per clip, already onset-trimmed), alongside ``lengths``,
    ``labels``, ``subjects`` and ``marker_names``. Training slices any fixed
    window / length out of these in memory (cheap — the whole set is a few tens
    of MB), so the modeling frame count is never baked into the dataset.

    Passing ``target_frames=N`` additionally writes a dense ``sequences_dense``
    array ``(n_clips, N, 41*3)`` (pad/truncate to ``N``) plus its ``mask``, for
    callers that want a ready-made fixed-length tensor.

Trimming. With ``DataConfig.detect_onset`` (default on) the active walking
window is found from marker speed — the standing frames before the first step
and after the last are dropped — and ``trim_start_frames`` / ``trim_end_frames``
are applied as an extra fixed margin inside it. With detection off, only the
fixed trims apply. See :func:`active_window` for the detection.

All logic lives here; ``cli/build_dataset.py`` is a thin entry point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from expr_movements.config import DataConfig


def _interim_npz(interim_dir: Path) -> list[Path]:
    return sorted(Path(interim_dir).glob("*.npz"))


def _frame_speeds(frames: np.ndarray) -> np.ndarray:
    """Mean marker speed per frame (NaN-robust).

    ``frames`` is ``(T, n_markers, 3)``. Speed at frame ``t`` is the mean over
    markers of the inter-frame displacement magnitude ``|p_t - p_{t-1}|``;
    markers missing on either frame (NaN) are ignored. Returns a length-``T``
    array with ``speeds[0] = 0`` (no previous frame to difference against).
    """
    if frames.shape[0] < 2:
        return np.zeros(frames.shape[0], dtype=np.float64)
    disp = np.diff(frames, axis=0)  # (T-1, n_markers, 3)
    dist = np.sqrt(np.sum(disp**2, axis=2))  # (T-1, n_markers); NaN if marker missing
    with np.errstate(invalid="ignore"):
        per_frame = np.nanmean(dist, axis=1)  # (T-1,)
    per_frame = np.nan_to_num(per_frame, nan=0.0)  # frames with all markers missing
    return np.concatenate([[0.0], per_frame])


def _first_run_start(mask: np.ndarray, min_run: int) -> int | None:
    """Index where the first run of ``>= min_run`` True values begins, or None."""
    run = 0
    for i, on in enumerate(mask):
        run = run + 1 if on else 0
        if run >= min_run:
            return i - min_run + 1
    return None


def active_window(frames: np.ndarray, speed_frac: float, min_run: int) -> tuple[int, int]:
    """Return ``(start, stop)`` frame indices of the active (walking) window.

    A frame is "moving" when its mean marker speed exceeds ``speed_frac`` of the
    clip's peak speed. ``start`` is the first index beginning a run of at least
    ``min_run`` consecutive moving frames; ``stop`` is one past the last index
    ending such a run (so ``frames[start:stop]`` is the active span). If no run
    qualifies (e.g. a near-static clip) the full clip ``(0, T)`` is returned so
    nothing is silently dropped.
    """
    speeds = _frame_speeds(frames)
    n = len(speeds)
    peak = float(speeds.max()) if n else 0.0
    if peak <= 0.0:
        return 0, n
    moving = speeds >= speed_frac * peak

    start = _first_run_start(moving, min_run)
    if start is None:
        return 0, n
    # Offset: mirror the same logic on the reversed mask.
    rstart = _first_run_start(moving[::-1], min_run)
    stop = n - rstart  # rstart is non-None whenever start was found
    return start, stop


def trim_clip(frames: np.ndarray, cfg: DataConfig) -> tuple[int, int]:
    """Resolve the kept ``(start, stop)`` frame slice for one clip under ``cfg``.

    Applies onset/offset detection (when enabled) then the fixed
    ``trim_start_frames`` / ``trim_end_frames`` margins inside the detected
    window. Always returns a non-empty slice: if the margins would empty the
    clip they are skipped and the detected window (or full clip) is kept, so a
    short clip is never reduced to zero frames.
    """
    n = frames.shape[0]
    if cfg.detect_onset:
        start, stop = active_window(frames, cfg.onset_speed_frac, cfg.onset_min_run)
    else:
        start, stop = 0, n

    inner_start = start + cfg.trim_start_frames
    inner_stop = stop - cfg.trim_end_frames
    if inner_start < inner_stop:
        return inner_start, inner_stop
    return start, stop  # margins would empty the clip -> keep the detected window


def build_manifest(
    interim_dir: str | Path,
    out_path: str | Path,
    cfg: DataConfig | None = None,
) -> Path:
    """Scan ``interim_dir`` for parsed ``.npz`` clips and write a JSONL manifest.

    Each line records the clip's interim path, subject, emotion, take, the raw
    frame count and the kept frame count after trimming under ``cfg`` (default
    :class:`DataConfig`). Returns ``out_path``. Raises if no clips are found, so
    a missing ``expr-parse`` step fails loudly rather than writing an empty set.
    """
    cfg = cfg or DataConfig()
    interim_dir = Path(interim_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clips = _interim_npz(interim_dir)
    if not clips:
        raise FileNotFoundError(f"no interim .npz clips under {interim_dir} — run expr-parse first")

    records = []
    for npz_path in clips:
        d = np.load(npz_path, allow_pickle=True)
        frames = d["frames"]
        start, stop = trim_clip(frames, cfg)
        records.append(
            {
                "clip": npz_path.name,
                "interim_path": str(npz_path),
                "subject": str(d["subject"]),
                "emotion_code": str(d["emotion_code"]),
                "emotion": str(d["emotion"]),
                "take": int(d["take"]),
                "n_frames_raw": int(frames.shape[0]),
                "n_markers": int(frames.shape[1]),
                "trim_start": int(start),
                "trim_stop": int(stop),
                "n_frames": int(stop - start),
            }
        )

    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote manifest: {len(records)} clips -> {out_path}")
    return out_path


def _read_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _clip_sequence(rec: dict) -> np.ndarray:
    """Load one clip's trimmed, flattened ``(T, n_markers*3)`` float32 array."""
    d = np.load(rec["interim_path"], allow_pickle=True)
    frames = d["frames"][rec["trim_start"] : rec["trim_stop"]]  # (T, n_markers, 3)
    flat = frames.reshape(frames.shape[0], -1)  # (T, n_markers*3)
    return flat.astype(np.float32)


def _pad_or_truncate(seq: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad (zeros) or truncate ``seq`` (T, F) to length ``target``; return (out, mask).

    ``mask`` is ``(target,)`` bool, True for real frames. Truncation keeps the
    first ``target`` frames; padding is appended at the end and masked False so
    callers can ignore the fake frames.
    """
    t, f = seq.shape
    out = np.zeros((target, f), dtype=np.float32)
    mask = np.zeros(target, dtype=bool)
    keep = min(t, target)
    out[:keep] = seq[:keep]
    mask[:keep] = True
    return out, mask


def build_sequence_dataset(
    manifest_path: str | Path,
    out_path: str | Path,
    target_frames: int | None = None,
) -> Path:
    """Write the approach-B sequence dataset (``.npz``) from a manifest.

    Always stores the variable-length canonical form: an object array
    ``sequences`` of per-clip ``(T_i, n_markers*3)`` float32 arrays plus
    ``lengths``, ``labels``, ``emotion_codes``, ``subjects``, ``clips`` and
    ``marker_names``. When ``target_frames`` is given, additionally stores a
    dense ``sequences_dense`` ``(n_clips, target_frames, n_markers*3)`` and its
    ``mask`` (pad/truncate to ``target_frames``). Returns ``out_path``.
    """
    manifest_path = Path(manifest_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = _read_manifest(manifest_path)
    if not records:
        raise ValueError(f"empty manifest: {manifest_path}")

    sequences = [_clip_sequence(rec) for rec in records]
    lengths = np.asarray([s.shape[0] for s in sequences], dtype=np.int64)
    labels = np.asarray([rec["emotion"] for rec in records], dtype=object)
    emotion_codes = np.asarray([rec["emotion_code"] for rec in records], dtype=object)
    subjects = np.asarray([rec["subject"] for rec in records], dtype=object)
    clips = np.asarray([rec["clip"] for rec in records], dtype=object)

    # marker_names is identical across clips (same rig); carry one copy through.
    first = np.load(records[0]["interim_path"], allow_pickle=True)
    marker_names = np.asarray(list(first["marker_names"]), dtype=object)

    seq_obj = np.empty(len(sequences), dtype=object)
    seq_obj[:] = sequences

    arrays: dict[str, np.ndarray] = {
        "sequences": seq_obj,
        "lengths": lengths,
        "labels": labels,
        "emotion_codes": emotion_codes,
        "subjects": subjects,
        "clips": clips,
        "marker_names": marker_names,
    }

    if target_frames is not None:
        n_features = sequences[0].shape[1]
        dense = np.zeros((len(sequences), target_frames, n_features), dtype=np.float32)
        mask = np.zeros((len(sequences), target_frames), dtype=bool)
        for i, s in enumerate(sequences):
            dense[i], mask[i] = _pad_or_truncate(s, target_frames)
        arrays["sequences_dense"] = dense
        arrays["mask"] = mask

    np.savez_compressed(out_path, **arrays)
    extra = (
        f" + dense {len(sequences)}x{target_frames}x{sequences[0].shape[1]}"
        if target_frames is not None
        else ""
    )
    print(
        f"wrote sequences: {len(sequences)} clips "
        f"(lengths {int(lengths.min())}-{int(lengths.max())}){extra} -> {out_path}"
    )
    return out_path
