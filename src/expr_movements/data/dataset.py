"""Build modeling-ready datasets from parsed TRC, and persist them.

Per the roadmap (#1): produce, per clip, the marker coordinates from N frames
after motion onset to M frames before the end, paired with the clip's emotion
class. A human-inspectable manifest of (clip path, subject, emotion, n_frames)
is written as JSONL; the numeric arrays are written as a dense ``.npz`` sequence
tensor (approach B) and the feature table as Parquet (approach A).

Implementation lands in the dataset-build / feature phases (see issues from #1).
"""

from __future__ import annotations

from pathlib import Path


def build_manifest(raw_dir: str | Path, out_path: str | Path) -> Path:
    """Scan ``raw_dir`` for ``.trc`` files and write a JSONL clip manifest."""
    raise NotImplementedError("dataset-build phase")


def build_sequence_dataset(manifest_path: str | Path, out_path: str | Path) -> Path:
    """Write the raw-sequence tensor dataset (approach B) as ``.npz``."""
    raise NotImplementedError("dataset-build phase")
