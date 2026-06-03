"""Expert/hand-crafted feature extraction for approach A.

Turns per-clip pose sequences into a fixed-length feature vector (e.g. velocity
/ acceleration statistics, joint-angle ranges, gait cadence, posture
descriptors). Output is a Parquet feature table: one row per clip, plus subject
and emotion columns for grouped splitting.

Implementation lands in the feature-engineering phase (see issues from #1).
"""

from __future__ import annotations

from pathlib import Path


def build_feature_table(manifest_path: str | Path, out_path: str | Path) -> Path:
    """Write the expert-feature table (approach A) as Parquet."""
    raise NotImplementedError("feature-engineering phase")
