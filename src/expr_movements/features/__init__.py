"""Feature/representation building.

Two distinct things live here:

- :mod:`expr_movements.features.preprocess` — the **invariant pre-processing**
  (Phase 3, #5): pelvis-local coordinates + yaw alignment + a speed channel.
  This is shared by both modeling approaches and is applied by
  ``data/dataset.py`` when building ``sequences.npz`` (the common contract).
- :func:`build_feature_table` below — **approach-A expert features** (velocity /
  acceleration statistics, joint-angle ranges, gait cadence, posture
  descriptors) into a Parquet table. Per the #1 roadmap v2 this is the
  expert-feature *team's* work and is left as a stub here.
"""

from __future__ import annotations

from pathlib import Path


def build_feature_table(manifest_path: str | Path, out_path: str | Path) -> Path:
    """Write the expert-feature table (approach A) as Parquet."""
    raise NotImplementedError("feature-engineering phase")
