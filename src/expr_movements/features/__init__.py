"""Feature/representation building.

Two distinct things live here:

- :mod:`expr_movements.features.preprocess` — the **invariant pre-processing**
  (Phase 3, #5): pelvis-local coordinates + yaw alignment + a speed channel.
  This is shared by both modeling approaches and is applied by
  ``data/dataset.py`` when building ``sequences.npz`` (the common contract).
- :mod:`expr_movements.features.expert` + :func:`build_feature_table` below —
  **approach-A expert features** (Phase 9, #20): interpretable gait descriptors
  (walking speed, stride/arm-swing amplitude, head bob) computed per clip from
  the normalized sequences and written to a Parquet table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from expr_movements.features.expert import FEATURE_NAMES, compute_features


def build_feature_table(
    sequences_path: str | Path,
    out_path: str | Path,
    *,
    up_axis: int = 1,
) -> Path:
    """Write the approach-A expert-feature table as Parquet.

    Reads the canonical ``sequences.npz`` (the common contract — normalized,
    yaw-aligned per-clip pose with a trailing speed channel) and computes the
    :mod:`features.expert` descriptors for every clip. The output frame carries
    the clip's identity/label columns (``clip``, ``subject``, ``emotion``,
    ``emotion_code``) followed by one column per feature in
    :data:`features.expert.FEATURE_NAMES`, so the split layer can group by
    subject exactly as the sequence pipeline does. Returns ``out_path``.
    """
    sequences_path = Path(sequences_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(sequences_path, allow_pickle=True)
    sequences = data["sequences"]
    marker_names = [str(m) for m in data["marker_names"]]
    has_speed = bool(data["has_speed_channel"])
    clips = data["clips"]
    subjects = data["subjects"]
    labels = data["labels"]
    emotion_codes = data["emotion_codes"]

    rows = []
    for i, seq in enumerate(sequences):
        feats = compute_features(
            np.asarray(seq), marker_names, has_speed=has_speed, up_axis=up_axis
        )
        rows.append(
            {
                "clip": str(clips[i]),
                "subject": str(subjects[i]),
                "emotion": str(labels[i]),
                "emotion_code": str(emotion_codes[i]),
                **feats,
            }
        )

    columns = ["clip", "subject", "emotion", "emotion_code", *FEATURE_NAMES]
    df = pd.DataFrame(rows, columns=columns)
    df.to_parquet(out_path, index=False)
    print(f"wrote features: {len(df)} clips x {len(FEATURE_NAMES)} features -> {out_path}")
    return out_path
