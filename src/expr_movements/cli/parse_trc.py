"""``expr-parse``: parse raw ``.trc`` files into interim per-clip arrays.

Each ``.trc`` under ``--raw-dir`` is parsed by :func:`expr_movements.data.trc.read_trc`
and written as one compressed ``.npz`` under ``--out-dir``. The ``.npz`` holds the
``(n_frames, n_markers, 3)`` coordinate array plus the filename metadata (subject,
emotion, take) and marker names, so the dataset-build phase can consume interim
clips without re-reading the raw TRC.

This is a thin entry point: all parsing logic lives in ``data/trc.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from expr_movements.data.trc import read_trc


def parse_dir(raw_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """Parse every ``.trc`` in ``raw_dir`` into a per-clip ``.npz`` in ``out_dir``.

    Returns the list of written ``.npz`` paths. Files that fail to parse are
    reported and skipped so one bad clip doesn't abort the whole batch.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    trc_files = sorted(raw_dir.glob("*.trc"))
    if not trc_files:
        print(f"no .trc files found under {raw_dir}")
        return written

    for trc_path in trc_files:
        try:
            clip = read_trc(trc_path)
        except (ValueError, KeyError, OSError) as exc:
            print(f"skip {trc_path.name}: {exc}")
            continue
        out_path = out_dir / f"{trc_path.stem}.npz"
        np.savez_compressed(
            out_path,
            frames=clip.frames,
            times=clip.times,
            marker_names=np.asarray(clip.marker_names, dtype=object),
            frame_rate=clip.frame_rate,
            subject=clip.meta.subject,
            emotion_code=clip.meta.emotion_code,
            emotion=clip.meta.emotion,
            take=clip.meta.take,
        )
        written.append(out_path)
        print(
            f"{trc_path.name} -> {out_path.name}  "
            f"({clip.n_frames} frames, {clip.n_markers} markers, {clip.meta.emotion})"
        )

    print(f"parsed {len(written)}/{len(trc_files)} clips into {out_dir}")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parse raw TRC into interim arrays.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/interim")
    args = parser.parse_args(argv)
    parse_dir(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
