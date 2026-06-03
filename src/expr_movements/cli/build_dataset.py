"""``expr-build-dataset``: build the clip manifest (JSONL) + sequence tensor (npz).

Thin entry point. Reads the parsed interim clips (``expr-parse`` output), trims
each to its active walking window per :class:`~expr_movements.config.DataConfig`,
and writes ``manifest.jsonl`` + ``sequences.npz`` under ``--out-dir``. All logic
lives in :mod:`expr_movements.data.dataset`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from expr_movements.config import DataConfig, load_experiment
from expr_movements.data.dataset import build_manifest, build_sequence_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build modeling-ready sequence dataset.")
    parser.add_argument(
        "--interim-dir",
        default="data/interim",
        help="directory of parsed per-clip .npz (expr-parse output)",
    )
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument(
        "--config",
        default=None,
        help="experiment YAML to take the DataConfig (trim/onset) from; "
        "defaults to DataConfig() defaults",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=None,
        help="if set, also write a dense (n_clips, N, 41*3) tensor + mask "
        "alongside the variable-length sequences",
    )
    args = parser.parse_args(argv)

    cfg = load_experiment(args.config).data if args.config else DataConfig()

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "manifest.jsonl"
    sequences_path = out_dir / "sequences.npz"

    build_manifest(args.interim_dir, manifest_path, cfg=cfg)
    build_sequence_dataset(manifest_path, sequences_path, target_frames=args.target_frames, cfg=cfg)


if __name__ == "__main__":
    main()
