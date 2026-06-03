"""``expr-featurize``: build the expert-feature table for approach A."""

from __future__ import annotations

import argparse
from pathlib import Path

from expr_movements.features import build_feature_table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build expert-feature table for approach A."
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/manifest.jsonl",
        help=(
            "Path to manifest.jsonl, sequences.npz, or the processed data directory. "
            "If manifest.jsonl is given, sequences.npz is read from the same directory."
        ),
    )
    parser.add_argument(
        "--out",
        default="data/processed/features.parquet",
        help="Output Parquet path for the expert-feature table.",
    )

    args = parser.parse_args(argv)

    out_path = build_feature_table(
        Path(args.manifest),
        Path(args.out),
    )

    print(f"Wrote expert-feature table to {out_path}")


if __name__ == "__main__":
    main()
