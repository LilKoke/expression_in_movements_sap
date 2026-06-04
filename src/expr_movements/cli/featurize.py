"""``expr-featurize``: build the expert-feature table (Parquet) for approach A.

Thin entry point — reads the canonical ``sequences.npz`` and delegates to
:func:`expr_movements.features.build_feature_table`.
"""

from __future__ import annotations

import argparse

from expr_movements.features import build_feature_table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build expert-feature table (approach A).")
    parser.add_argument(
        "--sequences",
        default="data/processed/sequences.npz",
        help="canonical normalized sequence store (the common contract)",
    )
    parser.add_argument("--out", default="data/processed/features.parquet")
    parser.add_argument(
        "--up-axis",
        type=int,
        default=1,
        choices=(0, 1, 2),
        help="vertical axis index of the marker coordinates (bundled TRC is Y-up=1)",
    )
    args = parser.parse_args(argv)
    build_feature_table(args.sequences, args.out, up_axis=args.up_axis)


if __name__ == "__main__":
    main()
