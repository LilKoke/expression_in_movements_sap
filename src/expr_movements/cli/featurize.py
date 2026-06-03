"""``expr-featurize``: build the expert-feature table (Parquet) for approach A."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build expert-feature table (approach A).")
    parser.add_argument("--manifest", default="data/processed/manifest.jsonl")
    parser.add_argument("--out", default="data/processed/features.parquet")
    parser.parse_args(argv)
    raise NotImplementedError("feature-engineering phase")


if __name__ == "__main__":
    main()
