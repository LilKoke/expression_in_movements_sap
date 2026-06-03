"""``expr-build-dataset``: build the clip manifest (JSONL) + sequence tensor (npz)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build modeling-ready sequence dataset.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.parse_args(argv)
    raise NotImplementedError("dataset-build phase")


if __name__ == "__main__":
    main()
