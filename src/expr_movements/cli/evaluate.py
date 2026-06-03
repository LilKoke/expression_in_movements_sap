"""``expr-evaluate``: evaluate a run, or compare two runs (approach A vs B)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a run or compare two runs.")
    parser.add_argument("--run", help="run directory to evaluate")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="compare two runs")
    parser.add_argument("--out", default="outputs/comparison.md")
    parser.parse_args(argv)
    raise NotImplementedError("evaluation phase")


if __name__ == "__main__":
    main()
