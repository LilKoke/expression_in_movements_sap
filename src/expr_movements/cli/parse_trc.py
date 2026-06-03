"""``expr-parse``: parse raw ``.trc`` files into interim per-clip arrays."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parse raw TRC into interim arrays.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/interim")
    parser.parse_args(argv)
    raise NotImplementedError("TRC-parsing phase")


if __name__ == "__main__":
    main()
