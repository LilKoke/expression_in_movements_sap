"""``expr-evaluate``: read a run's metrics and print the Phase 6 evaluation.

    expr-evaluate --run outputs/cnn1d_a1b2c3d4

Loads the run's persisted ``metrics.json`` (written by ``expr-train``) and prints
the shared Phase 6 read-out: Macro-F1 / accuracy / balanced accuracy as mean ± std
at the window and trial levels, per-class F1, confusion matrix, the split
protocol, and — for the multi-task NN — reconstruction error and latent
separability. Use ``--json`` for the machine-readable summary, ``--out`` to also
write the markdown report to a file.

The A-vs-B comparison (``--compare``) lands in Phase 7 (#8).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expr_movements.evaluate import compare_runs, evaluate_run, format_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a run or compare two runs.")
    parser.add_argument("--run", help="run directory to evaluate")
    parser.add_argument(
        "--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="compare two runs (Phase 7)"
    )
    parser.add_argument("--out", help="also write the markdown report to this path")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)

    if args.compare:
        out = args.out or "outputs/comparison.md"
        compare_runs(args.compare[0], args.compare[1], out)
        return

    if not args.run:
        parser.error("one of --run or --compare is required")

    summary = evaluate_run(args.run)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        report = format_report(summary)
        print(report)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(report)
            print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
