"""``expr-evaluate``: read a run's metrics, compare two runs, or plot the latent.

Single run (Phase 6 read-out)::

    expr-evaluate --run outputs/cnn1d_a1b2c3d4

Loads the run's persisted ``metrics.json`` (written by ``expr-train``) and prints
the shared read-out: Macro-F1 / accuracy / balanced accuracy as mean ± std at the
window and trial levels, per-class F1, confusion matrix, the split protocol, and
— for the multi-task NN — reconstruction error and latent separability. ``--json``
prints the machine-readable summary; ``--out`` also writes the markdown report.

Head-to-head comparison (Phase 7, #13)::

    expr-evaluate --compare RUN_A RUN_B --out outputs/comparison.md --plots outputs/figs

``--compare`` puts approach A (expert features) and B (the NN) side by side on the
*same* split and metrics. ``--compare-protocols INTRA LOSO`` instead contrasts the
intra-subject and LOSO runs of one approach (the gap = subject dependence).
``--plots DIR`` additionally writes the comparison bar-charts, and ``--pca RUN``
writes the latent PCA figure (emotion=colour, subject=marker) for a NN run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expr_movements.evaluate import (
    compare_protocols,
    compare_runs,
    evaluate_run,
    format_comparison,
    format_protocol_comparison,
    format_report,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a run, or compare two runs.")
    parser.add_argument("--run", help="run directory to evaluate (Phase 6 read-out)")
    parser.add_argument(
        "--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="A-vs-B head-to-head (same split)"
    )
    parser.add_argument(
        "--compare-protocols",
        nargs=2,
        metavar=("INTRA", "LOSO"),
        help="intra-subject vs LOSO for one approach (the gap = subject dependence)",
    )
    parser.add_argument("--pca", metavar="RUN", help="write the latent PCA figure for a NN run")
    parser.add_argument("--out", help="also write the markdown report to this path")
    parser.add_argument(
        "--plots", metavar="DIR", help="also write comparison figures into this dir"
    )
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)

    if args.pca:
        from expr_movements.viz import latent_pca

        out = args.out or f"outputs/figs/latent_pca_{Path(args.pca).name}.png"
        path = latent_pca(args.pca, out)
        print(f"latent PCA written to {path}")
        return

    if args.compare:
        out = args.out or "outputs/comparison.md"
        summary = compare_runs(args.compare[0], args.compare[1], out)
        print(format_comparison(summary))
        print(f"\nreport written to {out}")
        if args.plots:
            from expr_movements.viz import plot_metric_comparison

            for level in ("clip_level", "window_level"):
                p = Path(args.plots) / f"compare_{level}.png"
                plot_metric_comparison(summary, p, level=level)
                print(f"chart written to {p}")
        return

    if args.compare_protocols:
        out = args.out or "outputs/protocol_comparison.md"
        summary = compare_protocols(args.compare_protocols[0], args.compare_protocols[1], out)
        print(format_protocol_comparison(summary))
        print(f"\nreport written to {out}")
        return

    if not args.run:
        parser.error("one of --run / --compare / --compare-protocols / --pca is required")

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
