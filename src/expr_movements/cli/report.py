"""``expr-report``: generate the whole Phase 7 (#13) analysis bundle in one go.

A thin driver over :mod:`expr_movements.evaluate` and :mod:`expr_movements.viz`
(the logic lives there; this only wires runs to outputs) so the comparison /
report / figures are produced by a **reproducible script**, not a notebook —
``notebooks/`` is EDA-only by the architecture's rule.

Given approach A (expert-features / classic ML) and approach B (the NN), each at
LOSO and — for B — intra-subject, it writes into ``--out-dir`` (default
``outputs/report``):

* ``comparison.md`` + ``compare_{clip,window}_level.png`` — A-vs-B on the same
  LOSO split and metrics.
* ``protocol_comparison.md`` — B's intra-subject vs inter-subject(LOSO) gap.
* ``latent_pca.png`` — B's multi-task latent (emotion=colour, subject=marker).
* ``confusion_{A,B}.png`` — clip-level confusion for each approach.

Example::

    uv run expr-report \\
        --run-a outputs/rf_1f7822cf \\
        --run-b outputs/cnn1d_4359693b \\
        --run-b-intra outputs/cnn1d_intra_f52b073c

``--run-b-intra`` is optional; omit it to skip the intra-vs-LOSO section.
Requires the figure deps: ``uv sync --extra viz``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from expr_movements.evaluate import (
    compare_protocols,
    compare_runs,
    evaluate_run,
)


def generate_report(
    run_a: str | Path,
    run_b: str | Path,
    out_dir: str | Path = "outputs/report",
    run_b_intra: str | Path | None = None,
) -> Path:
    """Write the full Phase 7 bundle into ``out_dir`` and return it.

    ``run_a`` is approach A (expert features, classic ML), ``run_b`` approach B
    (the NN), both at LOSO. ``run_b_intra`` is B's intra-subject run for the
    subject-dependence gap section (optional).
    """
    from expr_movements.viz import (
        latent_pca,
        plot_confusion_matrix,
        plot_metric_comparison,
    )

    out_dir = Path(out_dir)
    figs = out_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1. A vs B head-to-head (same LOSO split, same eval code).
    ab = compare_runs(run_a, run_b, out_path=out_dir / "comparison.md")
    written.append(out_dir / "comparison.md")
    for level in ("clip_level", "window_level"):
        p = figs / f"compare_{level}.png"
        plot_metric_comparison(ab, p, level=level)
        written.append(p)

    # 2. Intra-subject vs inter-subject(LOSO) for B — the subject-dependence gap.
    if run_b_intra is not None:
        compare_protocols(run_b_intra, run_b, out_path=out_dir / "protocol_comparison.md")
        written.append(out_dir / "protocol_comparison.md")

    # 3. Latent PCA of B's multi-task latent (emotion=colour, subject=marker).
    pca_path = figs / "latent_pca.png"
    latent_pca(run_b, pca_path, title="Multi-task latent (NN)")
    written.append(pca_path)

    # 4. Clip-level confusion matrices for each approach (Joy is the hard class).
    for run, tag in ((run_a, "A_expert"), (run_b, "B_nn")):
        s = evaluate_run(run)
        p = figs / f"confusion_{tag}.png"
        plot_confusion_matrix(
            s["confusion_matrix"]["clip"], s["labels"], p, title=f"Confusion (clip) — {tag}"
        )
        written.append(p)

    for p in written:
        print(f"wrote {p}")
    print(f"\nPhase 7 report bundle in {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the Phase 7 comparison/report bundle.")
    parser.add_argument("--run-a", required=True, help="approach A run dir (expert features, LOSO)")
    parser.add_argument("--run-b", required=True, help="approach B run dir (NN, LOSO)")
    parser.add_argument("--run-b-intra", help="approach B intra-subject run dir (optional)")
    parser.add_argument("--out-dir", default="outputs/report", help="output dir for the bundle")
    args = parser.parse_args(argv)

    generate_report(args.run_a, args.run_b, args.out_dir, args.run_b_intra)


if __name__ == "__main__":
    main()
