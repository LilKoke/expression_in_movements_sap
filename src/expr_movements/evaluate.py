"""Phase 6 (#7): the shared evaluation read-out over a run dir.

Both teams (NN / expert-features) write a run dir via the common harness, so the
metrics they report are computed by the *same* code in ``train.py`` during the
fold loop and persisted to ``outputs/<name>_<hash>/metrics.json``. This module is
the consumer side: it loads that persisted ``metrics.json`` and folds it into a
single self-describing summary — the figures the roadmap's Phase 6 checklist asks
for, in one place, for any run regardless of which team produced it.

Why read the persisted metrics rather than re-infer here: the held-out
reconstruction error and latent-separability numbers depend on the *per-fold*
models (one trained per LOSO fold), which only exist inside the fold loop. The
saved artifact is refit on all data, so re-inferring from it would not reproduce
the held-out latent. Computing them in ``train.py`` and reading them back here
keeps a run reproducible from its own folder and ``evaluate_run`` deterministic.

What ``evaluate_run`` surfaces (from ``metrics.json``):

* **Macro-F1 (primary)**, accuracy and balanced accuracy as ``mean ± std`` across
  folds, at **both** the window level and the trial/clip level (majority vote).
  The LOSO std is the subject-dependence signal.
* Per-class F1 and the confusion matrix (aggregated across folds).
* The split protocol, so an intra-subject run reads as such next to a LOSO one
  (the intra-vs-inter gap is itself the headline finding).
* For the multi-task NN: held-out **reconstruction MSE** and latent
  **Silhouette / Davies-Bouldin**, when the run recorded them.

``compare_runs`` (the head-to-head A-vs-B write-up) is Phase 7; it stays a stub
here so the import surface is stable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _agg_line(block: dict, metric: str) -> str:
    """Format a ``mean ± std`` line for one aggregated metric, if present."""
    mean = block.get(f"{metric}_mean")
    std = block.get(f"{metric}_std")
    if mean is None:
        return ""
    return f"{mean:.4f} ± {std:.4f}"


def _aggregate_per_class_f1(folds: list[dict], level: str, labels: list[str]) -> dict:
    """Mean per-class F1 across folds at ``level`` ('window' | 'clip')."""
    out = {}
    for lab in labels:
        vals = [f[level]["per_class_f1"].get(lab) for f in folds if level in f]
        vals = [v for v in vals if v is not None]
        out[lab] = float(np.mean(vals)) if vals else None
    return out


def _sum_confusion(folds: list[dict], level: str, labels: list[str]) -> list[list[int]]:
    """Element-wise sum of the per-fold confusion matrices at ``level``.

    Folds partition the data (LOSO) or are pooled CV rounds (intra-subject); in
    both cases summing the per-fold confusions gives the overall confusion over
    every held-out prediction, which is what the report shows.
    """
    n = len(labels)
    total = np.zeros((n, n), dtype=int)
    for f in folds:
        cm = f.get(level, {}).get("confusion_matrix")
        if cm is not None:
            total += np.asarray(cm, dtype=int)
    return total.tolist()


def evaluate_run(run_dir: str | Path) -> dict:
    """Load a run's persisted metrics and return the Phase 6 evaluation summary.

    ``run_dir`` is an ``outputs/<name>_<hash>/`` directory written by
    ``train.run_training``. Returns a dict with the aggregated window- and
    clip-level scores, per-class F1, summed confusion matrices, the split
    protocol, and (for the multi-task NN) reconstruction / latent-separability
    figures. Raises ``FileNotFoundError`` if the run has no ``metrics.json``.
    """
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"no metrics.json in run dir {run_dir!r}")

    metrics = json.loads(metrics_path.read_text())
    labels = metrics.get("labels", [])
    folds = metrics.get("folds", [])

    summary = {
        "run_dir": str(run_dir),
        "labels": labels,
        "split_strategy": metrics.get("split_strategy"),
        "n_folds": len(folds),
        "window_level": metrics.get("window_level", {}),
        "clip_level": metrics.get("clip_level", {}),
        "per_class_f1": {
            "window": _aggregate_per_class_f1(folds, "window", labels),
            "clip": _aggregate_per_class_f1(folds, "clip", labels),
        },
        "confusion_matrix": {
            "window": _sum_confusion(folds, "window", labels),
            "clip": _sum_confusion(folds, "clip", labels),
        },
    }
    # NN-only blocks: present iff the run recorded them.
    if "reconstruction" in metrics:
        summary["reconstruction"] = metrics["reconstruction"]
    if "separability" in metrics:
        summary["separability"] = metrics["separability"]
    return summary


def format_report(summary: dict) -> str:
    """Render :func:`evaluate_run`'s summary as a human-readable markdown report."""
    labels = summary["labels"]
    lines: list[str] = []
    lines.append(f"# Evaluation — {Path(summary['run_dir']).name}")
    lines.append("")
    lines.append(f"- Split protocol: **{summary['split_strategy']}** ({summary['n_folds']} folds)")
    lines.append(f"- Classes: {', '.join(labels)}")
    lines.append("")

    for level, title in (("clip_level", "Trial/clip level (majority vote)"), ("window_level", "Window level")):
        block = summary.get(level, {})
        if not block:
            continue
        lines.append(f"## {title}")
        lines.append(f"- Macro-F1 (primary): {_agg_line(block, 'macro_f1')}")
        lines.append(f"- Accuracy: {_agg_line(block, 'accuracy')}")
        lines.append(f"- Balanced accuracy: {_agg_line(block, 'balanced_accuracy')}")
        pc = summary["per_class_f1"]["clip" if level == "clip_level" else "window"]
        pc_str = ", ".join(
            f"{lab}={v:.3f}" for lab, v in pc.items() if v is not None
        )
        if pc_str:
            lines.append(f"- Per-class F1: {pc_str}")
        cm = summary["confusion_matrix"]["clip" if level == "clip_level" else "window"]
        if cm:
            lines.append("- Confusion matrix (rows=true, cols=pred; order = classes above):")
            lines.append("")
            header = "  | " + " | ".join([""] + labels) + " |"
            sep = "  |" + "---|" * (len(labels) + 1)
            lines.append(header)
            lines.append(sep)
            for lab, row in zip(labels, cm):
                lines.append("  | " + " | ".join([lab] + [str(c) for c in row]) + " |")
        lines.append("")

    if "reconstruction" in summary:
        r = summary["reconstruction"]
        lines.append("## Reconstruction (multi-task NN, held-out)")
        lines.append(
            f"- Overall MSE: {r['overall_mse_mean']:.5f} ± {r['overall_mse_std']:.5f}"
        )
        lines.append("")
    if "separability" in summary:
        s = summary["separability"]
        lines.append("## Latent separability (held-out subjects)")
        if s.get("silhouette"):
            lines.append(
                f"- Silhouette (higher=better): {s['silhouette']['mean']:.4f} ± {s['silhouette']['std']:.4f}"
            )
        if s.get("davies_bouldin"):
            lines.append(
                f"- Davies-Bouldin (lower=better): {s['davies_bouldin']['mean']:.4f} ± {s['davies_bouldin']['std']:.4f}"
            )
        lines.append("")

    return "\n".join(lines)


def compare_runs(run_a: str | Path, run_b: str | Path, out_path: str | Path) -> Path:
    """Compare two runs (approach A vs B) and write a report. Phase 7 (#8)."""
    raise NotImplementedError("A-vs-B comparison lands in Phase 7 (#8)")
