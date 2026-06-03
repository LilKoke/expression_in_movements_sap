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

``compare_runs`` (Phase 7, #13) is the head-to-head A-vs-B write-up: it reads two
runs through :func:`evaluate_run` and renders one report putting approach A
(expert features) and approach B (the NN) side by side on the *same* metrics.
``compare_protocols`` is the companion intra-vs-inter(LOSO) read-out for one
approach — the gap between the two is itself the subject-dependence finding the
roadmap asks us to argue. Both assert the runs they fold together are actually
comparable (same label space; same split for A-vs-B) and refuse otherwise, so a
comparison can't silently put apples next to oranges. The latent-PCA figure that
rounds out Phase 7 lives in :mod:`expr_movements.viz` (it needs matplotlib).
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
    if "early_stopping" in metrics:
        summary["early_stopping"] = metrics["early_stopping"]
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

    for level, title in (
        ("clip_level", "Trial/clip level (majority vote)"),
        ("window_level", "Window level"),
    ):
        block = summary.get(level, {})
        if not block:
            continue
        lines.append(f"## {title}")
        lines.append(f"- Macro-F1 (primary): {_agg_line(block, 'macro_f1')}")
        lines.append(f"- Accuracy: {_agg_line(block, 'accuracy')}")
        lines.append(f"- Balanced accuracy: {_agg_line(block, 'balanced_accuracy')}")
        pc = summary["per_class_f1"]["clip" if level == "clip_level" else "window"]
        pc_str = ", ".join(f"{lab}={v:.3f}" for lab, v in pc.items() if v is not None)
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
        lines.append(f"- Overall MSE: {r['overall_mse_mean']:.5f} ± {r['overall_mse_std']:.5f}")
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

    if "early_stopping" in summary:
        es = summary["early_stopping"]
        lines.append("## Early stopping")
        lines.append(
            f"- Validation: **{es.get('strategy')}** split, monitor=`{es.get('monitor')}`, "
            f"patience={es.get('patience')}"
        )
        lines.append(f"- Folds using early stopping: {es.get('folds_used')}")
        er = es.get("epochs_run")
        if er:
            lines.append(f"- Epochs run: {er['mean']:.1f} ± {er['std']:.1f}")
        lines.append("")

    return "\n".join(lines)


def _metric_cell(block: dict, metric: str) -> str:
    """``mean ± std`` for one aggregated metric, or ``—`` when absent."""
    line = _agg_line(block, metric)
    return line if line else "—"


def _label_run(summary: dict) -> str:
    """A short human label for a run in a comparison table (its dir name)."""
    return Path(summary["run_dir"]).name


def compare_runs(run_a: str | Path, run_b: str | Path, out_path: str | Path | None = None) -> dict:
    """Head-to-head comparison of two runs (approach A vs B). Phase 7 (#13).

    Reads both runs through :func:`evaluate_run` and folds them into one
    self-describing summary that puts approach A (expert-features, conventionally
    ``run_a``) and approach B (the NN, ``run_b``) side by side on the *same*
    metrics. The two runs must share the **same label space** and the **same
    split protocol** — otherwise the comparison is not like-for-like and a
    ``ValueError`` is raised. When ``out_path`` is given, the markdown report from
    :func:`format_comparison` is written there.

    Returns a dict with both per-run summaries under ``a``/``b`` and a flat
    ``table`` of the headline window/clip metrics for each, ready to render or to
    drive a bar-chart in :mod:`expr_movements.viz`.
    """
    sa = evaluate_run(run_a)
    sb = evaluate_run(run_b)

    if sorted(sa["labels"]) != sorted(sb["labels"]):
        raise ValueError(
            f"runs have different label spaces ({sa['labels']} vs {sb['labels']}); "
            "they are not comparable"
        )
    if sa["split_strategy"] != sb["split_strategy"]:
        raise ValueError(
            f"runs use different split protocols ({sa['split_strategy']!r} vs "
            f"{sb['split_strategy']!r}); an A-vs-B comparison must use the same split. "
            "Use compare_protocols for an intra-vs-inter read-out instead."
        )

    table = {
        _label_run(sa): _headline_metrics(sa),
        _label_run(sb): _headline_metrics(sb),
    }
    summary = {
        "kind": "compare_runs",
        "split_strategy": sa["split_strategy"],
        "labels": sa["labels"],
        "a": sa,
        "b": sb,
        "table": table,
    }
    if out_path is not None:
        report = format_comparison(summary)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
    return summary


def _headline_metrics(summary: dict) -> dict:
    """Pull the headline window/clip scores out of an ``evaluate_run`` summary.

    A flat ``{level: {metric: "mean ± std"}}`` mapping plus the NN-only latent
    separability when present — the numbers a comparison table or bar-chart needs.
    """
    out: dict = {}
    for level in ("clip_level", "window_level"):
        block = summary.get(level, {})
        out[level] = {
            "macro_f1": _metric_cell(block, "macro_f1"),
            "accuracy": _metric_cell(block, "accuracy"),
            "balanced_accuracy": _metric_cell(block, "balanced_accuracy"),
        }
    sep = summary.get("separability")
    if sep:
        out["separability"] = {
            "silhouette": (sep.get("silhouette") or {}).get("mean"),
            "davies_bouldin": (sep.get("davies_bouldin") or {}).get("mean"),
        }
    return out


def format_comparison(summary: dict) -> str:
    """Render :func:`compare_runs`'s summary as a side-by-side markdown report."""
    a, b = summary["a"], summary["b"]
    name_a, name_b = _label_run(a), _label_run(b)
    labels = summary["labels"]
    lines: list[str] = []
    lines.append(f"# Approach comparison — {name_a} vs {name_b}")
    lines.append("")
    lines.append(f"- Split protocol: **{summary['split_strategy']}** (same for both runs)")
    lines.append(f"- Classes: {', '.join(labels)}")
    lines.append(f"- A = `{name_a}`  ·  B = `{name_b}`")
    lines.append("")

    for level, title in (
        ("clip_level", "Trial/clip level (majority vote)"),
        ("window_level", "Window level"),
    ):
        if not (a.get(level) and b.get(level)):
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Metric | A | B |")
        lines.append("|---|---|---|")
        for metric, pretty in (
            ("macro_f1", "Macro-F1 (primary)"),
            ("accuracy", "Accuracy"),
            ("balanced_accuracy", "Balanced accuracy"),
        ):
            lines.append(
                f"| {pretty} | {_metric_cell(a[level], metric)} | {_metric_cell(b[level], metric)} |"
            )
        # Per-class F1 (clip vs window key matches evaluate_run's nesting).
        pc_key = "clip" if level == "clip_level" else "window"
        lines.append("")
        lines.append("Per-class F1:")
        lines.append("")
        lines.append("| Class | A | B |")
        lines.append("|---|---|---|")
        pca, pcb = a["per_class_f1"][pc_key], b["per_class_f1"][pc_key]
        for lab in labels:
            va, vb = pca.get(lab), pcb.get(lab)
            lines.append(
                f"| {lab} | {va:.3f} | {vb:.3f} |"
                if va is not None and vb is not None
                else f"| {lab} | {'—' if va is None else f'{va:.3f}'} | "
                f"{'—' if vb is None else f'{vb:.3f}'} |"
            )
        lines.append("")

    # Latent separability is NN-only; show it when at least one side recorded it.
    sep_a = summary["table"][name_a].get("separability")
    sep_b = summary["table"][name_b].get("separability")
    if sep_a or sep_b:
        lines.append("## Latent separability (held-out, NN only)")
        lines.append("")
        lines.append("| Metric | A | B |")
        lines.append("|---|---|---|")

        def _fmt(d, k):
            v = (d or {}).get(k)
            return f"{v:.4f}" if v is not None else "—"

        lines.append(
            f"| Silhouette (↑) | {_fmt(sep_a, 'silhouette')} | {_fmt(sep_b, 'silhouette')} |"
        )
        lines.append(
            f"| Davies-Bouldin (↓) | {_fmt(sep_a, 'davies_bouldin')} | {_fmt(sep_b, 'davies_bouldin')} |"
        )
        lines.append("")

    return "\n".join(lines)


def compare_protocols(
    intra_run: str | Path, loso_run: str | Path, out_path: str | Path | None = None
) -> dict:
    """Intra-subject vs inter-subject(LOSO) read-out for one approach. Phase 7 (#13).

    The roadmap wants both numbers: intra-subject is directly comparable to the
    source paper (Venture 2014, >90%), while inter-subject (LOSO) is the real
    task. The **gap** between them is the headline subject-dependence finding —
    how much harder generalising to an unseen person is. The two runs should be
    the *same model/config* differing only in split protocol; we assert they are
    indeed an intra run and a LOSO run with the same labels, and surface the gap.
    """
    si = evaluate_run(intra_run)
    sl = evaluate_run(loso_run)
    if sorted(si["labels"]) != sorted(sl["labels"]):
        raise ValueError("runs have different label spaces; not comparable")
    if "intra" not in (si["split_strategy"] or ""):
        raise ValueError(
            f"intra_run split is {si['split_strategy']!r}, expected an intra-subject run"
        )
    if "subject_out" not in (sl["split_strategy"] or "") and "loso" not in (
        sl["split_strategy"] or ""
    ):
        raise ValueError(
            f"loso_run split is {sl['split_strategy']!r}, expected a leave-one-subject-out run"
        )

    def _macro(s, level):
        return (s.get(level) or {}).get("macro_f1_mean")

    gap = {}
    for level in ("clip_level", "window_level"):
        mi, ml = _macro(si, level), _macro(sl, level)
        gap[level] = (mi - ml) if (mi is not None and ml is not None) else None

    summary = {
        "kind": "compare_protocols",
        "labels": si["labels"],
        "intra": si,
        "loso": sl,
        "macro_f1_gap": gap,
    }
    if out_path is not None:
        report = format_protocol_comparison(summary)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
    return summary


def format_protocol_comparison(summary: dict) -> str:
    """Render :func:`compare_protocols`'s summary as a markdown report."""
    si, sl = summary["intra"], summary["loso"]
    lines: list[str] = []
    lines.append("# Intra-subject vs inter-subject (LOSO)")
    lines.append("")
    lines.append(f"- Intra run: `{_label_run(si)}` ({si['split_strategy']})")
    lines.append(f"- LOSO run: `{_label_run(sl)}` ({sl['split_strategy']})")
    lines.append(
        "- Intra is directly comparable to Venture 2014 (>90%); LOSO is the real "
        "(unseen-subject) task. The gap = subject dependence."
    )
    lines.append("")
    lines.append("| Macro-F1 | Intra-subject | Inter-subject (LOSO) | Gap (intra − LOSO) |")
    lines.append("|---|---|---|---|")
    for level, pretty in (("clip_level", "Trial/clip"), ("window_level", "Window")):
        bi, bl = si.get(level, {}), sl.get(level, {})
        g = summary["macro_f1_gap"].get(level)
        g_str = f"{g:+.4f}" if g is not None else "—"
        lines.append(
            f"| {pretty} | {_metric_cell(bi, 'macro_f1')} | {_metric_cell(bl, 'macro_f1')} | {g_str} |"
        )
    lines.append("")
    return "\n".join(lines)
