"""Phase 7 (#13): figures for the comparison / report / 2-minute video.

The numbers live in :mod:`expr_movements.evaluate`; this module turns them — and
the saved multi-task latent — into the figures the roadmap's Phase 7 asks for:

* :func:`latent_pca` — PCA of the multi-task NN latent ``z``, **coloured by
  emotion and marker-shaped by subject**. This is the headline interpretability
  figure: a clean emotion separation that nonetheless clusters by subject is the
  visual form of the subject-dependence story.
* :func:`plot_confusion_matrix` — a run's confusion matrix as a heatmap (the
  source paper finds Joy the hard class; this is where to see it).
* :func:`plot_metric_comparison` — A-vs-B headline metrics as grouped bars from a
  :func:`expr_movements.evaluate.compare_runs` summary.

matplotlib is imported lazily inside each plotting function so importing this
module (and the rest of the eval surface) never requires it; install the ``viz``
extra (``uv sync --extra viz``) to actually render.

On the latent source: the per-fold held-out models are not persisted (only the
final model refit on all data is), so :func:`latent_pca` encodes every window
with that saved final model. The figure is therefore a *visualisation* of the
learned latent over the whole dataset, not a held-out generalisation estimate —
the held-out separability **numbers** (Silhouette / Davies-Bouldin) in
``metrics.json`` are the honest generalisation signal, and the two are meant to
be read together.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import yaml

from expr_movements.data.windows import (
    SequenceDataset,
    apply_scaler,
    make_windows,
)


def _load_run_config(run_dir: Path) -> dict:
    """Read a run's resolved ``config.yaml`` (window length/stride, data dirs)."""
    with open(run_dir / "config.yaml") as f:
        return yaml.safe_load(f)


def latent_windows(run_dir: str | Path) -> dict:
    """Encode every window of a NN run's dataset into the multi-task latent.

    Reloads the run's saved model + scaler, re-windows the dataset its config
    points at, applies the **saved** scaler, and runs the model's ``transform``
    to get the per-window latent ``z``. Returns ``{z, labels, subjects, clips}``
    aligned per window (``z`` is ``(n_windows, latent_dim)``).

    Raises ``ValueError`` if the run's model is not a latent-exposing NN (classic
    ML has no ``transform``) — only approach B has a latent to visualise.
    """
    run_dir = Path(run_dir)
    bundle = joblib.load(run_dir / "model.joblib")
    model = bundle["model"]
    if not hasattr(model, "transform"):
        raise ValueError(
            f"run {run_dir.name!r} has no latent to visualise "
            "(its model exposes no transform; PCA is NN-only)"
        )

    cfg = _load_run_config(run_dir)
    processed = Path(cfg["data"]["processed_dir"])
    ds = SequenceDataset.load(processed / "sequences.npz")
    length, stride = cfg["window"]["length"], cfg["window"]["stride"]

    ws = make_windows(ds, np.arange(len(ds)), length, stride)
    ws = apply_scaler(ws, bundle["scaler_mean"], bundle["scaler_std"])
    z = np.asarray(model.transform(ws))

    return {
        "z": z,
        "labels": ws.y.astype(str),
        "subjects": np.asarray([str(ds.subjects[c]) for c in ws.clip_idx]),
        "clips": np.asarray([str(ds.clips[c]) for c in ws.clip_idx]),
    }


def latent_pca(run_dir: str | Path, out_path: str | Path, *, title: str | None = None) -> Path:
    """PCA-scatter the NN latent, coloured by emotion and marker-shaped by subject.

    Reduces the per-window latent ``z`` to 2 PCs and scatters it with one colour
    per emotion and one marker per subject, so the reader can see both axes of
    structure at once: are the emotions separable, and does each emotion still
    split by who is walking (subject dependence). Writes a PNG to ``out_path`` and
    returns the path. The PC axis labels carry the explained-variance ratio.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: write files, never open a window
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    data = latent_windows(run_dir)
    z, labels, subjects = data["z"], data["labels"], data["subjects"]

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(z)
    evr = pca.explained_variance_ratio_

    emotions = sorted(set(labels))
    subs = sorted(set(subjects))
    cmap = plt.get_cmap("tab10")
    colour = {e: cmap(i % 10) for i, e in enumerate(emotions)}
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker = {s: markers[i % len(markers)] for i, s in enumerate(subs)}

    fig, ax = plt.subplots(figsize=(7, 6))
    for e in emotions:
        for s in subs:
            sel = (labels == e) & (subjects == s)
            if not sel.any():
                continue
            ax.scatter(
                pcs[sel, 0],
                pcs[sel, 1],
                c=[colour[e]],
                marker=marker[s],
                s=22,
                alpha=0.6,
                edgecolors="none",
            )
    # Two legends: colour = emotion, marker = subject.
    emo_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colour[e], label=e) for e in emotions
    ]
    sub_handles = [
        plt.Line2D([], [], marker=marker[s], linestyle="", color="0.3", label=s) for s in subs
    ]
    leg1 = ax.legend(handles=emo_handles, title="Emotion", loc="upper right", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=sub_handles, title="Subject", loc="lower right", fontsize=8)

    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% var)")
    ax.set_title(title or f"Latent PCA — {Path(run_dir).name}")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    labels: list[str],
    out_path: str | Path,
    *,
    title: str = "Confusion matrix",
) -> Path:
    """Render a confusion matrix (rows=true, cols=pred) as an annotated heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 1.5, 1.4 * len(labels) + 1))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _mean_of_cell(cell: str) -> float | None:
    """Parse the ``mean`` out of an ``evaluate`` ``"mean ± std"`` cell, or None."""
    if not cell or cell == "—":
        return None
    try:
        return float(cell.split("±")[0].strip())
    except ValueError:
        return None


def plot_metric_comparison(
    comparison: dict, out_path: str | Path, *, level: str = "clip_level"
) -> Path:
    """Grouped bar-chart of A-vs-B headline metrics at ``level`` (clip or window).

    Takes a :func:`expr_movements.evaluate.compare_runs` summary and draws
    Macro-F1 / accuracy / balanced accuracy as paired bars (A vs B). Useful as
    the single "who wins" slide for the video.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = comparison["table"]
    runs = list(table.keys())
    metrics = ["macro_f1", "accuracy", "balanced_accuracy"]
    pretty = ["Macro-F1", "Accuracy", "Balanced acc."]

    x = np.arange(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, run in enumerate(runs):
        vals = [_mean_of_cell(table[run][level].get(m, "—")) for m in metrics]
        vals = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(x + (k - 0.5) * width, vals, width, label=run)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x, pretty)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (mean over folds)")
    ax.set_title(f"Approach comparison — {level.replace('_', ' ')}")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
