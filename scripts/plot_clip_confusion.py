from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


DEFAULT_LABELS = ["angry", "happy", "neutral", "sad"]


def is_clip_record(rec: dict) -> bool:
    """Best-effort判定で clip-level レコードだけを使う。"""
    for key in ("level", "unit", "kind", "record_type"):
        if key in rec:
            v = str(rec[key]).lower()
            if v in {"clip", "clip_level", "trial", "trial_level"}:
                return True
            if v in {"window", "window_level"}:
                return False

    if "window_idx" in rec:
        return False

    if "clip_pred" in rec or "majority_pred" in rec:
        return True

    if "clip_idx" in rec and "window_idx" not in rec:
        return True

    # predictions.jsonl が clip-level 行だけのケースを想定
    return True


def pick_first(rec: dict, candidates: list[str]) -> str | None:
    for key in candidates:
        if key in rec and rec[key] is not None:
            return rec[key]
    return None


def extract_true_pred(rec: dict) -> tuple[str, str] | None:
    true_candidates = [
        "y_true",
        "true_label",
        "label_true",
        "target",
        "label",
    ]
    pred_candidates = [
        "y_pred",
        "pred_label",
        "label_pred",
        "pred",
        "prediction",
        "clip_pred",
        "majority_pred",
    ]

    y_true = pick_first(rec, true_candidates)
    y_pred = pick_first(rec, pred_candidates)

    if y_true is None or y_pred is None:
        return None

    return str(y_true), str(y_pred)


def load_clip_predictions(predictions_path: Path) -> tuple[list[str], list[str]]:
    y_true = []
    y_pred = []

    with predictions_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)

            if not is_clip_record(rec):
                continue

            pair = extract_true_pred(rec)
            if pair is None:
                continue

            t, p = pair
            y_true.append(t)
            y_pred.append(p)

    if not y_true:
        raise ValueError(
            "No clip-level predictions were found in predictions.jsonl. "
            "If this happens, please show me the first 3-5 lines of the file."
        )

    return y_true, y_pred


def load_label_order(run_dir: Path) -> list[str]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                m = json.load(f)
            labels = m.get("labels")
            if isinstance(labels, list) and labels:
                return [str(x) for x in labels]
        except Exception:
            pass
    return DEFAULT_LABELS


def plot_confusion_matrix_image(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    out_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{title}\nClip-level confusion matrix  |  accuracy = {acc:.3f}")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a clip-level confusion matrix image from one run."
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Run directory, e.g. outputs/rf_expert_ff1c9e75",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output PNG path. If omitted, save under reports/confusion/.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Not found: {predictions_path}")

    labels = load_label_order(run_dir)
    y_true, y_pred = load_clip_predictions(predictions_path)

    if args.out is None:
        out_path = Path("reports/confusion") / f"{run_dir.name}_clip_confusion_matrix.png"
    else:
        out_path = Path(args.out)

    plot_confusion_matrix_image(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        out_path=out_path,
        title=run_dir.name,
    )

    acc = accuracy_score(y_true, y_pred)
    print("run:", run_dir)
    print("n_clips:", len(y_true))
    print("accuracy:", f"{acc:.4f}")
    print("saved:", out_path)


if __name__ == "__main__":
    main()
