"""Training orchestration: config -> dataset -> splits -> model -> run dir.

The single training path for both approaches (#6). Only the config differs
between approach A (classic ML) and approach B (NN), so they share this harness,
the same windowing and — critically — the *same* subject-grouped folds, which is
what makes the A-vs-B comparison valid.

Flow:
  1. load the clip-level common contract (``sequences.npz``).
  2. for each fold from ``splits.iter_splits`` (LOSO by default):
       - window the train clips and the test clips (``data/windows``),
       - fit the standardisation scaler on **train windows only**, apply to both,
       - ``build_model`` + ``fit`` on train windows, ``predict`` test windows,
       - record per-window predictions and the clip-level majority vote.
  3. refit the model on all windows (the artifact a downstream user reloads),
     and write resolved config + model + metrics + metadata + predictions into an
     immutable run dir (``run.py``).

Metrics are reported at both the window level and the clip level (majority vote),
per fold and aggregated (mean +/- std across folds) — the LOSO std is the
subject-dependence signal the roadmap cares about.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from expr_movements.config import ExperimentConfig
from expr_movements.data.windows import (
    SequenceDataset,
    apply_scaler,
    fit_scaler,
    flatten_windows,
    make_windows,
)
from expr_movements.models.registry import build_model
from expr_movements.run import create_run_dir, write_metadata
from expr_movements.splits import iter_splits


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def _majority_vote(window_pred: np.ndarray, clip_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce per-window predictions to one prediction per clip by majority vote.

    Returns ``(clip_ids, clip_pred)`` aligned, sorted by clip id. Ties break on
    the label that sorts first (deterministic).
    """
    clip_ids = np.unique(clip_idx)
    preds = []
    for ci in clip_ids:
        votes = window_pred[clip_idx == ci]
        winner = sorted(Counter(votes).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        preds.append(winner)
    return clip_ids, np.asarray(preds, dtype=object)


def _scores(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    """Macro-F1 (primary), accuracy, per-class F1 and confusion matrix."""
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": {
            lab: float(s)
            for lab, s in zip(
                labels, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_samples": int(len(y_true)),
    }


def _aggregate(fold_scores: list[dict], key: str) -> dict:
    """Mean +/- std of a scalar metric across folds (LOSO subject-dependence)."""
    vals = [f[key]["macro_f1"] for f in fold_scores]
    accs = [f[key]["accuracy"] for f in fold_scores]
    return {
        "macro_f1_mean": float(np.mean(vals)),
        "macro_f1_std": float(np.std(vals)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
    }


def run_training(cfg: ExperimentConfig, outputs_root: str | Path = "outputs") -> Path:
    """Train the model described by ``cfg`` and return its run directory.

    Trains a fresh model per fold to compute honest held-out metrics, then
    refits on all windows for the saved artifact. Writes config, model, metrics,
    metadata and per-window predictions into the run dir.
    """
    seq_path = Path(cfg.data.processed_dir) / "sequences.npz"
    ds = SequenceDataset.load(seq_path)
    labels_sorted = sorted(set(map(str, ds.labels)))

    length, stride = cfg.window.length, cfg.window.stride

    fold_scores: list[dict] = []
    all_rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(
        iter_splits(ds.subjects, ds.labels, cfg.split, trials=ds.trials)
    ):
        train_ws = make_windows(ds, train_idx, length, stride)
        test_ws = make_windows(ds, test_idx, length, stride)

        # Normalisation statistics come from TRAIN windows only.
        mean, std = fit_scaler(train_ws)
        train_ws = apply_scaler(train_ws, mean, std)
        test_ws = apply_scaler(test_ws, mean, std)

        model = build_model(cfg.model.name, **cfg.model.params)
        model.fit(flatten_windows(train_ws), train_ws.y)
        win_pred = np.asarray(model.predict(flatten_windows(test_ws)), dtype=object)

        clip_ids, clip_pred = _majority_vote(win_pred, test_ws.clip_idx)
        clip_true = np.asarray([str(ds.labels[c]) for c in clip_ids], dtype=object)

        fold_scores.append(
            {
                "fold": fold,
                "test_subjects": sorted(set(str(ds.subjects[c]) for c in test_idx)),
                "window": _scores(test_ws.y.astype(str), win_pred.astype(str), labels_sorted),
                "clip": _scores(clip_true.astype(str), clip_pred.astype(str), labels_sorted),
            }
        )
        for ci, pred in zip(clip_ids, clip_pred):
            all_rows.append(
                {
                    "fold": fold,
                    "clip": str(ds.clips[ci]),
                    "subject": str(ds.subjects[ci]),
                    "y_true": str(ds.labels[ci]),
                    "y_pred": str(pred),
                }
            )

    if not fold_scores:
        raise ValueError(f"split strategy {cfg.split.strategy!r} produced no folds")

    # Final artifact: refit on every window with one global scaler.
    full_ws = make_windows(ds, np.arange(len(ds)), length, stride)
    mean, std = fit_scaler(full_ws)
    full_ws = apply_scaler(full_ws, mean, std)
    final_model = build_model(cfg.model.name, **cfg.model.params)
    final_model.fit(flatten_windows(full_ws), full_ws.y)

    metrics = {
        "labels": labels_sorted,
        "folds": fold_scores,
        "window_level": _aggregate(fold_scores, "window"),
        "clip_level": _aggregate(fold_scores, "clip"),
    }

    run_dir = create_run_dir(cfg, outputs_root=outputs_root)
    joblib.dump(
        {"model": final_model, "scaler_mean": mean, "scaler_std": std, "labels": labels_sorted},
        run_dir / "model.joblib",
    )
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with open(run_dir / "predictions.jsonl", "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    write_metadata(
        run_dir,
        {
            "config_name": cfg.name,
            "model": cfg.model.name,
            "split_strategy": cfg.split.strategy,
            "window": {"length": length, "stride": stride},
            "n_clips": len(ds),
            "n_folds": len(fold_scores),
            "git_commit": _git_commit(),
            "seed": cfg.seed,
            "metrics_summary": {
                "window_level": metrics["window_level"],
                "clip_level": metrics["clip_level"],
            },
        },
    )
    return run_dir
