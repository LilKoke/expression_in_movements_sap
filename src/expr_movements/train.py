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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)

from expr_movements.config import ExperimentConfig
from expr_movements.data.windows import (
    SequenceDataset,
    WindowSet,
    apply_scaler,
    fit_scaler,
    flatten_windows,
    make_windows,
)
from expr_movements.models.base import BaseClassifier
from expr_movements.models.registry import build_model
from expr_movements.run import create_run_dir, write_metadata
from expr_movements.splits import iter_splits, nested_validation_split


def _is_nn(model: BaseClassifier) -> bool:
    """True for window-consuming NN models — the only ones that support early stop."""
    return getattr(model, "consumes", "table") == "window"


def _fit_fold(
    model: BaseClassifier,
    ds: SequenceDataset,
    train_idx: np.ndarray,
    train_ws: WindowSet,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: ExperimentConfig,
    length: int,
    stride: int,
) -> dict | None:
    """Fit ``model`` on a fold, with early stopping when enabled for an NN.

    Returns ``None`` for the plain fixed-epochs path, or an info dict
    (validation strategy, held-out subjects, best score, epochs run) when early
    stopping was actually used. Early stopping is skipped — silently, falling
    back to fixed epochs — for classic ML, or when the nested split is
    impossible for this fold (e.g. a single train subject); the reason is
    returned in the dict's ``note`` so the run records why.
    """
    if not (cfg.validation.enabled and _is_nn(model)):
        if cfg.validation.enabled and not _is_nn(model):
            # Classic ML has no epochs; ES does not apply. Record it once via return.
            model.fit(_model_X(model, train_ws), train_ws.y)
            return {"used": False, "note": "early stopping is NN-only; classic ML trained normally"}
        model.fit(_model_X(model, train_ws), train_ws.y)
        return None

    # Carve a validation set out of this fold's train side (clip-level), window
    # it with the SAME scaler, and early-stop on it.
    try:
        fit_pos, val_pos = nested_validation_split(
            ds.subjects[train_idx], ds.labels[train_idx], cfg.validation
        )
    except ValueError as e:
        model.fit(_model_X(model, train_ws), train_ws.y)
        return {"used": False, "note": f"fell back to fixed epochs: {e}"}

    fit_clip_idx = train_idx[fit_pos]
    val_clip_idx = train_idx[val_pos]
    fit_ws = apply_scaler(make_windows(ds, fit_clip_idx, length, stride), mean, std)
    val_ws = apply_scaler(make_windows(ds, val_clip_idx, length, stride), mean, std)

    model.fit(
        _model_X(model, fit_ws),
        fit_ws.y,
        validation_data=(_model_X(model, val_ws), val_ws.y),
        patience=cfg.validation.patience,
        monitor=cfg.validation.monitor,
        min_delta=cfg.validation.min_delta,
        restore_best=cfg.validation.restore_best,
    )
    return {
        "used": True,
        "strategy": cfg.validation.strategy,
        "monitor": cfg.validation.monitor,
        "val_subjects": sorted(set(str(ds.subjects[c]) for c in val_clip_idx)),
        "n_val_clips": int(len(val_clip_idx)),
        "best_score": getattr(model, "best_score_", None),
        "epochs_run": getattr(model, "stopped_epoch_", None),
    }


EXPERT_FEATURE_COLUMNS = (
    "walking_speed",
    "stride_length_proxy",
    "arm_swing_mean",
    "head_vertical_range",
)


def _expert_feature_X(model: BaseClassifier, ws: WindowSet) -> np.ndarray:
    """Return clip-level expert features broadcast to each source window."""

    import pandas as pd

    features_path = Path(getattr(model, "features_path", "data/processed/features.parquet"))
    df = pd.read_parquet(features_path)

    required = {"clip_idx", *EXPERT_FEATURE_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"expert feature table missing columns: {sorted(missing)}")

    feature_df = df.set_index("clip_idx")
    missing_clips = sorted(set(map(int, ws.clip_idx)) - set(map(int, feature_df.index)))
    if missing_clips:
        raise ValueError(f"expert feature table missing clip_idx values: {missing_clips[:10]}")

    x = feature_df.loc[ws.clip_idx, list(EXPERT_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    return x


def _model_X(model: BaseClassifier, ws: WindowSet):
    """Hand each model the window representation it declares via ``consumes``.

    Classic ML wants a flat ``(n, length*F)`` feature table; the NN keeps the
    ``(n, length, F)`` shape (and the mask). Expert-feature models consume a
    precomputed clip-level feature table, broadcast to each source window.
    This is the one place the shared harness forks on approach.
    """
    consumes = getattr(model, "consumes", "table")
    if consumes == "window":
        return ws
    if consumes == "expert":
        return _expert_feature_X(model, ws)
    return flatten_windows(ws)


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
    """Macro-F1 (primary), accuracy, balanced accuracy, per-class F1, confusion.

    Balanced accuracy (mean per-class recall) is reported alongside plain
    accuracy because the four emotions are only *roughly* balanced and the
    confusion is class-skewed (the source paper finds Joy the hard class), so a
    metric that weights every class equally is the honest accuracy figure.
    """
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_f1": {
            lab: float(s)
            for lab, s in zip(
                labels, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_samples": int(len(y_true)),
    }


def _is_multitask(model: BaseClassifier) -> bool:
    """True for the NN multi-task model — it can reconstruct and expose a latent.

    The reconstruction error and latent-separability metrics only make sense for
    approach B (the multi-task net). We detect it structurally (a torch ``net_``
    with a ``decoder`` + a ``transform``) rather than by class name so the LSTM /
    GRU baselines, which share the wrapper, are covered too.
    """
    net = getattr(model, "net_", None)
    return net is not None and hasattr(net, "decoder") and hasattr(model, "transform")


def _reconstruction_mse(model: BaseClassifier, ws: WindowSet) -> dict | None:
    """Held-out reconstruction error of the multi-task net over ``ws``.

    Returns overall MSE plus the per-frame and per-joint breakdowns the roadmap
    asks for, averaged over the real (unmasked) frames only — the model is never
    scored on the zero-padding of short clips. ``None`` for non-NN models.

    Per-joint is reported over the *pose* channels (xyz triples); the trailing
    speed channel, when present, is excluded from the joint breakdown and folded
    into the overall figure only. Returns ``None`` if there are no windows.
    """
    if not _is_multitask(model) or ws.X.shape[0] == 0:
        return None
    import torch

    net = model.net_
    net.eval()
    with torch.no_grad():
        xb = torch.from_numpy(ws.X.astype(np.float32))
        mb = torch.from_numpy(ws.mask.astype(np.float32))
        _, recon, _ = net(xb, mb)
        sq = (recon - xb) ** 2  # (B, T, F)
        m = mb.unsqueeze(-1)  # (B, T, 1)
        denom_bt = mb.sum().clamp_min(1.0)

        overall = float((sq.mean(dim=2) * mb).sum() / denom_bt)
        # Per-frame index within the window: mean squared error at frame t over
        # every real window-row and feature (how recon quality varies along time).
        per_frame_num = (sq.mean(dim=2) * mb).sum(dim=0)  # (T,)
        per_frame_den = mb.sum(dim=0).clamp_min(1.0)  # (T,)
        per_frame = (per_frame_num / per_frame_den).cpu().numpy()
        # Per-feature: mean squared error of each channel over real frames.
        per_feat_num = (sq * m).sum(dim=(0, 1))  # (F,)
        per_feat = (per_feat_num / denom_bt).cpu().numpy()

    n_pose = per_feat.shape[0] - (1 if ws.has_speed else 0)
    out = {
        "overall_mse": overall,
        "per_frame_mse": per_frame.astype(float).tolist(),
        "per_feature_mse": per_feat.astype(float).tolist(),
        "n_windows": int(ws.X.shape[0]),
    }
    if n_pose % 3 == 0 and n_pose > 0:
        # xyz are contiguous per joint -> mean the three coords of each joint.
        per_joint = per_feat[:n_pose].reshape(-1, 3).mean(axis=1)
        out["per_joint_mse"] = per_joint.astype(float).tolist()
    if ws.has_speed:
        out["speed_channel_mse"] = float(per_feat[-1])
    return out


def _latent_separability(model: BaseClassifier, ws: WindowSet) -> dict | None:
    """Silhouette / Davies-Bouldin of the held-out latent ``z`` by emotion label.

    Measures how well the multi-task latent separates the four emotions on
    *unseen* data (the held-out fold), which is the property the PCA story in
    Phase 7 leans on. Higher silhouette / lower Davies-Bouldin == cleaner
    separation. ``None`` for non-NN models, or when there are fewer than two
    labels or two samples (both metrics are undefined then).
    """
    if not _is_multitask(model) or ws.X.shape[0] == 0:
        return None
    z = np.asarray(model.transform(ws))
    y = ws.y.astype(str)
    labels = np.unique(y)
    if len(labels) < 2 or len(z) <= len(labels):
        return None
    return {
        "silhouette": float(silhouette_score(z, y)),
        "davies_bouldin": float(davies_bouldin_score(z, y)),
        "n_windows": int(len(z)),
        "n_classes": int(len(labels)),
    }


def _mean_std(vals: list[float]) -> dict | None:
    """``{mean, std}`` of a list of fold scalars, or ``None`` if empty."""
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def _aggregate(fold_scores: list[dict], key: str) -> dict:
    """Mean +/- std of a scalar metric across folds (LOSO subject-dependence)."""
    vals = [f[key]["macro_f1"] for f in fold_scores]
    accs = [f[key]["accuracy"] for f in fold_scores]
    bals = [f[key]["balanced_accuracy"] for f in fold_scores]
    return {
        "macro_f1_mean": float(np.mean(vals)),
        "macro_f1_std": float(np.std(vals)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "balanced_accuracy_mean": float(np.mean(bals)),
        "balanced_accuracy_std": float(np.std(bals)),
    }


def _aggregate_reconstruction(fold_scores: list[dict]) -> dict | None:
    """Mean +/- std of the held-out reconstruction MSE across folds (NN only)."""
    vals = [f["reconstruction"]["overall_mse"] for f in fold_scores if f.get("reconstruction")]
    agg = _mean_std(vals)
    return {"overall_mse_mean": agg["mean"], "overall_mse_std": agg["std"]} if agg else None


def _aggregate_separability(fold_scores: list[dict]) -> dict | None:
    """Mean +/- std of held-out latent Silhouette / Davies-Bouldin (NN only)."""
    sil = [f["separability"]["silhouette"] for f in fold_scores if f.get("separability")]
    db = [f["separability"]["davies_bouldin"] for f in fold_scores if f.get("separability")]
    s_agg, d_agg = _mean_std(sil), _mean_std(db)
    if not s_agg and not d_agg:
        return None
    return {"silhouette": s_agg, "davies_bouldin": d_agg}


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
        val_info = _fit_fold(model, ds, train_idx, train_ws, mean, std, cfg, length, stride)
        win_pred = np.asarray(model.predict(_model_X(model, test_ws)), dtype=object)

        clip_ids, clip_pred = _majority_vote(win_pred, test_ws.clip_idx)
        clip_true = np.asarray([str(ds.labels[c]) for c in clip_ids], dtype=object)

        fold_score = {
            "fold": fold,
            "test_subjects": sorted(set(str(ds.subjects[c]) for c in test_idx)),
            "window": _scores(test_ws.y.astype(str), win_pred.astype(str), labels_sorted),
            "clip": _scores(clip_true.astype(str), clip_pred.astype(str), labels_sorted),
        }
        # Multi-task NN extras on the held-out fold (None for classic ML): how
        # well it rebuilds unseen motion, and how cleanly the latent separates
        # the emotions on an unseen subject.
        recon = _reconstruction_mse(model, test_ws)
        if recon is not None:
            fold_score["reconstruction"] = recon
        sep = _latent_separability(model, test_ws)
        if sep is not None:
            fold_score["separability"] = sep
        if val_info is not None:
            fold_score["validation"] = val_info
        fold_scores.append(fold_score)
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
    final_model.fit(_model_X(final_model, full_ws), full_ws.y)

    metrics = {
        "labels": labels_sorted,
        "split_strategy": cfg.split.strategy,
        "folds": fold_scores,
        "window_level": _aggregate(fold_scores, "window"),
        "clip_level": _aggregate(fold_scores, "clip"),
    }
    recon_agg = _aggregate_reconstruction(fold_scores)
    if recon_agg is not None:
        metrics["reconstruction"] = recon_agg
    sep_agg = _aggregate_separability(fold_scores)
    if sep_agg is not None:
        metrics["separability"] = sep_agg
    if cfg.validation.enabled:
        used = [f["validation"] for f in fold_scores if f.get("validation", {}).get("used")]
        epochs_run = [v["epochs_run"] for v in used if v.get("epochs_run") is not None]
        metrics["early_stopping"] = {
            "enabled": True,
            "strategy": cfg.validation.strategy,
            "monitor": cfg.validation.monitor,
            "patience": cfg.validation.patience,
            "folds_used": len(used),
            "epochs_run": _mean_std(epochs_run),
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
            "early_stopping": cfg.validation.enabled,
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
