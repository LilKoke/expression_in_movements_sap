"""Subject-grouped train/test splitting (Phase 4, #6).

Random or frame-level splits leak a subject's motion into both train and test
and inflate accuracy. Every split here is keyed on the *clip* (one motion file),
grouped so a whole subject (inter-subject) or a whole trial (intra-subject)
falls entirely in one side of the fold. Approach A and B MUST use the identical
split for the comparison to be valid, so this is the single source of folds for
both — call it with the same ``groups`` / ``cfg`` from either training path.

The indices yielded are positions into the *clip* arrays (``groups``, ``y``),
not into the expanded window rows. The dataset loader expands a clip's windows
*after* the fold is chosen, so windows inherit their clip's side of the split
and never straddle it (no window-level leakage).

Two protocol families, switched by ``cfg.strategy``:

* **inter-subject** (``leave_one_subject_out`` / ``group_kfold`` /
  ``group_shuffle``): the held-out subject's clips never appear in train. This
  is the real task — generalising to an *unseen person*.
* **intra-subject** (``intra_subject``): split *within* each subject at the
  trial level, then pool the per-subject train/test clip indices into one fold
  per CV round. For direct comparison with the source paper's intra-subject
  >90% (Venture et al. 2014).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, KFold

from expr_movements.config import SplitConfig


def iter_splits(
    groups: Sequence[str],
    y: Sequence,
    cfg: SplitConfig,
    trials: Sequence | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` clip-index pairs per ``cfg.strategy``.

    ``groups`` is the per-clip subject id; ``y`` the per-clip label (used only
    for stratification-free CV counts). ``trials`` is the per-clip trial id
    (defaults to a unique id per clip) and is only consulted for the
    ``intra_subject`` strategy, where the within-subject split is made on trials
    so windows of one trial never straddle train/test.

    Inter-subject strategies are backed by sklearn ``LeaveOneGroupOut`` /
    ``GroupKFold`` / ``GroupShuffleSplit`` on ``groups``. ``intra_subject`` runs
    a ``KFold`` over each subject's trials and concatenates the per-subject
    train/test indices into a single fold per CV round.
    """
    groups = np.asarray(groups)
    y = np.asarray(y)
    n = len(groups)
    if n == 0:
        raise ValueError("no clips to split")
    x = np.zeros((n, 1))  # sklearn splitters want an X of the right length

    strategy = cfg.strategy
    if strategy == "leave_one_subject_out":
        yield from ((tr, te) for tr, te in LeaveOneGroupOut().split(x, y, groups))
    elif strategy == "group_kfold":
        n_groups = len(np.unique(groups))
        n_splits = min(cfg.n_splits, n_groups)
        yield from GroupKFold(n_splits=n_splits).split(x, y, groups)
    elif strategy == "group_shuffle":
        gss = GroupShuffleSplit(
            n_splits=cfg.n_splits, test_size=cfg.test_size, random_state=cfg.seed
        )
        yield from gss.split(x, y, groups)
    elif strategy == "intra_subject":
        yield from _intra_subject_splits(groups, trials, cfg)
    else:
        raise ValueError(
            f"unknown split strategy {strategy!r}; expected one of "
            "leave_one_subject_out | group_kfold | group_shuffle | intra_subject"
        )


def _intra_subject_splits(
    groups: np.ndarray, trials: Sequence | None, cfg: SplitConfig
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Per-subject trial-level KFold, pooled into one fold per CV round.

    For each subject we ``KFold`` its unique trials into ``n_splits`` folds; CV
    round ``k`` takes fold ``k`` of every subject as test and the rest as train.
    Splitting on trials (not clips, not windows) guarantees every window of a
    trial lands on the same side. A subject with fewer trials than ``n_splits``
    is capped to its trial count for that subject's own folds, and rounds beyond
    a subject's fold count simply contribute no test clips from that subject.
    """
    n = len(groups)
    idx = np.arange(n)
    trials = np.arange(n) if trials is None else np.asarray(trials)

    # Per subject: an ordered list of (train_clip_idx, test_clip_idx) over its trials.
    per_subject_folds: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for subj in np.unique(groups):
        smask = groups == subj
        s_idx = idx[smask]
        s_trials = trials[smask]
        uniq = np.unique(s_trials)
        k = min(cfg.n_splits, len(uniq))
        if k < 2:
            # Single trial for this subject: it can only ever be train; no test fold.
            per_subject_folds.append([])
            continue
        kf = KFold(n_splits=k, shuffle=True, random_state=cfg.seed)
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for tr_t, te_t in kf.split(uniq):
            test_trials = set(uniq[te_t].tolist())
            te = s_idx[np.isin(s_trials, list(test_trials))]
            tr = s_idx[~np.isin(s_trials, list(test_trials))]
            folds.append((tr, te))
        per_subject_folds.append(folds)

    rounds = max((len(f) for f in per_subject_folds), default=0)
    for k in range(rounds):
        tr_parts, te_parts = [], []
        for folds in per_subject_folds:
            if k < len(folds):
                tr_parts.append(folds[k][0])
                te_parts.append(folds[k][1])
        if not te_parts:
            continue
        train_idx = np.concatenate(tr_parts) if tr_parts else np.array([], dtype=int)
        test_idx = np.concatenate(te_parts)
        yield train_idx, test_idx
