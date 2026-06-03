"""Tests for subject-grouped / intra-subject splitting (Phase 4, #6).

The whole point of these splits is no leakage, so the tests assert exactly that:
a held-out subject never appears in train (inter-subject), and an intra-subject
fold never lets a trial's clips straddle train/test.
"""

from __future__ import annotations

import numpy as np
import pytest

from expr_movements.config import SplitConfig
from expr_movements.splits import iter_splits

SUBJECTS = np.array(["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5)
LABELS = np.array((["sad", "angry", "neutral", "happy", "sad"]) * 4, dtype=object)


def test_loso_one_subject_held_out_per_fold():
    cfg = SplitConfig(strategy="leave_one_subject_out")
    folds = list(iter_splits(SUBJECTS, LABELS, cfg))
    assert len(folds) == 4  # 4 subjects -> 4 folds
    for train_idx, test_idx in folds:
        test_subjects = set(SUBJECTS[test_idx])
        train_subjects = set(SUBJECTS[train_idx])
        assert len(test_subjects) == 1  # exactly one subject is the test fold
        assert test_subjects.isdisjoint(train_subjects)  # no leakage
        # every index appears exactly once across train+test
        assert sorted(np.concatenate([train_idx, test_idx])) == list(range(len(SUBJECTS)))


def test_group_kfold_no_subject_leakage():
    cfg = SplitConfig(strategy="group_kfold", n_splits=4)
    for train_idx, test_idx in iter_splits(SUBJECTS, LABELS, cfg):
        assert set(SUBJECTS[test_idx]).isdisjoint(set(SUBJECTS[train_idx]))


def test_group_kfold_caps_splits_to_group_count():
    # Asking for more folds than subjects must not crash; it caps at #subjects.
    cfg = SplitConfig(strategy="group_kfold", n_splits=10)
    folds = list(iter_splits(SUBJECTS, LABELS, cfg))
    assert len(folds) == 4


def test_group_shuffle_held_out_group():
    cfg = SplitConfig(strategy="group_shuffle", n_splits=3, test_size=0.25)
    folds = list(iter_splits(SUBJECTS, LABELS, cfg))
    assert len(folds) == 3
    for train_idx, test_idx in folds:
        assert set(SUBJECTS[test_idx]).isdisjoint(set(SUBJECTS[train_idx]))


def test_intra_subject_no_trial_straddle():
    # Each subject has its own trials; an intra-subject fold must keep all clips
    # of a trial on one side, and the test set must mix all subjects.
    trials = np.arange(len(SUBJECTS))  # one clip per trial here
    cfg = SplitConfig(strategy="intra_subject", n_splits=5)
    folds = list(iter_splits(SUBJECTS, LABELS, cfg, trials=trials))
    assert folds  # at least one fold
    for train_idx, test_idx in folds:
        # within each fold a subject can appear on both sides (that's intra),
        # but a given trial id never does.
        assert set(trials[test_idx]).isdisjoint(set(trials[train_idx]))
    # test sets across folds cover every clip exactly once
    all_test = np.concatenate([te for _, te in folds])
    assert sorted(all_test) == list(range(len(SUBJECTS)))


def test_unknown_strategy_raises():
    cfg = SplitConfig(strategy="nonsense")
    with pytest.raises(ValueError, match="unknown split strategy"):
        list(iter_splits(SUBJECTS, LABELS, cfg))
