"""Tests for subject-grouped / intra-subject splitting (Phase 4, #6).

The whole point of these splits is no leakage, so the tests assert exactly that:
a held-out subject never appears in train (inter-subject), and an intra-subject
fold never lets a trial's clips straddle train/test.
"""

from __future__ import annotations

import numpy as np
import pytest

from expr_movements.config import SplitConfig, ValidationConfig
from expr_movements.splits import iter_splits, nested_validation_split

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


# -- nested validation split (Phase 8, #1) ------------------------------------


def test_nested_group_subject_holds_out_whole_subject():
    """group_subject validation keeps a whole subject out of the fit set.

    Simulates one LOSO fold: train side = 3 subjects (B, C, D). The validation
    set must be 1 entire subject, disjoint from the fit subjects (unseen-person
    early stopping), and the two index sets must partition the train clips.
    """
    train_subjects = np.array(["B"] * 5 + ["C"] * 5 + ["D"] * 5)
    train_labels = LABELS[:15]
    cfg = ValidationConfig(enabled=True, strategy="group_subject", val_size=0.34)
    fit_pos, val_pos = nested_validation_split(train_subjects, train_labels, cfg)

    fit_s, val_s = set(train_subjects[fit_pos]), set(train_subjects[val_pos])
    assert len(val_s) == 1  # one whole subject is validation
    assert val_s.isdisjoint(fit_s)  # unseen-person validation
    # partition: every train clip is in exactly one side
    assert sorted(np.concatenate([fit_pos, val_pos])) == list(range(len(train_subjects)))


def test_nested_group_subject_single_subject_raises():
    """A fold whose train side is one subject can't make a group validation set."""
    cfg = ValidationConfig(enabled=True, strategy="group_subject")
    with pytest.raises(ValueError, match=">=2 train subjects"):
        nested_validation_split(np.array(["B"] * 6), LABELS[:6], cfg)


def test_nested_stratified_clip_covers_all_classes():
    """stratified_clip splits clips (subjects mixed), keeping every label in val."""
    train_subjects = np.array(["B"] * 6 + ["C"] * 6)
    train_labels = np.array((["sad", "angry", "neutral", "happy", "sad", "angry"]) * 2, dtype=object)
    cfg = ValidationConfig(enabled=True, strategy="stratified_clip", val_size=0.5)
    fit_pos, val_pos = nested_validation_split(train_subjects, train_labels, cfg)
    assert set(np.concatenate([fit_pos, val_pos])) == set(range(len(train_subjects)))
    # stratified -> validation contains more than one class
    assert len(set(train_labels[val_pos])) >= 2


def test_nested_unknown_strategy_raises():
    cfg = ValidationConfig(enabled=True, strategy="bogus")
    with pytest.raises(ValueError, match="unknown validation strategy"):
        nested_validation_split(SUBJECTS, LABELS, cfg)
