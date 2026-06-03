"""Tests for the implemented architecture pieces: config loading, the model
registry, and the run-directory config<->artifact binding.

Phase implementations (parsing, features, training) are tested in their own PRs.
"""

from __future__ import annotations

import yaml

from expr_movements.config import EMOTION_CODES, load_experiment
from expr_movements.data.trc import parse_filename
from expr_movements.models import build_model, registered_names
from expr_movements.run import config_hash, create_run_dir


def test_load_experiment_configs():
    rf = load_experiment("configs/experiment_rf.yaml")
    assert rf.model.name == "random_forest"
    assert rf.split.strategy == "leave_one_subject_out"  # LOSO is the default protocol
    assert rf.window.length == 64  # sliding-window prediction unit

    lstm = load_experiment("configs/experiment_lstm.yaml")
    assert lstm.model.name == "lstm"
    assert lstm.model.params["hidden_size"] == 128
    # A and B must share the same windowing + folds for a fair comparison.
    assert lstm.split.strategy == rf.split.strategy
    assert lstm.window == rf.window


def test_registry_has_both_approaches():
    names = registered_names()
    assert "random_forest" in names  # approach A
    assert "lstm" in names  # approach B
    assert isinstance(build_model("random_forest", n_estimators=10), object)


def test_parse_filename():
    meta = parse_filename("a_EMLACOE01.4.trc")
    assert meta.subject == "EMLA"
    assert meta.emotion_code == "COE"
    assert meta.emotion == "angry"
    assert meta.take == 1

    assert parse_filename("NABATRE05.4.trc").emotion == "sad"


def test_emotion_vocabulary():
    assert set(EMOTION_CODES.values()) == {"sad", "angry", "neutral", "happy"}


def test_run_dir_binds_config_to_artifact(tmp_path):
    cfg = load_experiment("configs/experiment_rf.yaml")
    run_dir = create_run_dir(cfg, outputs_root=tmp_path)

    # The resolved config is written next to where the model will live.
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert saved["model"]["name"] == "random_forest"
    # Directory name embeds a stable content hash -> re-running is detectable.
    assert config_hash(cfg) in run_dir.name
    assert create_run_dir(cfg, outputs_root=tmp_path) == run_dir
