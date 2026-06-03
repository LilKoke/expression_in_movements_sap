"""Training orchestration: config -> dataset -> model -> run dir.

Flow (implemented in the modeling phase, see issues from #1):
  1. load_experiment(config_path) -> validated ExperimentConfig
  2. load the processed dataset the chosen model consumes
     (feature table for classic ML, sequence tensor for the NN)
  3. build_model(cfg.model.name, **cfg.model.params)
  4. fit over subject-grouped splits (splits.iter_splits)
  5. create_run_dir(cfg) and save model + resolved config + metadata + metrics

Keeping this as the single training path means approach A and B share the exact
same harness and split — only the config differs.
"""

from __future__ import annotations

from pathlib import Path

from expr_movements.config import ExperimentConfig


def run_training(cfg: ExperimentConfig, outputs_root: str | Path = "outputs") -> Path:
    """Train the model described by ``cfg`` and return its run directory."""
    raise NotImplementedError("modeling phase")
