"""Pydantic config schemas.

YAML files under ``configs/`` are validated into these models at startup. The
*resolved* config is dumped back out next to every trained model artifact so a
run is always reproducible from its own output directory (see ``run.py``).

The schema is intentionally small; extend it as phases land (see the phase
issues linked from #1). Keep it validation-only — no I/O, no model building.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    """Base: reject unknown keys so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid")


# ---- emotion label vocabulary -------------------------------------------------
# Filename code (3 letters) -> class name. Source of truth for the label space.
EMOTION_CODES: dict[str, str] = {
    "TRE": "sad",
    "COE": "angry",
    "NEE": "neutral",
    "JOE": "happy",
}


class DataConfig(_Strict):
    """How raw TRC is turned into modeling-ready datasets."""

    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")

    # Trim the start/end of each motion (frames) to drop standing/idle padding.
    trim_start_frames: int = Field(0, ge=0)
    trim_end_frames: int = Field(0, ge=0)


class SplitConfig(_Strict):
    """Train/test split. MUST be subject-grouped to avoid leakage (see docs)."""

    strategy: str = "group_kfold"  # group_kfold | group_shuffle | leave_one_subject_out
    n_splits: int = Field(5, ge=2)
    test_size: float = Field(0.2, gt=0, lt=1)
    seed: int = 42


class ModelConfig(_Strict):
    """Selects a model from the registry and supplies its hyperparameters.

    ``name`` is a registry key (e.g. "random_forest", "lstm"); ``params`` is
    passed verbatim to that model's constructor.
    """

    name: str
    params: dict = Field(default_factory=dict)


class ExperimentConfig(_Strict):
    """Top-level config composing data + split + model for one run."""

    name: str
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML into an :class:`ExperimentConfig`."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)
