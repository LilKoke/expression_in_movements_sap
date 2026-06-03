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

    # Drop the standing/idle frames before walking starts and after it ends.
    # When ``detect_onset`` is on, the active window is found from marker speed
    # (see ``data/dataset.py``); ``trim_start_frames`` / ``trim_end_frames`` are
    # then applied as an *additional* fixed margin inside the detected window.
    # With detection off they are the only trimming and act on the raw clip.
    detect_onset: bool = True
    # Speed threshold as a fraction of each clip's peak marker speed: a frame is
    # "moving" once the body's mean marker speed exceeds this fraction of the
    # clip's max.
    onset_speed_frac: float = Field(0.1, gt=0, lt=1)
    # Require this many consecutive moving frames before declaring onset, to
    # ignore brief jitter at the start (and symmetrically for offset).
    onset_min_run: int = Field(3, ge=1)
    trim_start_frames: int = Field(0, ge=0)
    trim_end_frames: int = Field(0, ge=0)

    # Invariant pre-processing (Phase 3, #5). Make the pose sequence invariant to
    # where in the room the subject walked and which way they faced, while
    # keeping the walking *speed* as an explicit feature (it carries emotion —
    # arousal). See ``features/preprocess.py``.
    #   - normalize: master switch. Off -> raw world coordinates are stored.
    #   - pelvis_markers: bare marker tokens (suffix after the last "_") whose
    #     centroid defines the per-frame body origin. All four exist in every
    #     bundled clip; names carry a per-subject prefix (e.g. ``NABA_LFWT``).
    #   - yaw_align: rotate each frame about the vertical axis so the pelvis
    #     left->right axis points the same way (heading removed; pitch/roll kept).
    #   - keep_speed: append the per-frame body speed (pre-normalization,
    #     world-scale) as one extra channel so arousal cues survive.
    normalize: bool = True
    pelvis_markers: tuple[str, ...] = ("LFWT", "RFWT", "LBWT", "RBWT")
    yaw_align: bool = True
    keep_speed: bool = True
    # Index (0/1/2) of the vertical (gravity) axis in the marker coordinates.
    # yaw rotation is applied in the plane of the other two axes. TRC here is
    # Y-up (axis 1).
    up_axis: int = Field(1, ge=0, le=2)


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
