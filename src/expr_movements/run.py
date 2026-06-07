"""Per-run output directory: binds a config to the artifact it produced.

Every training run creates one immutable directory under ``outputs/`` named
``<config-name>_<short-config-hash>`` and writes the *resolved* config into it.
The model checkpoint, metrics and metadata land in the same directory, so a run
is always reproducible from its own folder. This is the concrete form of the
owner's requirement: "params + corresponding model are output together".

  outputs/rf_a1b2c3d4/
    ├── config.yaml      # resolved ExperimentConfig that produced this run
    ├── metadata.json    # git commit, seed, data hashes, timestamp, metrics summary
    ├── model.joblib     # (A) or model.pt (B)
    ├── metrics.json
    └── predictions.jsonl

``Date.now``-style timestamps are injected by the caller (CLI), so this module
stays deterministic and testable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from expr_movements.config import ExperimentConfig


def config_hash(cfg: ExperimentConfig, length: int = 8) -> str:
    """Stable short hash of the resolved config (content-addressable runs)."""
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


def create_run_dir(cfg: ExperimentConfig, outputs_root: str | Path = "outputs") -> Path:
    """Create and return the immutable run directory, writing the resolved config.

    The directory name embeds the config hash so re-running the same config is
    detectable (the directory already exists).
    """
    run_dir = Path(outputs_root) / f"{cfg.name}_{config_hash(cfg)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False)
    return run_dir


def write_metadata(run_dir: str | Path, metadata: dict) -> Path:
    """Write run provenance (commit, seed, data hashes, metrics, timestamp)."""
    path = Path(run_dir) / "metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return path
