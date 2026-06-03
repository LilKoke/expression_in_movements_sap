"""Evaluation + the A-vs-B comparison.

Computes per-class F1, accuracy and confusion matrix for a run, and compares two
run directories (e.g. a classic-ML run vs an NN run) that share the same
subject-grouped split. Writes a comparison report into ``outputs/``.

Implementation lands in the evaluation phase (see issues from #1).
"""

from __future__ import annotations

from pathlib import Path


def evaluate_run(run_dir: str | Path) -> dict:
    """Load a run's predictions and compute metrics."""
    raise NotImplementedError("evaluation phase")


def compare_runs(run_a: str | Path, run_b: str | Path, out_path: str | Path) -> Path:
    """Compare two runs (approach A vs B) and write a report."""
    raise NotImplementedError("evaluation phase")
