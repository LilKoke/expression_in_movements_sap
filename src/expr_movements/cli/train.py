"""``expr-train``: train one model from an experiment config.

    expr-train --config configs/experiment_rf.yaml

Delegates to :func:`expr_movements.train.run_training`; swapping approach A <-> B
is just pointing ``--config`` at a different YAML.
"""

from __future__ import annotations

import argparse

from expr_movements.config import load_experiment
from expr_movements.train import run_training


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a model from an experiment config.")
    parser.add_argument("--config", required=True, help="path to an experiment YAML")
    parser.add_argument("--outputs", default="outputs", help="root dir for run artifacts")
    args = parser.parse_args(argv)

    cfg = load_experiment(args.config)
    run_dir = run_training(cfg, outputs_root=args.outputs)
    print(f"run written to {run_dir}")


if __name__ == "__main__":
    main()
