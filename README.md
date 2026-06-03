# expression_in_movements_sap

Emotion classification from motion-capture (TRC) walking sequences, comparing
two modeling approaches:

- **A** — expert / hand-crafted features + classic ML (RandomForest / SVM)
- **B** — a neural network (LSTM / 1D-CNN) on raw pose sequences

Emotion labels come from the TRC filename code: `TRE=sad`, `COE=angry`,
`NEE=neutral`, `JOE=happy`.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design, and issue
#1 for the roadmap / phase breakdown.

## Setup

Requires Python 3.13 and [`uv`](https://docs.astral.sh/uv/).

```sh
uv sync          # create .venv and install deps (incl. dev group)
uv run pytest    # run tests
```

Place the original `.trc` files under `data/raw/` (the `data/` directory is
gitignored).

**Working without the raw data (e.g. the expert-features team, no GPU):** the
processed dataset is distributed via Google Drive — clone, `uv sync`, then unzip
`expr-data-processed.zip` at the repo root. Step-by-step in
[docs/DATA_SETUP.md](docs/DATA_SETUP.md). GPU is not required; the NN weights
under `outputs/` are not needed.

## Pipeline (CLIs)

Each step is a thin entry point; all logic lives in `src/expr_movements/`.

```sh
uv run expr-parse          --raw-dir data/raw        --out-dir data/interim
uv run expr-build-dataset  --interim-dir data/interim --out-dir data/processed   # + --target-frames N for a dense tensor
uv run expr-featurize      --manifest data/processed/manifest.jsonl --out data/processed/features.parquet
uv run expr-train          --config configs/experiment_rf.yaml      # approach A
uv run expr-train          --config configs/experiment_lstm.yaml    # approach B
uv run expr-evaluate       --compare outputs/rf_XXXX outputs/lstm_XXXX
```

Each training run writes an immutable directory under `outputs/` containing the
resolved config, the model checkpoint, metrics and provenance metadata — so
every model is reproducible from its own folder.

## Status

The architecture, environment, config-driven run scaffold and tests are in
place. The data/feature/model/evaluation logic is implemented per the phase
issues linked from #1 — modules not yet filled in raise `NotImplementedError`.
