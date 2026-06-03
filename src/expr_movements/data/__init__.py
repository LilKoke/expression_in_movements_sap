"""Data layer: raw TRC -> interim sequences -> processed datasets.

``raw/`` is immutable original ``.trc``; ``interim/`` holds parsed/cleaned
per-clip arrays; ``processed/`` holds modeling-ready artifacts (a feature table
for approach A, a sequence tensor for approach B). Processed data is persisted
and reloaded, never recomputed every run.
"""
