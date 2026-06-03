"""Emotion classification from motion-capture (TRC) walking sequences.

Compares two modeling approaches on the same subject-grouped split:
  A) expert/hand-crafted features + classic ML (RandomForest / SVM)
  B) a neural network (LSTM / 1D-CNN) on raw pose sequences.

See docs/ARCHITECTURE.md for the overall design.
"""

__version__ = "0.1.0"
