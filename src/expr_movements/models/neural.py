"""Approach B: neural network on raw pose sequences.

Wraps a torch model (LSTM / 1D-CNN) in the same sklearn-style interface as the
classic models, so the same training/eval code drives it. ``X`` here is a 3-D
``(n, n_frames, n_markers*3)`` sequence tensor.

Bodies are stubs — implementation lands in the modeling phase (see issues
linked from #1).
"""

from __future__ import annotations

from expr_movements.models.base import BaseClassifier
from expr_movements.models.registry import register


@register("lstm")
class LSTMModel(BaseClassifier):
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y) -> "LSTMModel":
        raise NotImplementedError("modeling phase")

    def predict(self, X):
        raise NotImplementedError("modeling phase")


@register("cnn1d")
class CNN1DModel(BaseClassifier):
    def __init__(
        self,
        channels: tuple[int, ...] = (64, 128),
        kernel_size: int = 5,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.channels = channels
        self.kernel_size = kernel_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y) -> "CNN1DModel":
        raise NotImplementedError("modeling phase")

    def predict(self, X):
        raise NotImplementedError("modeling phase")
