"""Approach A: classic ML on expert/hand-crafted features.

Thin wrappers over scikit-learn estimators so they live in the same registry as
the neural model. ``X`` here is a 2-D feature table (one row per motion clip).

Bodies are stubs — implementation lands in the modeling phase (see issues
linked from #1).
"""

from __future__ import annotations

from expr_movements.models.base import BaseClassifier
from expr_movements.models.registry import register


@register("random_forest")
class RandomForestModel(BaseClassifier):
    def __init__(self, n_estimators: int = 300, max_depth: int | None = None, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y) -> "RandomForestModel":
        raise NotImplementedError("modeling phase")

    def predict(self, X):
        raise NotImplementedError("modeling phase")


@register("svm")
class SVMModel(BaseClassifier):
    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma: str = "scale", random_state: int = 42):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y) -> "SVMModel":
        raise NotImplementedError("modeling phase")

    def predict(self, X):
        raise NotImplementedError("modeling phase")
