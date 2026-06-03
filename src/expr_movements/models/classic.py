"""Approach A: classic ML on flattened pose-window features (Phase 4, #6).

Thin wrappers over scikit-learn estimators so they live in the same registry as
the neural model and are driven by the same training harness. ``X`` here is a
2-D feature table — one row per *window*, the window's ``(length, F)`` tensor
flattened to ``length*F`` (see ``data/windows.flatten_windows``). The dedicated
expert-feature table (``expr-featurize``) is a later phase; this baseline runs
classic ML directly on the standardised window features so the A-vs-B harness is
exercised end to end.

Each wrapper builds its sklearn estimator lazily in ``fit`` (so ``__init__``
stays logic-free for ``clone``/``set_params``) and stores it as ``model_``;
``predict`` validates the model was fitted first.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from expr_movements.models.base import BaseClassifier
from expr_movements.models.registry import register


@register("random_forest")
class RandomForestModel(BaseClassifier):
    def __init__(
        self, n_estimators: int = 300, max_depth: int | None = None, random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y) -> "RandomForestModel":
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)


@register("svm")
class SVMModel(BaseClassifier):
    def __init__(
        self, C: float = 1.0, kernel: str = "rbf", gamma: str = "scale", random_state: int = 42
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y) -> "SVMModel":
        self.model_ = SVC(
            C=self.C, kernel=self.kernel, gamma=self.gamma, random_state=self.random_state
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)
