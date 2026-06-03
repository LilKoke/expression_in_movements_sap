"""Common model interface.

Every model is a scikit-learn-compatible estimator: hyperparameters are stored
verbatim on ``__init__`` (no logic, so ``clone()``/``set_params`` work), ``fit``
returns ``self`` and sets fitted attributes with a trailing underscore, and
``predict`` validates that the model is fitted first.

Approach A (classic ML) is sklearn-native already. Approach B (neural net) wraps
a torch model in this same interface so both are interchangeable.
"""

from __future__ import annotations

from abc import abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin


class BaseClassifier(ClassifierMixin, BaseEstimator):
    """Abstract base for all classifiers in this project.

    Note: ``X`` differs by approach — a 2-D feature table for classic ML, a 3-D
    ``(n, n_frames, n_markers*3)`` sequence tensor for the NN. Each model declares
    which processed artifact it consumes; the objects stay interface-identical.
    """

    @abstractmethod
    def fit(self, X, y) -> "BaseClassifier":  # returns self
        ...

    @abstractmethod
    def predict(self, X):
        ...
