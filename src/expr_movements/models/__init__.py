"""Model layer: swappable classifiers behind one sklearn-style interface.

Both approaches (classic ML on expert features, NN on raw sequences) implement
the same ``fit(X, y) -> self`` / ``predict(X)`` contract, so the *same* training,
splitting and evaluation code drives either one. Pick a model by config name via
:func:`build_model`.

Importing this package registers the built-in models as a side effect.
"""

from expr_movements.models.registry import build_model, register, registered_names

# Side-effect imports populate the registry. Add new model modules here.
from expr_movements.models import classic, neural  # noqa: E402,F401

__all__ = ["build_model", "register", "registered_names"]
