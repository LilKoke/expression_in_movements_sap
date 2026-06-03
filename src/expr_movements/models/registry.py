"""Tiny model registry/factory.

Models register themselves with ``@register("name")``; the training CLI turns a
config ``model.name`` + ``model.params`` into an instance via ``build_model``.
This is what makes approach A <-> B a one-line config change.
"""

from __future__ import annotations

from collections.abc import Callable

from expr_movements.models.base import BaseClassifier

_REGISTRY: dict[str, type[BaseClassifier]] = {}


def register(name: str) -> Callable[[type[BaseClassifier]], type[BaseClassifier]]:
    """Class decorator registering a model under ``name``."""

    def deco(cls: type[BaseClassifier]) -> type[BaseClassifier]:
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} already registered")
        _REGISTRY[name] = cls
        return cls

    return deco


def build_model(name: str, **params) -> BaseClassifier:
    """Instantiate a registered model with hyperparameters from config."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {registered_names()}")
    return _REGISTRY[name](**params)


def registered_names() -> list[str]:
    return sorted(_REGISTRY)
