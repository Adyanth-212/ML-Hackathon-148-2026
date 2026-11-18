"""Model registry (TODO, phase P4).

Each entry returns an sklearn-compatible estimator (wrap scalers in a
``Pipeline`` so CV folds refit them). Log the config, seed, fit time and
inference time of every model under experiments/.
"""
from __future__ import annotations


def get_model(name: str, params: dict | None = None, seed: int = 148):
    raise NotImplementedError
