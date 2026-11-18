"""End-to-end inference for one trip (TODO, phase P3).

``predict_trip`` deliberately takes NO reference route: ground truth must never
enter the prediction pipeline. Scoring happens afterwards in api/evaluation.py.
"""
from __future__ import annotations


def predict_trip(trip, artifacts) -> dict:
    """Return per-observation predicted arc_ids plus route geometry and timings."""
    raise NotImplementedError
