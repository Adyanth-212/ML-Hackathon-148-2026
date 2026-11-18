"""Server-side scoring of a prediction against the evaluator reference route (TODO, P7).

This is the ONLY module that reads evaluator ``.route`` files, and it is called
only after ``predict_trip`` has returned. Reference data never goes to the UI;
only the computed metrics do.
"""
from __future__ import annotations


def score(trip_id: str, prediction: dict) -> dict:
    raise NotImplementedError
