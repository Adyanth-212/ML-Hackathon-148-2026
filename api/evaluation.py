"""Server-side evaluator scoring.

This is the only API module allowed to open evaluator ``.route`` files. It is
called after ``pathsnap.pipeline.predict_trip`` has completed. Only derived
metrics are returned; reference arc IDs never leave the server.
"""
from __future__ import annotations

import numpy as np

from pathsnap.config import load_config, repo_path
from pathsnap.io import discover_trips, read_arcs, read_route


def _edit_similarity(reference: list[int], predicted: list[int]) -> float:
    """Normalised Levenshtein similarity for two arc-id sequences."""
    if not reference and not predicted:
        return 1.0
    previous = list(range(len(predicted) + 1))
    for i, left in enumerate(reference, start=1):
        current = [i]
        for j, right in enumerate(predicted, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    distance = previous[-1]
    return float(1.0 - distance / max(len(reference), len(predicted), 1))


def _route_continuity(predicted: list[int], arcs) -> float | None:
    if len(predicted) < 2:
        return None
    pairs = arcs[["from_node", "to_node"]].to_numpy(dtype=int)
    valid = [arc_id for arc_id in predicted if 0 <= arc_id < len(pairs)]
    if len(valid) < 2:
        return None
    connected = sum(pairs[a, 1] == pairs[b, 0] for a, b in zip(valid, valid[1:]))
    return float(connected / (len(valid) - 1))


def score(trip_id: str, prediction: dict) -> dict:
    """Score a completed prediction using the evaluator reference route.

    The prediction contract deliberately contains no reference route. Route
    IDs are read only here, after prediction, and are never copied to output.
    """
    evaluator_root = repo_path(load_config("data")["partitions"]["evaluator"])
    found = discover_trips("evaluator", root=evaluator_root)
    if trip_id not in found:
        raise FileNotFoundError(f"Evaluator reference route not found for trip {trip_id}")
    files = found[trip_id]
    if files.route is None or files.arcs is None:
        raise FileNotFoundError(f"Evaluator route/network files are incomplete for trip {trip_id}")

    reference = read_route(files.route).astype(int).tolist()
    predicted = [int(value) for value in prediction.get("route_arc_ids", [])]
    ref_counts, pred_counts = {}, {}
    for value in reference:
        ref_counts[value] = ref_counts.get(value, 0) + 1
    for value in predicted:
        pred_counts[value] = pred_counts.get(value, 0) + 1
    overlap = sum(min(count, pred_counts.get(value, 0)) for value, count in ref_counts.items())
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    arcs = read_arcs(files.arcs)

    return {
        "route_precision": round(float(precision), 6),
        "route_recall": round(float(recall), 6),
        "route_f1": round(float(f1), 6),
        "route_exact_match": bool(reference == predicted),
        "route_edit_similarity": round(_edit_similarity(reference, predicted), 6),
        "route_continuity": _route_continuity(predicted, arcs),
        # Candidate-level metrics are not applicable to the nearest-arc stub.
        "candidate_miss_rate": None,
        "classifier_top1_accuracy": None,
    }
