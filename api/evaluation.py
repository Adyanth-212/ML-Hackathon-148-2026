"""Server-side evaluator scoring.

This is the only API module allowed to open evaluator ``.route`` files. It is
called after ``pathsnap.pipeline.predict_trip`` has completed. Only derived
metrics are returned; reference arc IDs never leave the server.
"""
from __future__ import annotations

import numpy as np

from pathsnap.candidates import generate_candidates
from pathsnap.config import load_config, repo_path
from pathsnap.geo import TripGeometry
from pathsnap.io import discover_trips, load_trip, read_arcs, read_route
from pathsnap.labels import align_route_to_observations
from pathsnap.metrics import candidate_miss_rate


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


def _arc_sequence(predicted: list[int], n_arcs: int) -> list[int]:
    """Per-point arcs -> driven arc sequence: drop invalid ids, collapse consecutive repeats.

    ``route_arc_ids`` from the pipeline has one entry per GPS point, so the same arc repeats
    while the vehicle is on it. Route metrics compare arc sequences, as in scripts/eval_hmm.py.
    """
    out: list[int] = []
    for arc_id in predicted:
        if 0 <= arc_id < n_arcs and (not out or out[-1] != arc_id):
            out.append(arc_id)
    return out


def _route_continuity(sequence: list[int], arcs) -> float | None:
    """Share of arc changes where the next arc starts at the node the previous one ends."""
    if len(sequence) < 2:
        return 1.0 if sequence else None
    pairs = arcs[["from_node", "to_node"]].to_numpy(dtype=int)
    connected = sum(pairs[a, 1] == pairs[b, 0] for a, b in zip(sequence, sequence[1:]))
    return float(connected / (len(sequence) - 1))


def _pointwise(files, prediction: dict) -> dict:
    """Observation-level Top-1 and candidate miss rate against per-point labels.

    The labels come from the same monotonic DP alignment of the reference route used for
    train/student_test (pathsnap.labels), computed here on the server after prediction.
    Points the alignment flags as unreliable (> 50 m from their aligned arc) are excluded,
    exactly as in training and evaluation. Candidates are regenerated from the track and
    network alone (the same call the pipeline makes).
    """
    trip = load_trip(files, with_route=True)
    geom = TripGeometry.from_trip(trip)
    aligned = align_route_to_observations(geom, trip.route)
    labels = aligned["arc_id"].to_numpy()
    ok = ~aligned["is_outlier"].to_numpy()
    pred = np.asarray(prediction.get("pred_arcs", []), dtype=int)
    if len(pred) != geom.n_obs or not ok.any():
        return {"top1_accuracy": None, "candidate_miss_rate": None, "n_points_scored": int(ok.sum())}
    cands, _ = generate_candidates(geom, trip.trip_id)
    return {"top1_accuracy": round(float(np.mean(pred[ok] == labels[ok])), 6),
            "candidate_miss_rate": round(candidate_miss_rate(cands, np.where(ok, labels, -1)), 6),
            "n_points_scored": int(ok.sum())}


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
    arcs = read_arcs(files.arcs)
    predicted = _arc_sequence([int(v) for v in prediction.get("route_arc_ids", [])], len(arcs))
    ref_counts, pred_counts = {}, {}
    for value in reference:
        ref_counts[value] = ref_counts.get(value, 0) + 1
    for value in predicted:
        pred_counts[value] = pred_counts.get(value, 0) + 1
    overlap = sum(min(count, pred_counts.get(value, 0)) for value, count in ref_counts.items())
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pointwise = _pointwise(files, prediction)

    return {
        **pointwise,
        "route_precision": round(float(precision), 6),
        "route_recall": round(float(recall), 6),
        "route_f1": round(float(f1), 6),
        "route_exact_match": bool(reference == predicted),
        "route_edit_similarity": round(_edit_similarity(reference, predicted), 6),
        "route_continuity": _route_continuity(predicted, arcs),
        # kept for older UI builds: the same value as top1_accuracy (pipeline output, HMM included)
        "classifier_top1_accuracy": pointwise["top1_accuracy"],
    }
