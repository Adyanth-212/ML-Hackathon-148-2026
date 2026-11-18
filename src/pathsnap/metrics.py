
"""Evaluation metrics for candidate classification and route reconstruction."""

from __future__ import annotations

from collections import Counter
from time import perf_counter

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def top1_accuracy(candidates, probabilities, true_arc_ids):
    """Calculate observation-level Top-1 accuracy."""

    probabilities = np.asarray(probabilities, dtype=float)
    true_arc_ids = np.asarray(true_arc_ids)
    obs_idx = candidates["obs_idx"].to_numpy()
    arc_ids = candidates["arc_id"].to_numpy()

    if len(probabilities) != len(candidates):
        raise ValueError("One probability is required per candidate.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    predictions = np.full(len(true_arc_ids), -1, dtype=np.int64)

    for obs in np.unique(obs_idx):
        if obs < 0 or obs >= len(true_arc_ids):
            raise ValueError("Invalid observation index.")

        rows = np.flatnonzero(obs_idx == obs)
        best_row = rows[np.argmax(probabilities[rows])]
        predictions[obs] = arc_ids[best_row]

    valid = (predictions >= 0) & (true_arc_ids >= 0)

    if not valid.any():
        return 0.0

    return float(np.mean(predictions[valid] == true_arc_ids[valid]))


def candidate_miss_rate(candidates, true_arc_ids):
    """Calculate the fraction of observations missing their true candidate."""

    true_arc_ids = np.asarray(true_arc_ids)
    obs_idx = candidates["obs_idx"].to_numpy()
    arc_ids = candidates["arc_id"].to_numpy()

    present = np.zeros(len(true_arc_ids), dtype=bool)

    valid = (
        (obs_idx >= 0)
        & (obs_idx < len(true_arc_ids))
    )

    np.logical_or.at(
        present,
        obs_idx[valid],
        arc_ids[valid] == true_arc_ids[obs_idx[valid]],
    )

    valid_truth = true_arc_ids >= 0

    if not valid_truth.any():
        return 0.0

    return float(np.mean(~present[valid_truth]))


def candidate_metrics(y_true, probabilities, threshold=0.5):
    """Calculate candidate-level classification metrics."""

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities must have equal length.")

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty candidate set.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must be binary.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    predictions = (probabilities >= threshold).astype(int)

    return {
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                np.column_stack([1 - probabilities, probabilities]),
                labels=[0, 1],
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }


def route_continuity(predicted_arcs, arcs):
    """Calculate the fraction of consecutive arc changes that connect."""

    predicted_arcs = np.asarray(predicted_arcs, dtype=int)
    predicted_arcs = predicted_arcs[predicted_arcs >= 0]

    changes = np.flatnonzero(
        predicted_arcs[1:] != predicted_arcs[:-1]
    )

    if len(changes) == 0:
        return 1.0

    from_nodes = arcs["from_node"].to_numpy()
    to_nodes = arcs["to_node"].to_numpy()

    previous_arcs = predicted_arcs[changes]
    next_arcs = predicted_arcs[changes + 1]

    connected = (
        to_nodes[previous_arcs] == from_nodes[next_arcs]
    )

    return float(np.mean(connected))


def sequence_metrics(true_sequences, predicted_sequences):
    """Calculate micro precision, recall and F1 across route sequences.

    Repeated arc visits are counted as separate occurrences.
    """

    if len(true_sequences) != len(predicted_sequences):
        raise ValueError("Sequence counts must match.")

    true_count = Counter()
    predicted_count = Counter()

    for true_seq, predicted_seq in zip(
        true_sequences, predicted_sequences
    ):
        true_count.update(
            int(x) for x in true_seq if int(x) >= 0
        )
        predicted_count.update(
            int(x) for x in predicted_seq if int(x) >= 0
        )

    correct = sum((true_count & predicted_count).values())

    total_true = sum(true_count.values())
    total_predicted = sum(predicted_count.values())

    precision = (
        correct / total_predicted if total_predicted else 0.0
    )

    recall = (
        correct / total_true if total_true else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def timing_metrics(timings_ms, n_observations):
    """Calculate total inference time and time per observation."""

    if n_observations < 0:
        raise ValueError("Observation count cannot be negative.")

    total = float(
        timings_ms.get("total_ms", sum(timings_ms.values()))
    )

    return {
        "timings_ms": {
            key: float(value)
            for key, value in timings_ms.items()
        },
        "total_ms": total,
        "ms_per_observation": (
            total / n_observations if n_observations else 0.0
        ),
    }


def measure_time(function, *args, **kwargs):
    """Execute a function and measure its execution time."""

    start = perf_counter()

    result = function(*args, **kwargs)

    elapsed_ms = (perf_counter() - start) * 1000

    return result, elapsed_ms
