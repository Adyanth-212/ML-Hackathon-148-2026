
import numpy as np
import pandas as pd
import pytest

from pathsnap.metrics import (
    top1_accuracy,
    candidate_miss_rate,
    candidate_metrics,
    route_continuity,
    sequence_metrics,
    timing_metrics,
    measure_time,
)


def test_top1_accuracy():
    candidates = pd.DataFrame({
        "obs_idx": [0, 0, 1, 1],
        "arc_id": [10, 11, 20, 21],
    })

    probabilities = [0.8, 0.2, 0.3, 0.7]
    true_arcs = [10, 21]

    assert top1_accuracy(
        candidates, probabilities, true_arcs
    ) == 1.0


def test_candidate_miss_rate():
    candidates = pd.DataFrame({
        "obs_idx": [0, 0, 1],
        "arc_id": [10, 11, 20],
    })

    true_arcs = [10, 21]

    assert candidate_miss_rate(
        candidates, true_arcs
    ) == 0.5


def test_candidate_metrics():
    y_true = [0, 1, 1, 0]
    probabilities = [0.1, 0.8, 0.4, 0.6]

    result = candidate_metrics(y_true, probabilities)

    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["macro_f1"] == 0.5
    assert result["log_loss"] > 0
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]


def test_route_continuity():
    arcs = pd.DataFrame({
        "from_node": [1, 2, 3],
        "to_node": [2, 3, 4],
    })

    assert route_continuity([0, 1, 2], arcs) == 1.0
    assert route_continuity([0, 2], arcs) == 0.0


def test_sequence_metrics():
    true_sequences = [[1, 2, 3]]
    predicted_sequences = [[1, 2, 4]]

    result = sequence_metrics(
        true_sequences, predicted_sequences
    )

    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)


def test_timing_metrics():
    result = timing_metrics(
        {"project_ms": 4, "scoring_ms": 8, "total_ms": 12},
        3,
    )

    assert result["total_ms"] == 12
    assert result["ms_per_observation"] == 4


def test_measure_time():
    result, elapsed = measure_time(sum, [1, 2, 3])

    assert result == 6
    assert elapsed >= 0


def test_invalid_probabilities():
    with pytest.raises(ValueError):
        candidate_metrics([0, 1], [0.2, 1.5])
