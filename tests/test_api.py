"""Toy-fixture tests for the API and UI deployment path."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pathsnap.io import discover_trips, load_trip
from pathsnap.pipeline import predict_trip

TOY = Path(__file__).parent / "fixtures" / "toy"


def test_nearest_arc_pipeline_uses_only_track_and_network():
    files = discover_trips("train", root=TOY)["toy01"]
    trip = load_trip(files, with_route=False)
    assert trip.route is None

    result = predict_trip(trip, {"mode": "nearest_arc"})

    assert result["model"] == "nearest_arc_baseline"
    assert len(result["pred_arcs"]) == len(trip.track) == 5
    assert result["route_arc_ids"]
    assert len(result["gps"]) == 5
    assert result["route_lonlat"]
    assert result["network_lonlat"]
    assert result["timings_ms"]["total"] >= 0



def test_api_health_trips_and_predict_on_toy_fixture(monkeypatch):
    from api import main

    toy_files = discover_trips("train", root=TOY)

    # API now uses student-test trips for prediction.
    monkeypatch.setattr(main, "_student_test_trips", lambda: toy_files)

    with TestClient(main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        trips = client.get("/trips")
        assert trips.status_code == 200
        assert trips.json() == [
            {
                "trip_id": "toy01",
                "n_observations": 5,
                "duration_s": 5.0,
            }
        ]

        response = client.post("/predict", json={"trip_id": "toy01"})
        assert response.status_code == 200, response.text

        body = response.json()
        assert len(body["pred_arcs"]) == 5
        assert body["metrics"] is None
        assert "route" not in body

        missing = client.post("/predict", json={"trip_id": "not-a-trip"})
        assert missing.status_code == 404



def test_evaluation_reads_reference_only_server_side(tmp_path, monkeypatch):
    import shutil

    from api import evaluation

    evaluator_root = tmp_path / "evaluator"
    trip_dir = evaluator_root / "toy01"
    trip_dir.mkdir(parents=True)
    source_dir = TOY / "toy01"
    for suffix in ("track", "nodes", "arcs", "route"):
        shutil.copy2(source_dir / f"toy01.{suffix}", trip_dir / f"toy01.{suffix}")

    monkeypatch.setenv("PATHSNAP_MODE", "serve")
    monkeypatch.setattr(evaluation, "repo_path", lambda relative: evaluator_root)
    monkeypatch.setattr(
        evaluation, "load_config",
        lambda name: {"partitions": {"evaluator": "data/raw/evaluator"}},
    )

    metrics = evaluation.score("toy01", {"route_arc_ids": [0, 2]})
    assert metrics["route_precision"] == 1.0
    assert metrics["route_recall"] == 1.0
    assert metrics["route_f1"] == 1.0
    assert metrics["route_exact_match"] is True
    assert "reference_route" not in metrics
    assert "route_arc_ids" not in metrics


def test_student_test_endpoints_work_without_evaluator_access(monkeypatch):
    from api import main

    toy_files = discover_trips("train", root=TOY)
    monkeypatch.setattr(main, "_student_test_trips", lambda: toy_files)
    monkeypatch.delenv("PATHSNAP_MODE", raising=False)

    with TestClient(main.app) as client:
        trips = client.get("/student-test/trips")
        assert trips.status_code == 200
        assert len(trips.json()) == 1
        assert trips.json()[0]["trip_id"] == "toy01"

        response = client.post(
            "/student-test/predict",
            json={"trip_id": "toy01"},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["metrics"] is None
        assert len(body["pred_arcs"]) == 5
        assert "route" not in body


def test_evaluator_endpoints_are_blocked_without_serve_mode(monkeypatch):
    from api import main

    monkeypatch.delenv("PATHSNAP_MODE", raising=False)

    with TestClient(main.app) as client:
        trips = client.get("/evaluator/trips")
        assert trips.status_code == 403

        prediction = client.post(
            "/evaluator/predict",
            json={"trip_id": "toy01"},
        )
        assert prediction.status_code == 403


def test_evaluator_prediction_runs_before_scoring(monkeypatch):
    from api import main

    toy_files = discover_trips("train", root=TOY)
    monkeypatch.setenv("PATHSNAP_MODE", "serve")
    monkeypatch.setattr(main, "_evaluator_trips", lambda: toy_files)

    events = []
    original_predict_trip = main.predict_trip

    def tracked_prediction(trip, artifacts):
        assert trip.route is None
        events.append("prediction")
        return original_predict_trip(trip, artifacts)

    def fake_score(trip_id, prediction):
        events.append("scoring")
        assert trip_id == "toy01"
        assert "route" not in prediction
        return {
            "route_precision": 1.0,
            "route_recall": 1.0,
            "route_f1": 1.0,
            "route_exact_match": True,
            "route_edit_similarity": 1.0,
            "route_continuity": 1.0,
            "candidate_miss_rate": None,
            "classifier_top1_accuracy": None,
        }

    monkeypatch.setattr(main, "predict_trip", tracked_prediction)
    monkeypatch.setattr(main.evaluation, "score", fake_score)

    with TestClient(main.app) as client:
        trips = client.get("/evaluator/trips")
        assert trips.status_code == 200
        assert trips.json()[0]["trip_id"] == "toy01"

        response = client.post(
            "/evaluator/predict",
            json={"trip_id": "toy01"},
        )

        assert response.status_code == 200, response.text

        body = response.json()
        assert body["metrics"]["route_f1"] == 1.0
        assert body["metrics"]["route_exact_match"] is True
        assert "route" not in body
        assert "reference_route" not in body

    assert events == ["prediction", "scoring"]
