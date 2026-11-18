"""PathSnap FastAPI service.

Student-test inference and evaluator assessment use separate endpoints.
Reference routes are accessed only by api.evaluation after prediction.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pathsnap import __version__
from pathsnap.config import load_config
from pathsnap.io import (
    LeakageError,
    discover_trips,
    load_trip,
    read_track,
)
from pathsnap.pipeline import (
    predict_trip,
    load_artifacts as load_pipeline_artifacts,
)

from . import evaluation
from .schemas import PredictRequest, TripInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("pathsnap.api")
cfg = load_config("api")


def load_artifacts() -> dict:
    """Load the shared trained-model artifacts once at startup."""
    artifact_dir = cfg.get("artifacts_dir", "artifacts")
    artifacts = load_pipeline_artifacts(directory=artifact_dir)

    if artifacts.get("mode") == "nearest_arc":
        log.warning("No trained model artifact found; using nearest-arc baseline.")
    else:
        log.info(
            "Loaded trained model: %s",
            artifacts.get("metadata", {}).get("model", "unknown"),
        )

    return artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifacts = load_artifacts()
    log.info("PathSnap started with model mode=%s", app.state.artifacts.get("mode"))
    yield
    app.state.artifacts = None


app = FastAPI(title="PathSnap API", version=__version__, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.get("cors_allow_origins", ["*"]),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = (time.perf_counter() - started) * 1000
        client = request.client.host if request.client else "?"
        log.exception(
            "%s %s from %s failed after %.1f ms",
            request.method,
            request.url.path,
            client,
            elapsed,
        )
        raise

    elapsed = (time.perf_counter() - started) * 1000
    client = request.client.host if request.client else "?"
    log.info(
        "%s %s from %s -> %s in %.1f ms",
        request.method,
        request.url.path,
        client,
        response.status_code,
        elapsed,
    )
    return response


def _student_test_trips():
    """Discover student-test trips; this partition has no evaluator labels."""
    return discover_trips("student_test")


def _evaluator_trips():
    """Discover evaluator trips, subject to the protected-partition guard."""
    try:
        return discover_trips("evaluator")
    except LeakageError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _trip_listing(found: dict) -> list[TripInfo]:
    """Build public trip metadata without exposing dataset file paths."""
    output = []

    for trip_id, files in sorted(found.items()):
        track = read_track(files.track)
        duration = (
            float(track["t"].iloc[-1] - track["t"].iloc[0])
            if len(track) > 1
            else 0.0
        )
        output.append(
            TripInfo(
                trip_id=trip_id,
                n_observations=len(track),
                duration_s=max(duration, 0.0),
            )
        )

    return output


def _run_prediction(trip_id: str, files, request: Request) -> dict:
    """Run inference without loading a reference route."""
    trip = load_trip(files, with_route=False)
    artifacts = (
        getattr(request.app.state, "artifacts", None)
        or {"mode": "nearest_arc"}
    )

    log.info(
        "Prediction started: trip=%s points=%d",
        trip_id,
        len(trip.track),
    )

    try:
        result = predict_trip(trip, artifacts)
    except (ValueError, RuntimeError) as exc:
        log.exception("Prediction failed for trip %s", trip_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    log.info(
        "Prediction complete: trip=%s points=%d model=%s timings=%s",
        trip_id,
        len(trip.track),
        result.get("model"),
        result.get("timings_ms", {}),
    )
    return result


@app.get("/health")
def health(request: Request):
    artifacts = getattr(request.app.state, "artifacts", None)
    return {
        "status": "ok",
        "version": __version__,
        "model_loaded": artifacts is not None,
        "model": artifacts.get("mode") if artifacts else None,
    }


# ---------------------------------------------------------
# Student-test workflow: prediction only, no ground truth.
# ---------------------------------------------------------

@app.get("/student-test/trips", response_model=list[TripInfo])
def student_test_trips():
    found = _student_test_trips()
    output = _trip_listing(found)
    log.info("Listed %d student-test trips", len(output))
    return output


@app.post("/student-test/predict")
def student_test_predict(req: PredictRequest, request: Request):
    found = _student_test_trips()

    if req.trip_id not in found:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown student-test trip {req.trip_id}",
        )

    result = _run_prediction(req.trip_id, found[req.trip_id], request)

    # No evaluator scoring or reference-route access in this workflow.
    return {**result, "metrics": None}


# ---------------------------------------------------------
# Evaluator workflow: prediction first, scoring afterwards.
# ---------------------------------------------------------

@app.get("/evaluator/trips", response_model=list[TripInfo])
def evaluator_trips():
    found = _evaluator_trips()
    output = _trip_listing(found)
    log.info("Listed %d evaluator trips", len(output))
    return output


@app.post("/evaluator/predict")
def evaluator_predict(req: PredictRequest, request: Request):
    found = _evaluator_trips()

    if req.trip_id not in found:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown evaluator trip {req.trip_id}",
        )

    # First perform inference without loading the reference route.
    result = _run_prediction(req.trip_id, found[req.trip_id], request)

    # Only after inference, evaluate server-side against the reference route.
    try:
        metrics = evaluation.score(req.trip_id, result)
    except (FileNotFoundError, ValueError) as exc:
        log.exception("Evaluation failed for trip %s", req.trip_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Return derived metrics only. Never return the reference route.
    return {**result, "metrics": metrics}


# ---------------------------------------------------------
# Temporary compatibility aliases for existing clients.
# These aliases deliberately use the student-test workflow.
# ---------------------------------------------------------

@app.get("/trips", response_model=list[TripInfo], include_in_schema=False)
def trips():
    return student_test_trips()


@app.post("/predict", include_in_schema=False)
def predict(req: PredictRequest, request: Request):
    return student_test_predict(req, request)
