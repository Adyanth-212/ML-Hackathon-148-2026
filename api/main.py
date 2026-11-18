"""PathSnap FastAPI service (Laptop 1).

The prediction pipeline never receives evaluator labels. The evaluator route is
opened only by ``api.evaluation`` after a prediction has been produced.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pathsnap import __version__
from pathsnap.config import load_config, repo_path
from pathsnap.io import discover_trips, load_trip, read_track
from pathsnap.pipeline import predict_trip

from . import evaluation
from .schemas import PredictRequest, TripInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")
cfg = load_config("api")


def load_artifacts() -> dict:
    """Load a trained bundle once at startup, falling back to the baseline."""
    artifact_dir = repo_path(cfg.get("artifacts_dir", "artifacts"))
    model_path = artifact_dir / cfg.get("model_file", "model.joblib")
    if model_path.is_file():
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict):
            bundle.setdefault("mode", "classifier")
            bundle["artifact_path"] = str(model_path)
            return bundle
        return {"mode": "classifier", "model": bundle, "artifact_path": str(model_path)}
    log.warning("No trained model artifact found at %s; using nearest-arc baseline", model_path)
    return {"mode": "nearest_arc", "model": None}


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
        log.exception("%s %s from %s failed after %.1f ms", request.method, request.url.path, client, elapsed)
        raise
    elapsed = (time.perf_counter() - started) * 1000
    client = request.client.host if request.client else "?"
    log.info("%s %s from %s -> %s in %.1f ms", request.method, request.url.path, client, response.status_code, elapsed)
    return response


def _evaluator_trips():
    # Evaluator access is guarded by pathsnap.io and enabled only by the API
    # launcher (PATHSNAP_MODE=serve).
    return discover_trips("evaluator")


@app.get("/health")
def health(request: Request):
    artifacts = getattr(request.app.state, "artifacts", None)
    return {
        "status": "ok",
        "version": __version__,
        "model_loaded": artifacts is not None,
        "model": artifacts.get("mode") if artifacts else None,
    }


@app.get("/trips", response_model=list[TripInfo])
def trips():
    found = _evaluator_trips()
    output = []
    for trip_id, files in sorted(found.items()):
        track = read_track(files.track)
        duration = float(track["t"].iloc[-1] - track["t"].iloc[0]) if len(track) > 1 else 0.0
        output.append(TripInfo(trip_id=trip_id, n_observations=len(track), duration_s=max(duration, 0.0)))
    log.info("Listed %d evaluator trips", len(output))
    return output


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    found = _evaluator_trips()
    if req.trip_id not in found:
        raise HTTPException(status_code=404, detail=f"Unknown trip {req.trip_id}")
    files = found[req.trip_id]
    trip = load_trip(files, with_route=False)
    artifacts = getattr(request.app.state, "artifacts", None) or {"mode": "nearest_arc"}
    log.info("Prediction started: trip=%s points=%d", req.trip_id, len(trip.track))

    try:
        result = predict_trip(trip, artifacts)
        # This is intentionally after predict_trip. Only derived metrics are
        # returned; evaluation.score never returns the reference route itself.
        metrics = evaluation.score(req.trip_id, result)
    except FileNotFoundError as exc:
        log.exception("Evaluation data missing for trip %s", req.trip_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        log.exception("Prediction failed for trip %s", req.trip_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    timings = result.get("timings_ms", {})
    log.info(
        "Prediction complete: trip=%s points=%d model=%s timings=%s metrics=%s",
        req.trip_id, len(trip.track), result.get("model"), timings, metrics,
    )
    return {**result, "metrics": metrics}
