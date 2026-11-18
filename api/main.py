"""PathSnap API (runs on Laptop 1).

Start with ``scripts/run_api.sh`` (sets PATHSNAP_MODE=serve so the evaluator
partition becomes readable here, and only here).

Endpoints
---------
GET  /health   -> server status
GET  /trips    -> the evaluator trip ids shown in the UI (no labels)
POST /predict  -> run the full pipeline on one trip, then score it server-side
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pathsnap import __version__
from pathsnap.config import load_config
from pathsnap.io import discover_trips, load_trip
from pathsnap.pipeline import predict_trip

from .schemas import PredictRequest, TripInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")

cfg = load_config("api")
app = FastAPI(title="PathSnap API", version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=cfg.get("cors_allow_origins", ["*"]),
                   allow_methods=["*"], allow_headers=["*"])

ARTIFACTS = None  # TODO(P7): load model, preprocessing and HMM config once at startup


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    client = request.client.host if request.client else "?"
    log.info("%s %s from %s -> %s in %.0f ms", request.method, request.url.path, client, response.status_code, ms)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "model_loaded": ARTIFACTS is not None}


@app.get("/trips", response_model=list[TripInfo])
def trips():
    found = discover_trips("evaluator")
    out = []
    for trip_id, files in found.items():
        n_obs = sum(1 for _ in files.track.open())
        out.append(TripInfo(trip_id=trip_id, n_observations=n_obs))
    log.info("Listing %d evaluator trips", len(out))
    return out


@app.post("/predict")
def predict(req: PredictRequest):
    found = discover_trips("evaluator")
    if req.trip_id not in found:
        raise HTTPException(status_code=404, detail=f"Unknown trip {req.trip_id}")
    trip = load_trip(found[req.trip_id], with_route=False)  # no ground truth in the pipeline
    log.info("Predicting trip %s (%d observations)", req.trip_id, len(trip.track))
    try:
        result = predict_trip(trip, ARTIFACTS)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="Prediction pipeline not implemented yet")
    # TODO(P7): metrics = evaluation.score(req.trip_id, result)  -- server-side only
    return {"trip_id": req.trip_id, **result}
