"""End-to-end inference for one trip (used by the API and the offline evaluation).

``predict_trip`` deliberately takes NO reference route: ground truth must never
enter the prediction pipeline. Scoring happens afterwards in api/evaluation.py.

Stages: project -> candidates (configs/candidates.yaml) -> score candidates ->
pick one arc per observation -> route geometry + network overlay for the UI.

Scoring, in order of preference:
1. ``artifacts["scorer"](geom, cands) -> P(candidate is the true arc)``, as built by
   ``load_artifacts()`` from the exported classifier (``scripts/export_artifacts.py``).
2. ``artifacts["model"]`` (a fitted estimator with ``predict_proba``, as loaded by the
   API's own loader): wrapped into the same feature-based scorer automatically.
3. Otherwise the **baseline rule**: score = -(dist_m + heading_weight_m * mismatch),
   where mismatch = (1 - cos(GPS heading, arc direction)) / 2. With heading_weight_m = 0
   this is the plain nearest-arc rule. ``artifacts={"mode": "nearest_arc"}`` forces it.
An optional ``artifacts["decoder"](geom, cands, scores) -> arc per obs`` (the HMM) replaces
the per-observation argmax. ``load_artifacts()`` picks it up from
``pathsnap.hmm.make_decoder(config)`` once that exists.

Observations without any candidate inherit the arc of the nearest observation in time
that has one, and are listed in ``no_candidate_obs``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .geo import TripGeometry
from .labels import obs_heading

BASELINE_HEADING_WEIGHT_M = 20.0  # chosen on TRAIN by scripts/eval_baseline.py (heading sweep)
HEADING_MIN_MOVE_M = 5.0
NETWORK_OVERLAY_MARGIN_M = 100.0
NETWORK_OVERLAY_MAX_ARCS = 3000


# --------------------------------------------------------------------------- scoring
def baseline_scores(geom: TripGeometry, cands: pd.DataFrame,
                    heading_weight_m: float = BASELINE_HEADING_WEIGHT_M) -> np.ndarray:
    """Higher = better. Distance plus GPS-heading / arc-direction agreement (no labels)."""
    scores = -cands["dist_m"].to_numpy(np.float64)
    if heading_weight_m > 0 and len(cands):
        obs = cands["obs_idx"].to_numpy(np.int64)
        arc = cands["arc_id"].to_numpy(np.int64)
        ux, uy, valid = obs_heading(geom, HEADING_MIN_MOVE_M)
        vx, vy = geom.bx[arc] - geom.ax[arc], geom.by[arc] - geom.ay[arc]
        length = np.hypot(vx, vy)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = (ux[obs] * vx + uy[obs] * vy) / length
        mismatch = np.where(valid[obs] & (length > 0), (1.0 - cosine) / 2.0, 0.0)
        scores -= heading_weight_m * mismatch
    return scores


def classifier_scorer(model):
    """Wrap a fitted estimator (predict_proba on FEATURE_COLUMNS) as a pipeline scorer."""
    from .features import FEATURE_COLUMNS, build_features

    def scorer(geom: TripGeometry, cands: pd.DataFrame) -> np.ndarray:
        if len(cands) == 0:
            return np.zeros(0)
        X = build_features(geom, cands)[FEATURE_COLUMNS].to_numpy(np.float32)
        return model.predict_proba(X)[:, 1]

    return scorer


def load_artifacts(directory: str = "artifacts", use_decoder: bool = True) -> dict:
    """Load the exported classifier (and the HMM decoder when available) for predict_trip.

    Returns ``{"mode", "model", "scorer", "metadata", ["decoder"]}``. Without
    ``model.joblib`` it returns ``{"mode": "nearest_arc"}``, i.e. the rule-based baseline.
    """
    import json

    import joblib

    from .config import load_config, repo_path
    from .features import FEATURE_COLUMNS

    root = repo_path(directory)
    model_path = root / "model.joblib"
    if not model_path.exists():
        return {"mode": "nearest_arc"}
    model = joblib.load(model_path)
    if isinstance(model, dict):  # bundle format {"model": est, ...}
        model = model["model"]
    meta = json.loads((root / "metadata.json").read_text()) if (root / "metadata.json").exists() else {}
    if meta.get("features", FEATURE_COLUMNS) != FEATURE_COLUMNS:
        raise ValueError("artifacts/model.joblib was trained on a different feature list; re-export it")

    out = {"mode": "classifier", "model": model, "scorer": classifier_scorer(model), "metadata": meta,
           "artifact_path": str(model_path)}
    if use_decoder:
        try:
            from . import hmm
            make = getattr(hmm, "make_decoder", None)
            if make is not None:
                out["decoder"] = make(load_config("hmm"))
        except NotImplementedError:
            pass
    return out


def _resolve_scorer(artifacts: dict):
    baseline = f"baseline: dist + {BASELINE_HEADING_WEIGHT_M:g} m x heading mismatch"
    if artifacts.get("mode") == "nearest_arc":
        return None, baseline
    if artifacts.get("scorer") is not None:
        return artifacts["scorer"], artifacts.get("metadata", {}).get("model", "artifact")
    if artifacts.get("model") is not None and hasattr(artifacts["model"], "predict_proba"):
        return classifier_scorer(artifacts["model"]), artifacts.get("metadata", {}).get("model", "artifact")
    return None, baseline


# --------------------------------------------------------------------------- decoding
def argmax_per_obs(cands: pd.DataFrame, scores: np.ndarray, n_obs: int) -> np.ndarray:
    """Best-scoring arc per observation (-1 where there is no candidate). Ties: first row,
    i.e. the nearer / lower arc_id candidate, because rows are sorted by (obs, dist, arc_id)."""
    pred = np.full(n_obs, -1, dtype=np.int64)
    if len(cands):
        d = pd.DataFrame({"obs": cands["obs_idx"].to_numpy(), "arc": cands["arc_id"].to_numpy(),
                          "s": np.asarray(scores)})
        best = d.loc[d.groupby("obs", sort=False)["s"].idxmax()]
        pred[best["obs"].to_numpy(np.int64)] = best["arc"].to_numpy(np.int64)
    return pred


def fill_missing(pred: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Give observations without a prediction the arc of the nearest-in-time observation that has one."""
    have = np.flatnonzero(pred >= 0)
    if len(have) == 0 or len(have) == len(pred):
        return pred
    out = pred.copy()
    miss = np.flatnonzero(pred < 0)
    pos = np.searchsorted(t[have], t[miss])
    left = have[np.clip(pos - 1, 0, len(have) - 1)]
    right = have[np.clip(pos, 0, len(have) - 1)]
    use_right = np.abs(t[right] - t[miss]) < np.abs(t[miss] - t[left])
    out[miss] = np.where(use_right, pred[right], pred[left])
    return out


# --------------------------------------------------------------------------- geometry for the UI
def _deduplicate(values) -> list[int]:
    out: list[int] = []
    for v in values:
        v = int(v)
        if not out or out[-1] != v:
            out.append(v)
    return out


def route_lonlat(trip, route_arc_ids: list[int]) -> list[list[float]]:
    """Polyline through the predicted arcs (consecutive duplicates removed). Arcs that do not
    connect are joined by a straight line; the HMM stage replaces this with connected routes."""
    if not route_arc_ids:
        return []
    pairs = trip.arcs[["from_node", "to_node"]].to_numpy(np.int64)
    nodes: list[int] = []
    for a in route_arc_ids:
        u, v = int(pairs[a, 0]), int(pairs[a, 1])
        if nodes and nodes[-1] == u:
            nodes.append(v)
        else:
            nodes.extend([u, v])
    lon = trip.nodes["lon"].to_numpy(float)[nodes]
    lat = trip.nodes["lat"].to_numpy(float)[nodes]
    return np.column_stack([lon, lat]).round(7).tolist()


def network_overlay(geom: TripGeometry) -> list[list[list[float]]]:
    """Arcs inside the track's bounding box + margin (at most NETWORK_OVERLAY_MAX_ARCS), as lon/lat segments."""
    m = NETWORK_OVERLAY_MARGIN_M
    gx, gy = geom.obs_x, geom.obs_y
    ax, ay, bx, by = geom.ax, geom.ay, geom.bx, geom.by
    in_view = ((np.maximum(ax, bx) >= gx.min() - m) & (np.minimum(ax, bx) <= gx.max() + m)
               & (np.maximum(ay, by) >= gy.min() - m) & (np.minimum(ay, by) <= gy.max() + m))
    ids = np.flatnonzero(in_view)[:NETWORK_OVERLAY_MAX_ARCS]
    if not len(ids):
        return []
    ex = np.column_stack([ax[ids], bx[ids]]).ravel()
    ey = np.column_stack([ay[ids], by[ids]]).ravel()
    lon, lat = geom.proj.inverse(ex, ey)
    return np.column_stack([lon, lat]).reshape(-1, 2, 2).astype(float).tolist()


def selected_distances(cands: pd.DataFrame, pred: np.ndarray) -> list[float | None]:
    """Distance (m) from each GPS point to its predicted arc; None where it is not a candidate."""
    out = np.full(len(pred), np.nan)
    if len(cands):
        obs = cands["obs_idx"].to_numpy(np.int64)
        hit = cands["arc_id"].to_numpy(np.int64) == pred[obs]
        out[obs[hit]] = cands["dist_m"].to_numpy(float)[hit]
    return [None if np.isnan(v) else round(float(v), 3) for v in out]


# --------------------------------------------------------------------------- entry point
def predict_trip(trip, artifacts=None) -> dict:
    """Predict one arc per GPS observation without accessing ground truth."""
    artifacts = artifacts or {}
    if trip.track is None or trip.track.empty:
        raise ValueError(f"Trip {trip.trip_id} has no GPS observations")
    if trip.nodes is None or trip.arcs is None or trip.arcs.empty:
        raise ValueError(f"Trip {trip.trip_id} is missing its road network")
    started = time.perf_counter()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    geom = TripGeometry.from_trip(trip)
    timings["project_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cands, no_cand = generate_candidates(geom, trip.trip_id)
    timings["candidates_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scorer, scorer_name = _resolve_scorer(artifacts)
    scores = np.asarray(scorer(geom, cands)) if scorer is not None else baseline_scores(geom, cands)
    timings["scoring_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    decoder = artifacts.get("decoder")
    decoder_breaks = np.zeros(geom.n_obs, dtype=bool)
    if decoder is not None:
        decoded = decoder(geom, cands, scores)
        if isinstance(decoded, tuple) and len(decoded) == 2:
            pred, decoder_breaks = decoded
        else:
            pred = decoded
        pred = np.asarray(pred, dtype=np.int64)
        decoder_breaks = np.asarray(decoder_breaks, dtype=bool)
        if pred.shape != (geom.n_obs,) or decoder_breaks.shape != (geom.n_obs,):
            raise ValueError("Decoder must return one arc and one break flag per observation.")
    else:
        pred = argmax_per_obs(cands, scores, geom.n_obs)
    pred = fill_missing(np.asarray(pred, dtype=np.int64), geom.t)
    timings["decode_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    valid_pred = pred[pred >= 0]
    route = route_lonlat(trip, _deduplicate(valid_pred))
    timings["route_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    overlay = network_overlay(geom)
    nearest = selected_distances(cands, pred)
    timings["overlay_ms"] = (time.perf_counter() - t0) * 1000
    timings["total"] = (time.perf_counter() - started) * 1000
    timings["total_ms"] = timings["total"]

    return {
        "trip_id": trip.trip_id,
        "gps": trip.track[["lon", "lat"]].to_numpy(float).round(7).tolist(),
        "pred_arcs": pred.tolist(),
        "route_arc_ids": valid_pred.tolist(),
        "route_lonlat": route,
        "network_lonlat": overlay,
        "nearest_dist_m": nearest,
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        "no_candidate_obs": np.asarray(no_cand, dtype=int).tolist(),
        "decoder_breaks": np.asarray(decoder_breaks, dtype=int).tolist(),
        "method": {"scorer": scorer_name,
                   "decoder": "artifact" if decoder is not None else "per-observation argmax",
                   "n_candidates": int(len(cands))},
        "model": "nearest_arc_baseline" if scorer is None else "trained_artifact",
    }
