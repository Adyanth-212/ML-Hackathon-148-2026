
"""Inference pipeline used by the API.

The first deployment milestone deliberately uses a nearest-directed-arc baseline.
It accepts no reference route, so evaluator labels cannot leak into prediction.
The returned payload is compatible with the later classifier/HMM integration.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .geo import LocalProjector, TripGeometry
from .labels import obs_heading


def _deduplicate(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        value = int(value)
        if not out or out[-1] != value:
            out.append(value)
    return out

def fill_missing(pred: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Fill missing predictions from the nearest-in-time observed prediction."""
        have = np.flatnonzero(pred >= 0)

        if len(have) == 0 or len(have) == len(pred):
            return pred

        out = pred.copy()
        missing = np.flatnonzero(pred < 0)

        positions = np.searchsorted(timestamps[have], timestamps[missing])

        left = have[np.clip(positions - 1, 0, len(have) - 1)]
        right = have[np.clip(positions, 0, len(have) - 1)]

        use_right = (
            np.abs(timestamps[right] - timestamps[missing])
            < np.abs(timestamps[missing] - timestamps[left])
        )

        out[missing] = np.where(use_right, pred[right], pred[left])
        return out

BASELINE_HEADING_WEIGHT_M = 20.0
HEADING_MIN_MOVE_M = 5.0


def baseline_scores(geom, cands, heading_weight_m=BASELINE_HEADING_WEIGHT_M):
    """Score candidates using distance and GPS/road heading agreement."""
    scores = -cands["dist_m"].to_numpy(dtype=np.float64)

    if heading_weight_m > 0 and len(cands):
        obs = cands["obs_idx"].to_numpy(dtype=np.int64)
        arc = cands["arc_id"].to_numpy(dtype=np.int64)

        ux, uy, valid = obs_heading(geom, HEADING_MIN_MOVE_M)

        vx = geom.bx[arc] - geom.ax[arc]
        vy = geom.by[arc] - geom.ay[arc]
        length = np.hypot(vx, vy)

        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = (ux[obs] * vx + uy[obs] * vy) / length

        mismatch = (1.0 - cosine) / 2.0
        mismatch = np.where(valid[obs] & (length > 0), mismatch, 0.0)

        scores -= heading_weight_m * mismatch

    return scores


def argmax_per_obs(cands, scores, n_obs):
    """Select the highest-scoring candidate for each GPS observation."""
    pred = np.full(n_obs, -1, dtype=np.int64)

    if len(cands):
        ranked = pd.DataFrame({
            "obs": cands["obs_idx"].to_numpy(),
            "arc": cands["arc_id"].to_numpy(),
            "score": np.asarray(scores),
        })

        best = ranked.loc[
            ranked.groupby("obs", sort=False)["score"].idxmax()
        ]

        pred[best["obs"].to_numpy(dtype=np.int64)] = (
            best["arc"].to_numpy(dtype=np.int64)
        )

    return pred


def predict_trip(trip, artifacts=None) -> dict:
    """Predict one arc per GPS observation without accessing ground truth."""
    artifacts = artifacts or {}
    started = time.perf_counter()
    timings = {}

    if trip.track is None or trip.track.empty:
        raise ValueError(f"Trip {trip.trip_id} has no GPS observations")

    if trip.nodes is None or trip.arcs is None or trip.arcs.empty:
        raise ValueError(f"Trip {trip.trip_id} is missing its road network")

    # 1. Project the trip into local metric coordinates.
    t0 = time.perf_counter()
    geom = TripGeometry(trip.track, trip.nodes, trip.arcs)
    timings["project_ms"] = (time.perf_counter() - t0) * 1000

    # 2. Generate nearby directed-arc candidates.
    t0 = time.perf_counter()
    candidates, no_candidate_obs = generate_candidates(
        geom, trip.trip_id
    )
    timings["candidates_ms"] = (time.perf_counter() - t0) * 1000

    # 3. Score candidates using an artifact or the distance + heading baseline.
    t0 = time.perf_counter()
    scorer = artifacts.get("scorer")

    if scorer is not None:
        scores = np.asarray(scorer(geom, candidates))
    else:
        scores = baseline_scores(geom, candidates)

    timings["scoring_ms"] = (time.perf_counter() - t0) * 1000

    # 4. Decode the best candidate for each observation.
    t0 = time.perf_counter()
    decoder = artifacts.get("decoder")

    if decoder is not None:
        pred = decoder(geom, candidates, scores)
    else:
        pred = argmax_per_obs(candidates, scores, geom.n_obs)

    pred = fill_missing(
        np.asarray(pred, dtype=np.int64),
        geom.t,
    )

    timings["decode_ms"] = (time.perf_counter() - t0) * 1000

    # 5. Construct route geometry from predicted directed arcs.
    t0 = time.perf_counter()

    valid_pred = pred[pred >= 0]
    route_arc_ids = _deduplicate(valid_pred.tolist())

    arc_pairs = trip.arcs[
        ["from_node", "to_node"]
    ].to_numpy(dtype=np.int64)

    route_node_ids = []

    for arc_id in route_arc_ids:
        u, v = arc_pairs[arc_id]

        if not route_node_ids:
            route_node_ids.extend([int(u), int(v)])
        elif route_node_ids[-1] == int(u):
            route_node_ids.append(int(v))
        else:
            route_node_ids.extend([int(u), int(v)])

    if route_node_ids:
        route_lon = trip.nodes["lon"].to_numpy(dtype=float)[route_node_ids]
        route_lat = trip.nodes["lat"].to_numpy(dtype=float)[route_node_ids]
        route_lonlat = np.column_stack(
            [route_lon, route_lat]
        ).round(7).tolist()
    else:
        route_lonlat = []

    # 6. Build a bounded network overlay around the GPS track.
    projector = geom.proj
    gps_x, gps_y = geom.obs_x, geom.obs_y
    ax, ay, bx, by = geom.ax, geom.ay, geom.bx, geom.by

    margin_m = 100.0

    in_view = (
        (np.maximum(ax, bx) >= gps_x.min() - margin_m)
        & (np.minimum(ax, bx) <= gps_x.max() + margin_m)
        & (np.maximum(ay, by) >= gps_y.min() - margin_m)
        & (np.minimum(ay, by) <= gps_y.max() + margin_m)
    )

    nearby_ids = np.flatnonzero(in_view)[:3000]
    network_lonlat = []

    if len(nearby_ids):
        endpoint_x = np.column_stack(
            [ax[nearby_ids], bx[nearby_ids]]
        ).ravel()

        endpoint_y = np.column_stack(
            [ay[nearby_ids], by[nearby_ids]]
        ).ravel()

        endpoint_lon, endpoint_lat = projector.inverse(
            endpoint_x, endpoint_y
        )

        segments = np.column_stack(
            [endpoint_lon, endpoint_lat]
        ).reshape(-1, 2, 2)

        network_lonlat = [
            segment.astype(float).tolist()
            for segment in segments
        ]

    # Distances for the selected candidates.
    distance_lookup = {
        (int(row.obs_idx), int(row.arc_id)): float(row.dist_m)
        for row in candidates[
            ["obs_idx", "arc_id", "dist_m"]
        ].itertuples(index=False)
    }

    nearest_distances = [
        distance_lookup.get((i, int(arc_id)))
        if arc_id >= 0 else None
        for i, arc_id in enumerate(pred)
    ]

    timings["route_ms"] = (time.perf_counter() - t0) * 1000
    timings["total"] = (time.perf_counter() - started) * 1000

    # 7. Report which prediction components were used.
    return {
        "trip_id": trip.trip_id,
        "gps": trip.track[
            ["lon", "lat"]
        ].to_numpy(dtype=float).round(7).tolist(),
        "pred_arcs": pred.tolist(),
        "route_arc_ids": valid_pred.tolist(),
        "route_lonlat": route_lonlat,
        "network_lonlat": network_lonlat,
        "nearest_dist_m": nearest_distances,
        "timings_ms": {
            key: round(value, 2)
            for key, value in timings.items()
        },
        "no_candidate_obs": np.asarray(
            no_candidate_obs, dtype=int
        ).tolist(),
        "method": {
            "scorer": (
                "artifact" if scorer is not None
                else "baseline: distance + heading"
            ),
            "decoder": (
                "artifact" if decoder is not None
                else "per-observation argmax"
            ),
            "n_candidates": int(len(candidates)),
        },
        "model": (
    "nearest_arc_baseline"
    if artifacts.get("mode") == "nearest_arc" or scorer is None
    else "trained_artifact"
),
    }

    
    

    