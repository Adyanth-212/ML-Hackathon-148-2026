"""End-to-end inference for one trip.

``predict_trip`` deliberately takes NO reference route: ground truth must never
enter the prediction pipeline. Scoring happens afterwards in api/evaluation.py.

Stages: project -> candidates (configs/candidates.yaml) -> score candidates ->
pick one arc per observation -> route geometry.

Scoring, in order of preference:
1. ``artifacts["scorer"](geom, cands) -> P(candidate is the true arc)``: the trained
   classifier (+ features) once it exists; higher is better.
2. Otherwise the **baseline rule**: score = -(dist_m + heading_weight_m * mismatch),
   where mismatch = (1 - cos(GPS heading, arc direction)) / 2. With heading_weight_m = 0
   this is the plain nearest-arc rule.
An optional ``artifacts["decoder"](geom, cands, scores) -> arc per obs`` (the HMM) replaces
the per-observation argmax.

Observations without any candidate inherit the arc of the nearest observation in time
that has one, and are listed in ``no_candidate_obs``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .geo import TripGeometry

BASELINE_HEADING_WEIGHT_M = 20.0  # chosen on TRAIN by scripts/eval_baseline.py (heading sweep)
HEADING_MIN_MOVE_M = 5.0


def baseline_scores(geom: TripGeometry, cands: pd.DataFrame,
                    heading_weight_m: float = BASELINE_HEADING_WEIGHT_M) -> np.ndarray:
    """Higher = better. Uses only GPS + network geometry."""
    score = -cands["dist_m"].to_numpy(np.float64)
    if heading_weight_m > 0 and len(cands):
        obs, arc = cands["obs_idx"].to_numpy(), cands["arc_id"].to_numpy()
        # mismatch for each (obs, arc) pair, computed per unique arc block to stay vectorised
        mm = _pair_heading_mismatch(geom, obs, arc)
        score -= heading_weight_m * mm
    return score


def _pair_heading_mismatch(geom: TripGeometry, obs: np.ndarray, arc: np.ndarray) -> np.ndarray:
    from .labels import obs_heading

    ux, uy, valid = obs_heading(geom, HEADING_MIN_MOVE_M)
    vx, vy = geom.bx[arc] - geom.ax[arc], geom.by[arc] - geom.ay[arc]
    ln = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (ux[obs] * vx + uy[obs] * vy) / ln
    mm = (1.0 - cos) / 2.0
    return np.where(valid[obs] & (ln > 0), mm, 0.0)


def argmax_per_obs(cands: pd.DataFrame, scores: np.ndarray, n_obs: int) -> np.ndarray:
    """Best-scoring arc per observation (-1 where there is no candidate). Ties: first row,
    i.e. the nearer / lower arc_id candidate, because rows are sorted by (obs, dist, arc_id)."""
    pred = np.full(n_obs, -1, dtype=np.int64)
    if len(cands):
        d = pd.DataFrame({"obs": cands["obs_idx"].to_numpy(), "arc": cands["arc_id"].to_numpy(), "s": scores})
        best = d.loc[d.groupby("obs", sort=False)["s"].idxmax()]
        pred[best["obs"].to_numpy()] = best["arc"].to_numpy()
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


def route_lonlat(trip, arc_seq: np.ndarray) -> list[list[float]]:
    """Polyline through the predicted arcs (consecutive duplicates removed). Arcs that do not
    connect are joined by a straight line; the HMM stage replaces this with shortest-path fill."""
    if len(arc_seq) == 0:
        return []
    keep = np.r_[True, arc_seq[1:] != arc_seq[:-1]]
    seq = arc_seq[keep]
    f = trip.arcs["from_node"].to_numpy()[seq]
    t = trip.arcs["to_node"].to_numpy()[seq]
    nodes = np.ravel(np.c_[f, t])
    nodes = nodes[np.r_[True, nodes[1:] != nodes[:-1]]]
    lon, lat = trip.nodes["lon"].to_numpy()[nodes], trip.nodes["lat"].to_numpy()[nodes]
    return np.c_[lon, lat].round(7).tolist()


def predict_trip(trip, artifacts=None) -> dict:
    """Per-observation predicted arc ids plus route geometry and timings. No reference route."""
    artifacts = artifacts or {}
    timings = {}
    t0 = time.perf_counter()
    geom = TripGeometry.from_trip(trip)
    timings["project_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cands, no_cand = generate_candidates(geom, trip.trip_id)
    timings["candidates_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scorer = artifacts.get("scorer")
    scores = scorer(geom, cands) if scorer is not None else baseline_scores(geom, cands)
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
    route = route_lonlat(trip, pred[pred >= 0])
    timings["route_ms"] = (time.perf_counter() - t0) * 1000
    timings["total_ms"] = sum(timings.values())

    return {
        "gps": trip.track[["lon", "lat"]].to_numpy().round(7).tolist(),
        "pred_arcs": pred.tolist(),
        "route_lonlat": route,
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        "no_candidate_obs": no_cand.tolist(),
        "decoder_breaks": decoder_breaks.tolist(),
        "method": {"scorer": "artifact" if scorer is not None else
                   f"baseline: dist + {BASELINE_HEADING_WEIGHT_M:g} m x heading mismatch",
                   "decoder": "artifact" if decoder is not None else "per-observation argmax",
                   "n_candidates": int(len(cands))},
    }
