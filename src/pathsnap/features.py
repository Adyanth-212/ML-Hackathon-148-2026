"""Feature engineering for (observation, candidate arc) pairs.

Input: one trip's candidate table (candidate-table contract, docs/REFERENCE.md: ``trip_id, obs_idx, arc_id,
dist_m, proj_frac[, label]``, rows sorted by obs then distance) and the trip's
``TripGeometry`` (UTM metres). Output: the same rows plus numeric feature columns.
Only GPS + road-network information is used, never labels or the reference
route, so the same code runs at inference time.

Feature groups (motion, endpoint distances, arc length/bearing, heading error from
the previous and next GPS step, and distance/length ratio follow Advitha's first
``pathsnap_features`` draft; here they are vectorised and computed in projected metres):

* distance  - dist_m, rank, gap to the nearest candidate, ratio to it, projection
              fraction (unclipped) and an outside-the-segment flag, endpoint distances,
              distance / arc length
* direction - cos of the angle between the GPS heading (previous step, next step,
              central difference) and the arc direction; validity flag; gap to the best
              candidate's cos
* context   - mean / max distance from the neighbouring points t-2..t+2 to the same arc,
              and the gap to the best candidate's mean
* motion    - dt to previous / next point, speeds, acceleration, turn angle
* network   - arc length, in/out degree of both end nodes, reverse-twin flag
* sequence  - is the arc (or a neighbour of it) the best rule-based arc of the previous /
              next observation (rule = distance + heading, no labels)
* density   - number of candidates within 10 m, 25 m and in total

Clipping: speeds at 70 m/s, accelerations at +/-10 m/s^2, dt at 600 s (EDA found sustained
> 60 m/s stretches and gaps up to 42 h; see docs/data_quality_log.md).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import repo_path
from .geo import TripGeometry, point_segment_distance

HEADING_MIN_MOVE_M = 5.0   # central-difference displacement needed to trust the heading
STEP_MIN_MOVE_M = 2.0      # single-step displacement needed to trust prev/next headings
CONTEXT_OFFSETS = (-2, -1, 1, 2)
RULE_HEADING_WEIGHT_M = 20.0  # same rule as pipeline.baseline_scores (chosen on train)

FEATURE_GROUPS: dict[str, list[str]] = {
    "distance": ["dist_m", "dist_rank", "dist_minus_min", "dist_ratio_min", "proj_frac", "proj_outside",
                 "dist_from_node_m", "dist_to_node_m", "dist_over_len"],
    "direction": ["heading_cos", "heading_cos_prev", "heading_cos_next", "heading_valid", "heading_cos_gap"],
    "context": ["ctx_mean_dist_m", "ctx_max_dist_m", "ctx_mean_gap_m"],
    "motion": ["dt_prev_s", "dt_next_s", "speed_prev_mps", "speed_next_mps", "accel_mps2", "turn_deg"],
    "network": ["arc_length_m", "from_in_deg", "from_out_deg", "to_in_deg", "to_out_deg", "has_reverse"],
    "sequence": ["prev_rule_same", "prev_rule_adjacent", "next_rule_same", "next_rule_adjacent"],
    "density": ["n_candidates", "n_within_10m", "n_within_25m"],
}
FEATURE_COLUMNS: list[str] = [c for cols in FEATURE_GROUPS.values() for c in cols]
META_COLUMNS = ["trip_id", "obs_idx", "arc_id"]


def _unit(dx: np.ndarray, dy: np.ndarray, min_len: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.hypot(dx, dy)
    ok = n >= min_len
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, dx / n, 0.0), np.where(ok, dy / n, 0.0), ok


def observation_features(geom: TripGeometry) -> dict[str, np.ndarray]:
    """Per-observation motion and heading quantities (length n_obs)."""
    x, y, t = geom.obs_x, geom.obs_y, geom.t
    n = len(x)
    ip = np.r_[0, np.arange(n - 1)]           # previous index (self for the first point)
    inx = np.r_[np.arange(1, n), n - 1]       # next index (self for the last point)
    dxp, dyp = x - x[ip], y - y[ip]
    dxn, dyn = x[inx] - x, y[inx] - y
    dtp = np.where(np.arange(n) > 0, t - t[ip], np.nan)
    dtn = np.where(np.arange(n) < n - 1, t[inx] - t, np.nan)
    sp = np.hypot(dxp, dyp) / dtp
    sn = np.hypot(dxn, dyn) / dtn
    sp = np.where(np.isfinite(sp), sp, np.where(np.isfinite(sn), sn, 0.0))
    sn = np.where(np.isfinite(sn), sn, sp)
    dtp = np.where(np.isfinite(dtp), dtp, np.nan_to_num(dtn, nan=1.0))
    dtn = np.where(np.isfinite(dtn), dtn, dtp)
    accel = (sn - sp) / np.maximum((dtp + dtn) / 2, 1e-3)

    pux, puy, pok = _unit(dxp, dyp, STEP_MIN_MOVE_M)
    nux, nuy, nok = _unit(dxn, dyn, STEP_MIN_MOVE_M)
    cux, cuy, cok = _unit(x[inx] - x[ip], y[inx] - y[ip], HEADING_MIN_MOVE_M)
    turn = np.degrees(np.arccos(np.clip(pux * nux + puy * nuy, -1, 1)))
    turn = np.where(pok & nok, turn, 0.0)
    return {
        "dt_prev_s": np.minimum(dtp, 600.0), "dt_next_s": np.minimum(dtn, 600.0),
        "speed_prev_mps": np.minimum(sp, 70.0), "speed_next_mps": np.minimum(sn, 70.0),
        "accel_mps2": np.clip(accel, -10, 10), "turn_deg": turn,
        "pux": pux, "puy": puy, "pok": pok, "nux": nux, "nuy": nuy, "nok": nok,
        "cux": cux, "cuy": cuy, "cok": cok,
    }


def _group_reduce(obs: np.ndarray, values: np.ndarray, n_obs: int, how: str) -> np.ndarray:
    out = np.full(n_obs, -np.inf if how == "max" else np.inf)
    (np.maximum if how == "max" else np.minimum).at(out, obs, values)
    return out


def build_features(geom: TripGeometry, cands: pd.DataFrame) -> pd.DataFrame:
    """Candidate rows + every column in FEATURE_COLUMNS (float32). Keeps ``label`` if present."""
    n_obs = geom.n_obs
    obs = cands["obs_idx"].to_numpy(np.int64)
    arc = cands["arc_id"].to_numpy(np.int64)
    d = cands["dist_m"].to_numpy(np.float64)
    tfrac = cands["proj_frac"].to_numpy(np.float64)
    f: dict[str, np.ndarray] = {}

    # --- distance & density
    dmin = _group_reduce(obs, d, n_obs, "min")[obs]
    start = np.r_[0, np.flatnonzero(np.diff(obs)) + 1] if len(obs) else np.zeros(0, int)
    counts = np.diff(np.r_[start, len(obs)])
    f["dist_m"] = d
    f["dist_rank"] = np.arange(len(obs)) - np.repeat(start, counts)
    f["dist_minus_min"] = d - dmin
    f["dist_ratio_min"] = (dmin + 1.0) / (d + 1.0)
    f["proj_frac"] = np.clip(tfrac, -5, 5)
    f["proj_outside"] = ((tfrac < 0) | (tfrac > 1)).astype(float)
    ox, oy = geom.obs_x[obs], geom.obs_y[obs]
    f["dist_from_node_m"] = np.hypot(ox - geom.ax[arc], oy - geom.ay[arc])
    f["dist_to_node_m"] = np.hypot(ox - geom.bx[arc], oy - geom.by[arc])
    alen = geom.arc_lengths()
    f["dist_over_len"] = d / (alen[arc] + 1.0)
    f["n_candidates"] = np.repeat(counts, counts).astype(float)
    f["n_within_10m"] = np.bincount(obs[d <= 10], minlength=n_obs)[obs].astype(float)
    f["n_within_25m"] = np.bincount(obs[d <= 25], minlength=n_obs)[obs].astype(float)

    # --- direction
    of = observation_features(geom)
    vx, vy, vok = _unit(geom.bx[arc] - geom.ax[arc], geom.by[arc] - geom.ay[arc], 1e-6)
    f["heading_cos"] = np.where(of["cok"][obs] & vok, of["cux"][obs] * vx + of["cuy"][obs] * vy, 0.0)
    f["heading_cos_prev"] = np.where(of["pok"][obs] & vok, of["pux"][obs] * vx + of["puy"][obs] * vy, 0.0)
    f["heading_cos_next"] = np.where(of["nok"][obs] & vok, of["nux"][obs] * vx + of["nuy"][obs] * vy, 0.0)
    f["heading_valid"] = of["cok"][obs].astype(float)
    f["heading_cos_gap"] = _group_reduce(obs, f["heading_cos"], n_obs, "max")[obs] - f["heading_cos"]

    # --- context: neighbours t-2..t+2 against the same arc
    ctx = np.zeros((len(CONTEXT_OFFSETS), len(obs)))
    for k, o in enumerate(CONTEXT_OFFSETS):
        j = np.clip(obs + o, 0, n_obs - 1)
        ctx[k], _ = point_segment_distance(geom.obs_x[j], geom.obs_y[j], geom.ax[arc], geom.ay[arc],
                                           geom.bx[arc], geom.by[arc])
    cm = np.r_[ctx, d[None, :]].mean(axis=0)
    f["ctx_mean_dist_m"] = cm
    f["ctx_max_dist_m"] = np.r_[ctx, d[None, :]].max(axis=0)
    f["ctx_mean_gap_m"] = cm - _group_reduce(obs, cm, n_obs, "min")[obs]

    # --- motion (per observation, broadcast to rows)
    for k in FEATURE_GROUPS["motion"]:
        f[k] = of[k][obs]

    # --- network
    n_nodes = len(geom.node_x)
    fn, tn = geom.from_node, geom.to_node
    indeg, outdeg = np.bincount(tn, minlength=n_nodes), np.bincount(fn, minlength=n_nodes)
    f["arc_length_m"] = alen[arc]
    f["from_in_deg"], f["from_out_deg"] = indeg[fn[arc]], outdeg[fn[arc]]
    f["to_in_deg"], f["to_out_deg"] = indeg[tn[arc]], outdeg[tn[arc]]
    keys = fn * np.int64(n_nodes) + tn
    f["has_reverse"] = np.isin(tn[arc] * np.int64(n_nodes) + fn[arc], keys).astype(float)

    # --- sequence: best rule-based arc of the neighbouring observations (no labels)
    rule = d + RULE_HEADING_WEIGHT_M * (1.0 - f["heading_cos"]) / 2.0
    best = np.full(n_obs, -1, dtype=np.int64)
    if len(obs):
        order = np.lexsort((rule, obs))
        first = np.r_[True, obs[order][1:] != obs[order][:-1]]
        best[obs[order][first]] = arc[order][first]
    prev_b = np.where(obs > 0, best[np.maximum(obs - 1, 0)], -1)
    next_b = np.where(obs < n_obs - 1, best[np.minimum(obs + 1, n_obs - 1)], -1)
    pv, nv = prev_b >= 0, next_b >= 0
    f["prev_rule_same"] = (pv & (prev_b == arc)).astype(float)
    f["prev_rule_adjacent"] = (pv & ((prev_b == arc) | (tn[np.maximum(prev_b, 0)] == fn[arc]))).astype(float)
    f["next_rule_same"] = (nv & (next_b == arc)).astype(float)
    f["next_rule_adjacent"] = (nv & ((next_b == arc) | (fn[np.maximum(next_b, 0)] == tn[arc]))).astype(float)

    out = cands.copy()
    for c in FEATURE_COLUMNS:
        out[c] = np.asarray(f[c], dtype=np.float32)
    return out


def write_feature_list(path=None) -> str:
    """artifacts/feature_list.json: the ordered model inputs and their groups."""
    path = repo_path(path or "artifacts/feature_list.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"features": FEATURE_COLUMNS, "groups": FEATURE_GROUPS,
                                "meta_columns": META_COLUMNS, "label_column": "label"}, indent=2))
    return str(path)
