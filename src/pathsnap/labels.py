"""Per-observation training labels derived from the reference route.

The reference ``.route`` is an ordered list of arcs, not one label per GPS
point. Each observation is aligned to one route position with a monotonic
dynamic programme (DTW-like):

* state  = route position j (0 .. L-1); the label is ``route[j]``
* cost   = distance from the observation to route arc j in metres, capped at
  ``cap_m`` so a single outlier cannot drag the alignment
* moves  = stay on j, advance to j+1, or jump from k to j > k+1 paying
  ``skip_penalty_m`` per skipped route arc. Across a time gap longer than
  ``gap_s`` the skip penalty is waived (the vehicle may have covered many
  arcs while unobserved).
* heading: when the vehicle is clearly moving, ``heading_weight_m * (1 - cos)/2``
  is added, where cos is between the GPS heading and the arc direction. This
  separates out-and-back legs over the same (two-way) street, whose arcs have
  identical geometry and differ only in direction.
* free start and end: skipping route arcs before the first or after the last
  observation costs ``edge_skip_penalty_m`` per arc (small), because tracks and
  routes need not start/end at exactly the same place.

The route index never decreases, so loops and revisits in the route are
followed in order. Labels are DERIVED: the route itself is not independently
verified ground truth. Used for TRAIN and STUDENT_TEST trips only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geo import TripGeometry


@dataclass(frozen=True)
class AlignParams:
    skip_penalty_m: float = 10.0       # cost per skipped route arc (normal 1 Hz steps)
    edge_skip_penalty_m: float = 0.5   # cost per route arc skipped before the first / after the last point
    gap_s: float = 30.0                # dt above which skipping is free
    cap_m: float = 100.0               # per-point cost cap (robust to outliers)
    outlier_m: float = 50.0            # flag points farther than this from their aligned arc
    heading_weight_m: float = 10.0     # extra cost (m) for driving exactly against an arc's direction
    heading_min_move_m: float = 5.0    # heading is only trusted if the +/-1-step displacement exceeds this


def monotonic_align(cost: np.ndarray, skip_pen: np.ndarray, edge_pen: float = 0.0) -> np.ndarray:
    """Minimum-cost monotonic alignment of n observations to L ordered states.

    ``cost``     (n, L) per-observation, per-state cost
    ``skip_pen`` (n,) penalty per skipped state when moving from obs i-1 to i
                 (``skip_pen[0]`` is ignored)
    Returns the state index for every observation (non-decreasing).

    Transition i-1 -> i from state k to state j (j >= k) costs
    ``skip_pen[i] * max(0, j - k - 1)``. The jump term is computed with a
    running prefix minimum, so each row is O(L) and the whole DP is O(n L).
    """
    n, L = cost.shape
    idx = np.arange(L, dtype=np.float64)
    back = np.empty((n, L), dtype=np.int32)
    prev = cost[0].astype(np.float64) + edge_pen * idx
    back[0] = np.arange(L)
    ar = np.arange(L, dtype=np.int32)
    for i in range(1, n):
        p = skip_pen[i]
        # option A: stay (k = j)  /  option B: advance by one (k = j-1)
        best = prev.copy()
        arg = ar.copy()
        if L > 1:
            adv = prev[:-1]
            better = adv < best[1:]
            best[1:] = np.where(better, adv, best[1:])
            arg[1:] = np.where(better, ar[:-1], arg[1:])
        # option C: jump k <= j-2, cost prev[k] + p*(j-k-1) = (prev[k] - p*k) + p*(j-1)
        if L > 2:
            g = prev - p * idx
            run_arg = _prefix_argmin(g)                 # argmin of g[0..m]
            k = run_arg[: L - 2]                        # for j = 2..L-1, best k in 0..j-2
            jump = g[k] + p * (idx[2:] - 1.0)
            better = jump < best[2:]
            best[2:] = np.where(better, jump, best[2:])
            arg[2:] = np.where(better, k.astype(np.int32), arg[2:])
        prev = best + cost[i]
        back[i] = arg
    # free end: skipping trailing route arcs costs edge_pen each
    final = prev + edge_pen * (L - 1 - idx)
    out = np.empty(n, dtype=np.int64)
    out[-1] = int(np.argmin(final))
    for i in range(n - 1, 0, -1):
        out[i - 1] = back[i, out[i]]
    return out


def _prefix_argmin(g: np.ndarray) -> np.ndarray:
    run_min = np.minimum.accumulate(g)
    # index where the running minimum was last updated (first occurrence of each new minimum)
    is_new = np.r_[True, g[1:] < run_min[:-1]]
    pos = np.where(is_new, np.arange(len(g)), 0)
    return np.maximum.accumulate(pos)


def align_route_to_observations(geom: TripGeometry, route: np.ndarray,
                                params: AlignParams = AlignParams()) -> pd.DataFrame:
    """Align every observation of a trip to one arc of its reference route.

    Returns one row per observation: ``obs_idx, route_pos, arc_id, dist_m,
    is_outlier, after_gap``.
    """
    route = np.asarray(route, dtype=np.int64)
    D = geom.distance_matrix(route)                      # (n_obs, L) metres, float32
    cost = np.minimum(D, params.cap_m)
    if params.heading_weight_m > 0 and geom.n_obs > 1:
        cost = cost + params.heading_weight_m * heading_mismatch(geom, route, params.heading_min_move_m)
    dt = np.r_[0.0, np.diff(geom.t)]
    skip = np.where(dt > params.gap_s, 0.0, params.skip_penalty_m)
    pos = monotonic_align(cost, skip, params.edge_skip_penalty_m)
    dist = D[np.arange(len(pos)), pos].astype(float)
    return pd.DataFrame({
        "obs_idx": np.arange(len(pos)),
        "route_pos": pos,
        "arc_id": route[pos],
        "dist_m": dist,
        "is_outlier": dist > params.outlier_m,
        "after_gap": dt > params.gap_s,
    })


def obs_heading(geom: TripGeometry, min_move_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit GPS heading per observation from the central difference (i-1 -> i+1).

    Returns ``(ux, uy, valid)``; ``valid`` is False where the displacement is
    below ``min_move_m`` (stationary / jitter), where heading is meaningless.
    """
    x, y = geom.obs_x, geom.obs_y
    ip = np.r_[0, np.arange(len(x) - 1)]
    inx = np.r_[np.arange(1, len(x)), len(x) - 1]
    dx, dy = x[inx] - x[ip], y[inx] - y[ip]
    n = np.hypot(dx, dy)
    valid = n >= min_move_m
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(valid, dx / n, 0.0), np.where(valid, dy / n, 0.0), valid


def heading_mismatch(geom: TripGeometry, arc_ids: np.ndarray, min_move_m: float) -> np.ndarray:
    """(n_obs, len(arc_ids)) in [0, 1]: 0 = same direction, 1 = opposite; 0 where heading is unknown."""
    ux, uy, valid = obs_heading(geom, min_move_m)
    a = np.asarray(arc_ids)
    vx, vy = geom.bx[a] - geom.ax[a], geom.by[a] - geom.ay[a]
    ln = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        vx, vy = np.where(ln > 0, vx / ln, 0.0), np.where(ln > 0, vy / ln, 0.0)
    cos = ux[:, None] * vx[None, :] + uy[:, None] * vy[None, :]
    mm = (1.0 - cos) / 2.0
    mm[~valid] = 0.0
    mm[:, ln == 0] = 0.0
    return mm.astype(np.float32)


def label_quality(trip_id: str, partition: str, aligned: pd.DataFrame, route_len: int) -> dict:
    d = aligned["dist_m"].to_numpy()
    return {
        "partition": partition, "trip_id": trip_id, "n_obs": len(d), "route_len": route_len,
        "dist_median_m": float(np.median(d)), "dist_p95_m": float(np.percentile(d, 95)),
        "dist_max_m": float(d.max()),
        "frac_gt_50m": float(np.mean(d > 50)), "frac_gt_100m": float(np.mean(d > 100)),
        "route_coverage": float(len(np.unique(aligned["route_pos"])) / route_len),
        "first_route_pos": int(aligned["route_pos"].iloc[0]),
        "last_route_pos": int(aligned["route_pos"].iloc[-1]),
    }
