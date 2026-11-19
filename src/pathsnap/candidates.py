"""Candidate generation: nearby arcs for every GPS observation.

For each observation, the arcs within ``radius_m`` metres (at most the
``max_candidates`` nearest) are found with a shapely ``STRtree`` over the
trip's projected arc segments. Observations with no arc in range get one retry
at ``fallback_radius_m``; if that also finds nothing the observation keeps zero
rows and is reported in ``no_candidate_obs``.

Ties: two-way streets are two arcs with identical geometry, so equal distances
are common. Candidates are ordered by ``(dist_m, arc_id)`` everywhere (here and
in the miss-rate sweep) so the K cut is deterministic.

Output = the candidate-table contract (docs/REFERENCE.md section 3):
``trip_id, obs_idx, arc_id, dist_m, proj_frac[, label]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from .config import load_config
from .geo import TripGeometry

CONTRACT_COLUMNS = ["trip_id", "obs_idx", "arc_id", "dist_m", "proj_frac", "label"]


@dataclass
class ArcIndex:
    """STRtree over a trip's arc segments (projected metres)."""

    geom: TripGeometry
    tree: STRtree = field(init=False)

    def __post_init__(self):
        g = self.geom
        coords = np.stack([np.c_[g.ax, g.ay], np.c_[g.bx, g.by]], axis=1)  # (n_arcs, 2, 2)
        self.tree = STRtree(shapely.linestrings(coords))
        self._points = shapely.points(np.c_[g.obs_x, g.obs_y])

    def pairs_within(self, radius_m: float, obs_idx: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
        """All (obs, arc) pairs closer than ``radius_m``, with exact distance and projection fraction,
        sorted by (obs, dist, arc_id)."""
        pts = self._points if obs_idx is None else self._points[obs_idx]
        qi, arc = self.tree.query(pts, predicate="dwithin", distance=radius_m)
        obs = qi if obs_idx is None else np.asarray(obs_idx)[qi]
        d, t = self.geom.obs_arc_distance(obs, arc)
        keep = d <= radius_m
        obs, arc, d, t = obs[keep], arc[keep], d[keep], t[keep]
        order = np.lexsort((arc, d, obs))
        return obs[order], arc[order], d[order], t[order]


def _rank_within_obs(obs_sorted: np.ndarray) -> np.ndarray:
    """0-based position of each row inside its observation's block (input sorted by obs)."""
    if len(obs_sorted) == 0:
        return np.zeros(0, dtype=np.int64)
    start = np.r_[0, np.flatnonzero(np.diff(obs_sorted)) + 1]
    block_start = np.repeat(start, np.diff(np.r_[start, len(obs_sorted)]))
    return np.arange(len(obs_sorted)) - block_start


def generate_candidates(geom: TripGeometry, trip_id: str, radius_m: float | None = None,
                        max_candidates: int | None = None, fallback_radius_m: float | None = None,
                        index: ArcIndex | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Candidate rows for one trip (no labels) and the observations left without candidates.

    Defaults come from ``configs/candidates.yaml``. Needs no reference route, so
    it is used unchanged at inference time.
    """
    cfg = load_config("candidates")
    R = float(radius_m if radius_m is not None else cfg["radius_m"])
    K = int(max_candidates if max_candidates is not None else cfg["max_candidates"])
    FR = float(fallback_radius_m if fallback_radius_m is not None else cfg["fallback_radius_m"])
    index = index or ArcIndex(geom)

    obs, arc, d, t = index.pairs_within(R)
    r = _rank_within_obs(obs)
    keep = r < K
    obs, arc, d, t = obs[keep], arc[keep], d[keep], t[keep]

    missing = np.setdiff1d(np.arange(geom.n_obs), obs)
    if len(missing) and FR > R:
        o2, a2, d2, t2 = index.pairs_within(FR, missing)
        k2 = _rank_within_obs(o2) < K
        obs, arc, d, t = (np.r_[obs, o2[k2]], np.r_[arc, a2[k2]], np.r_[d, d2[k2]], np.r_[t, t2[k2]])
        order = np.lexsort((arc, d, obs))
        obs, arc, d, t = obs[order], arc[order], d[order], t[order]
        missing = np.setdiff1d(missing, o2)

    df = pd.DataFrame({
        "trip_id": pd.Series([trip_id] * len(obs), dtype="string"),
        "obs_idx": obs.astype(np.int32),
        "arc_id": arc.astype(np.int32),
        "dist_m": d.astype(np.float32),
        "proj_frac": t.astype(np.float32),
    })
    return df, missing


def attach_labels(cands: pd.DataFrame, aligned_arc: np.ndarray) -> pd.DataFrame:
    """label = 1 where the candidate is the observation's aligned route arc, else 0 (int8)."""
    cands = cands.copy()
    cands["label"] = (cands["arc_id"].to_numpy() == aligned_arc[cands["obs_idx"].to_numpy()]).astype(np.int8)
    return cands


def true_arc_rank(index: ArcIndex, aligned_arc: np.ndarray, aligned_dist: np.ndarray,
                  max_radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    """For every observation: the 0-based rank of its aligned arc in (dist, arc_id) order, and
    the number of arcs within ``max_radius_m``. Rank is -1 if the aligned arc is farther than
    ``max_radius_m``. Used for the R/K miss-rate sweep (one query per trip covers every R <= max)."""
    n = index.geom.n_obs
    obs, arc, d, _ = index.pairs_within(max_radius_m)
    r = _rank_within_obs(obs)
    hit = arc == aligned_arc[obs]
    rank = np.full(n, -1, dtype=np.int64)
    rank[obs[hit]] = r[hit]
    return rank, obs, d


def miss_rate_grid(rank: np.ndarray, true_dist: np.ndarray, obs: np.ndarray, d: np.ndarray, use: np.ndarray,
                   radii: list[float], ks: list[int]) -> list[dict]:
    """Miss count and candidate-set size for every (R, K) on one trip, over observations where ``use``
    is True (sums, so trips can be pooled). Ignores the fallback radius (it only affects empty sets)."""
    rows = []
    n_obs = len(rank)
    for R in radii:
        within = d <= R
        cnt = np.bincount(obs[within], minlength=n_obs)[use]
        for K in ks:
            hit = ((rank >= 0) & (rank < K) & (true_dist <= R))[use]
            rows.append({"radius_m": R, "max_candidates": K, "n_obs": int(use.sum()), "n_miss": int((~hit).sum()),
                         "sum_cands": int(np.minimum(cnt, K).sum()), "n_empty": int((cnt == 0).sum())})
    return rows
