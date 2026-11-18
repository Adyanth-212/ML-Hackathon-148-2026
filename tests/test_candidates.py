"""Tests for candidate generation (candidates.py)."""
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.candidates import (CONTRACT_COLUMNS, ArcIndex, attach_labels, generate_candidates, miss_rate_grid,
                                 true_arc_rank)
from pathsnap.geo import TripGeometry
from pathsnap.io import discover_trips, load_trip

TOY = Path(__file__).parent / "fixtures" / "toy"


def toy_geom():
    trip = load_trip(discover_trips("train", root=TOY)["toy01"], with_route=True)
    return trip, TripGeometry.from_trip(trip)


def test_matches_brute_force():
    trip, g = toy_geom()
    for R, K in ((30, 2), (60, 3), (200, 10)):
        c, _ = generate_candidates(g, "toy01", R, K, fallback_radius_m=0)
        n_arcs = g.n_arcs
        o, a = np.repeat(np.arange(g.n_obs), n_arcs), np.tile(np.arange(n_arcs), g.n_obs)
        d, t = g.obs_arc_distance(o, a)
        bf = pd.DataFrame({"obs_idx": o, "arc_id": a, "dist_m": d, "proj_frac": t})
        bf = bf[bf.dist_m <= R].sort_values(["obs_idx", "dist_m", "arc_id"]).groupby("obs_idx").head(K)
        assert c[["obs_idx", "arc_id"]].values.tolist() == bf[["obs_idx", "arc_id"]].values.tolist()
        assert np.allclose(c.dist_m, bf.dist_m, atol=1e-3) and np.allclose(c.proj_frac, bf.proj_frac, atol=1e-4)


def test_fallback_and_missing():
    _, g = toy_geom()
    c, missing = generate_candidates(g, "toy01", radius_m=0.5, max_candidates=5, fallback_radius_m=0.6)
    assert len(c) == 0 and missing.tolist() == [0, 1, 2, 3, 4]       # nothing within 0.6 m
    c, missing = generate_candidates(g, "toy01", radius_m=0.5, max_candidates=5, fallback_radius_m=50)
    assert len(missing) == 0 and set(c.obs_idx) == {0, 1, 2, 3, 4}   # fallback finds them all


def test_labels_and_contract():
    trip, g = toy_geom()
    c, _ = generate_candidates(g, "toy01", 60, 5, 150)
    aligned = np.array([0, 0, 0, 2, 2])
    c = attach_labels(c, aligned)[CONTRACT_COLUMNS]
    assert list(c.columns) == CONTRACT_COLUMNS
    assert (c.groupby("obs_idx").label.sum() == 1).all()           # exactly one positive per observation here
    assert c.loc[c.label == 1, "arc_id"].tolist() == aligned.tolist()
    # the reverse twin of arc 0 (arc 1) has the same distance but label 0
    o0 = c[c.obs_idx == 0].set_index("arc_id")
    assert np.isclose(o0.dist_m[0], o0.dist_m[1]) and o0.label[1] == 0


def test_miss_rate_grid_consistent_with_generation():
    _, g = toy_geom()
    aligned = np.array([0, 0, 0, 2, 2])
    idx = ArcIndex(g)
    tdist, _ = g.obs_arc_distance(np.arange(5), aligned)
    rank, obs, d = true_arc_rank(idx, aligned, tdist, 200)
    for row in miss_rate_grid(rank, tdist, obs, d, np.ones(5, bool), [2, 60], [1, 5]):
        c, _ = generate_candidates(g, "toy01", row["radius_m"], row["max_candidates"], 0)
        c = attach_labels(c, aligned)
        assert row["n_miss"] == 5 - int(c.label.sum())
        assert row["sum_cands"] == len(c)
