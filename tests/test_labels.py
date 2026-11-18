"""Tests for the monotonic route alignment (labels.py)."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.geo import TripGeometry
from pathsnap.io import discover_trips, load_trip
from pathsnap.labels import AlignParams, align_route_to_observations, monotonic_align

TOY = Path(__file__).parent / "fixtures" / "toy"


def _path_cost(cost, skip, edge, a):
    n, L = cost.shape
    c = cost[np.arange(n), a].sum() + edge * a[0] + edge * (L - 1 - a[-1])
    return c + sum(skip[i] * max(0, a[i] - a[i - 1] - 1) for i in range(1, n))


def test_dp_matches_brute_force():
    rng = np.random.default_rng(148)
    for _ in range(150):
        n, L = int(rng.integers(1, 6)), int(rng.integers(1, 7))
        cost, skip, edge = rng.random((n, L)) * 10, rng.random(n) * 3, float(rng.random())
        a = monotonic_align(cost, skip, edge)
        assert np.all(np.diff(a) >= 0)
        best = min(_path_cost(cost, skip, edge, np.array(s))
                   for s in itertools.combinations_with_replacement(range(L), n))
        assert abs(_path_cost(cost, skip, edge, a) - best) < 1e-6


def test_toy_trip_alignment():
    trip = load_trip(discover_trips("train", root=TOY)["toy01"], with_route=True)
    al = align_route_to_observations(TripGeometry.from_trip(trip), trip.route)
    # points 0-1 run along arc 0 (node 0 -> 1, eastwards), points 3-4 along arc 2 (node 1 -> 2, northwards);
    # point 2 sits at the corner (node 1) and is closer to arc 2, so either label is acceptable there
    a = al.arc_id.tolist()
    assert a[:2] == [0, 0] and a[3:] == [2, 2] and a[2] in (0, 2)
    assert al.dist_m.max() < 5 and not al.is_outlier.any()


def _square_geom(track_xy, t=None):
    """Hand-built geometry in degrees near the equator: nodes on a 0.001 deg (~111 m) square."""
    nodes = pd.DataFrame({"lon": [0, 0.001, 0.001, 0], "lat": [0, 0, 0.001, 0.001]})
    arcs = pd.DataFrame({"from_node": [0, 1, 2, 3, 1], "to_node": [1, 2, 3, 0, 0]})
    lon, lat = np.asarray(track_xy, float).T
    track = pd.DataFrame({"lon": lon, "lat": lat, "t": t if t is not None else np.arange(len(lon), dtype=float)})
    return TripGeometry(track, nodes, arcs)


def test_route_revisit_is_followed_in_order():
    # drive 0->1 (arc 0), back 1->0 (arc 4), then 0->1 again (arc 0): the three legs have identical
    # geometry, so only route order + GPS heading can separate them
    pts = [(0.0002, 0), (0.0008, 0), (0.0007, 0.00001), (0.0003, 0.00001), (0.0004, 0), (0.0008, 0)]
    geom = _square_geom(pts)
    al = align_route_to_observations(geom, np.array([0, 4, 0]))
    assert al.route_pos.tolist() == [0, 0, 1, 1, 2, 2]


def test_outlier_flagged_but_labelled():
    pts = [(0.0001, 0), (0.0003, 0), (0.0005, 0.003), (0.0007, 0), (0.0009, 0)]  # point 2 is ~330 m off
    al = align_route_to_observations(_square_geom(pts), np.array([0, 1]))
    assert al.is_outlier.tolist() == [False, False, True, False, False]
    assert al.arc_id.tolist()[:2] == [0, 0] and al.arc_id.iloc[-1] == 0  # outlier does not drag the rest


def test_gap_makes_skipping_free():
    # observations on arc 0, then a long gap, then on arc 2: arc 1 is skipped
    pts = [(0.0002, 0), (0.0005, 0), (0.0006, 0.001), (0.0003, 0.001)]
    al = align_route_to_observations(_square_geom(pts, t=np.array([0, 1, 500, 501.0])), np.array([0, 1, 2]))
    assert al.route_pos.tolist() == [0, 0, 2, 2] and al.after_gap.tolist() == [False, False, True, False]
