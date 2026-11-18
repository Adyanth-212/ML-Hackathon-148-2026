"""Tests for loaders, the leakage guard, geometry and network statistics (toy data)."""
from pathlib import Path

import numpy as np
import pytest

from pathsnap.geo import LocalProjector, angle_diff_deg, bearing_deg, haversine_m, point_segment_distance, utm_epsg
from pathsnap.io import LeakageError, discover_trips, load_trip
from pathsnap.network import network_summary, route_summary, successor_arcs

TOY = Path(__file__).parent / "fixtures" / "toy"


def toy_trip(with_route=False):
    files = discover_trips("train", root=TOY)["toy01"]
    return load_trip(files, with_route=with_route)


def test_discover_and_load():
    trip = toy_trip()
    assert list(trip.track.columns) == ["lon", "lat", "t"]
    assert len(trip.track) == 5 and len(trip.nodes) == 5 and len(trip.arcs) == 5
    assert trip.route is None  # never loaded unless asked for
    assert toy_trip(with_route=True).route.tolist() == [0, 2]


def test_evaluator_guard(monkeypatch):
    monkeypatch.delenv("PATHSNAP_MODE", raising=False)
    with pytest.raises(LeakageError):
        discover_trips("evaluator", root=TOY)
    monkeypatch.setenv("PATHSNAP_MODE", "serve")
    assert "toy01" in discover_trips("evaluator", root=TOY)


def test_utm_zone():
    assert utm_epsg(77.59, 12.93) == 32643   # Bengaluru, UTM 43N
    assert utm_epsg(-58.4, -34.6) == 32721   # Buenos Aires, UTM 21S


def test_projection_matches_haversine():
    proj = LocalProjector(77.59, 12.93)
    x, y = proj.forward([77.590, 77.591], [12.930, 12.930])
    planar = float(np.hypot(x[1] - x[0], y[1] - y[0]))
    great_circle = float(haversine_m(77.590, 12.930, 77.591, 12.930))
    # haversine assumes a sphere; UTM uses the WGS84 ellipsoid, so allow ~0.5%
    assert abs(planar - great_circle) / great_circle < 5e-3
    lon, lat = proj.inverse(x, y)
    assert np.allclose(lon, [77.590, 77.591]) and np.allclose(lat, [12.930, 12.930])


def test_point_segment_distance():
    d, t = point_segment_distance([5, -3, 15], [4, 0, 0], 0, 0, 10, 0)
    assert np.allclose(d, [4, 3, 5]) and np.allclose(t, [0.5, -0.3, 1.5])


def test_bearings():
    assert bearing_deg(0, 0, 0, 1) == pytest.approx(0)
    assert bearing_deg(0, 0, 1, 0) == pytest.approx(90)
    assert angle_diff_deg(350, 10) == pytest.approx(20)


def test_network_and_route_summary():
    trip = toy_trip(with_route=True)
    s = network_summary(trip.nodes, trip.arcs)
    assert s["n_nodes"] == 5 and s["n_arcs"] == 5
    assert s["isolated_nodes"] == 1          # node 4 has no arcs
    assert s["n_weak_components"] == 2
    assert s["frac_arcs_with_reverse"] == pytest.approx(2 / 5)
    r = route_summary(trip.route, trip.arcs)
    assert r["route_consecutive_connected_frac"] == 1.0  # arc 0 (0->1) then arc 2 (1->2)
    assert successor_arcs(trip.arcs)[0] == [1, 2]
