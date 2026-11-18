"""Geometry helpers.

Longitude/latitude are angular, so distances in degrees are meaningless. Each
trip is projected into the UTM zone of its centroid (a conformal projection
with < 0.1% scale error inside a zone), after which plain Euclidean geometry
in metres is valid at the scale of a single trip.
"""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer

EARTH_RADIUS_M = 6_371_008.8


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing (lon, lat)."""
    zone = int((lon + 180.0) // 6.0) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


class LocalProjector:
    """Projects lon/lat to metres in the UTM zone of a reference point."""

    def __init__(self, ref_lon: float, ref_lat: float):
        self.epsg = utm_epsg(ref_lon, ref_lat)
        self._fwd = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(self.epsg), always_xy=True)
        self._inv = Transformer.from_crs(CRS.from_epsg(self.epsg), CRS.from_epsg(4326), always_xy=True)

    @classmethod
    def for_points(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalProjector":
        return cls(float(np.nanmedian(lon)), float(np.nanmedian(lat)))

    def forward(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def inverse(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = self._inv.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in metres (vectorised)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def point_segment_distance(px, py, ax, ay, bx, by) -> tuple[np.ndarray, np.ndarray]:
    """Distance from points P to segments AB in projected metres (vectorised).

    Returns ``(distance, t)`` where ``t`` is the *unclipped* projection fraction
    along AB (t < 0 or t > 1 means the closest point is an endpoint).
    """
    px, py, ax, ay, bx, by = map(lambda v: np.asarray(v, float), (px, py, ax, ay, bx, by))
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(seg_len2 > 0, ((px - ax) * dx + (py - ay) * dy) / seg_len2, 0.0)
    tc = np.clip(t, 0.0, 1.0)
    cx, cy = ax + tc * dx, ay + tc * dy
    return np.hypot(px - cx, py - cy), t


def bearing_deg(ax, ay, bx, by) -> np.ndarray:
    """Compass bearing of A->B in projected coords: 0 = north (+y), 90 = east (+x)."""
    return np.degrees(np.arctan2(np.asarray(bx) - ax, np.asarray(by) - ay)) % 360.0


def angle_diff_deg(a, b) -> np.ndarray:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)
