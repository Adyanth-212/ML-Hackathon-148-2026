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


class TripGeometry:
    """Everything about one trip projected once into its local UTM zone.

    Attributes are plain numpy arrays in metres so downstream code (labels,
    candidates, features) never touches degrees:

    * ``obs_x, obs_y``  - GPS observations (track order)
    * ``node_x, node_y`` - network nodes (row = node_id)
    * ``ax, ay, bx, by`` - start/end of every arc (row = arc_id); arcs are
      straight segments between their two nodes (the files carry no shape points)
    """

    def __init__(self, track, nodes, arcs, projector: "LocalProjector | None" = None):
        lon, lat = track["lon"].to_numpy(float), track["lat"].to_numpy(float)
        self.proj = projector or LocalProjector.for_points(lon, lat)
        self.t = track["t"].to_numpy(float)
        self.obs_x, self.obs_y = self.proj.forward(lon, lat)
        self.node_x, self.node_y = self.proj.forward(nodes["lon"].to_numpy(float), nodes["lat"].to_numpy(float))
        self.from_node = arcs["from_node"].to_numpy(np.int64)
        self.to_node = arcs["to_node"].to_numpy(np.int64)
        self.ax, self.ay = self.node_x[self.from_node], self.node_y[self.from_node]
        self.bx, self.by = self.node_x[self.to_node], self.node_y[self.to_node]

    @classmethod
    def from_trip(cls, trip) -> "TripGeometry":
        return cls(trip.track, trip.nodes, trip.arcs)

    @property
    def n_obs(self) -> int:
        return len(self.obs_x)

    @property
    def n_arcs(self) -> int:
        return len(self.ax)

    def arc_lengths(self) -> np.ndarray:
        return np.hypot(self.bx - self.ax, self.by - self.ay)

    def obs_arc_distance(self, obs_idx: np.ndarray, arc_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distance (m) and unclipped projection fraction for paired (obs, arc) indices."""
        return point_segment_distance(self.obs_x[obs_idx], self.obs_y[obs_idx],
                                      self.ax[arc_ids], self.ay[arc_ids], self.bx[arc_ids], self.by[arc_ids])

    def distance_matrix(self, arc_ids: np.ndarray) -> np.ndarray:
        """Dense (n_obs x len(arc_ids)) distance matrix in metres (float32)."""
        a = np.asarray(arc_ids)
        ax, ay, bx, by = (v[a][None, :] for v in (self.ax, self.ay, self.bx, self.by))
        out = np.empty((self.n_obs, len(a)), dtype=np.float32)
        step = max(1, 2_000_000 // max(len(a), 1))  # bound peak memory of the float64 temporaries
        for s in range(0, self.n_obs, step):
            d, _ = point_segment_distance(self.obs_x[s:s + step, None], self.obs_y[s:s + step, None], ax, ay, bx, by)
            out[s:s + step] = d
        return out
