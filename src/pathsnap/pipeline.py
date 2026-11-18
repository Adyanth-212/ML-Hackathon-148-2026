"""Inference pipeline used by the API.

The first deployment milestone deliberately uses a nearest-directed-arc baseline.
It accepts no reference route, so evaluator labels cannot leak into prediction.
The returned payload is compatible with the later classifier/HMM integration.
"""
from __future__ import annotations

import time

import numpy as np

from .geo import LocalProjector, point_segment_distance


def _deduplicate(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        value = int(value)
        if not out or out[-1] != value:
            out.append(value)
    return out


def predict_trip(trip, artifacts=None) -> dict:
    """Predict one arc per GPS observation using the nearest-arc baseline.

    Parameters
    ----------
    trip:
        A :class:`pathsnap.io.Trip` with track, nodes and directed arcs. Its
        ``route`` attribute is intentionally never read.
    artifacts:
        Reserved for the trained classifier/HMM bundle. ``None`` or a bundle
        with ``mode='nearest_arc'`` selects the milestone baseline.

    Returns
    -------
    dict
        JSON-safe coordinates, per-observation arc ids, a de-duplicated route,
        a lightweight nearby-network overlay and stage timings.
    """
    started = time.perf_counter()
    if trip.track is None or trip.track.empty:
        raise ValueError(f"Trip {trip.trip_id} has no GPS observations")
    if trip.nodes is None or trip.arcs is None or trip.arcs.empty:
        raise ValueError(f"Trip {trip.trip_id} is missing its road network")

    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    if arcs[["from_node", "to_node"]].to_numpy().min() < 0 or arcs[["from_node", "to_node"]].to_numpy().max() >= len(nodes):
        raise ValueError(f"Trip {trip.trip_id} contains an arc with an invalid node reference")

    prep_start = time.perf_counter()
    projector = LocalProjector.for_points(track["lon"].to_numpy(), track["lat"].to_numpy())
    gps_x, gps_y = projector.forward(track["lon"].to_numpy(), track["lat"].to_numpy())
    node_x, node_y = projector.forward(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    arc_pairs = arcs[["from_node", "to_node"]].to_numpy(dtype=int)
    ax, ay = node_x[arc_pairs[:, 0]], node_y[arc_pairs[:, 0]]
    bx, by = node_x[arc_pairs[:, 1]], node_y[arc_pairs[:, 1]]
    prep_ms = (time.perf_counter() - prep_start) * 1000

    infer_start = time.perf_counter()
    # One row per observation, one column per directed arc. This is the
    # baseline only; later model artifacts can replace this selection step.
    distances, _ = point_segment_distance(
        gps_x[:, None], gps_y[:, None], ax[None, :], ay[None, :], bx[None, :], by[None, :]
    )
    pred_arcs = np.argmin(distances, axis=1).astype(int).tolist()
    nearest_dist = distances[np.arange(len(track)), pred_arcs].astype(float).tolist()
    route_arc_ids = _deduplicate(pred_arcs)

    # Construct the route from the directed endpoints of the selected arcs.
    route_node_ids: list[int] = []
    for arc_id in route_arc_ids:
        u, v = arc_pairs[arc_id]
        if not route_node_ids:
            route_node_ids.extend([int(u), int(v)])
        elif route_node_ids[-1] == int(u):
            route_node_ids.append(int(v))
        else:
            # Preserve a visible discontinuity instead of silently inventing
            # a shortest-path connection before Sudhe's HMM is integrated.
            route_node_ids.extend([int(u), int(v)])
    route_lon, route_lat = projector.inverse(node_x[route_node_ids], node_y[route_node_ids]) if route_node_ids else ([], [])
    route_lonlat = [[float(lon), float(lat)] for lon, lat in zip(route_lon, route_lat)]

    # Return a bounded local network overlay for the Folium map. This avoids
    # sending an entire large road graph to the browser.
    margin_m = 100.0
    min_x, max_x = float(np.min(gps_x) - margin_m), float(np.max(gps_x) + margin_m)
    min_y, max_y = float(np.min(gps_y) - margin_m), float(np.max(gps_y) + margin_m)
    in_view = (
        (np.maximum(ax, bx) >= min_x) & (np.minimum(ax, bx) <= max_x)
        & (np.maximum(ay, by) >= min_y) & (np.minimum(ay, by) <= max_y)
    )
    nearby_ids = np.flatnonzero(in_view)[:3000]
    network_lonlat = []
    if len(nearby_ids):
        endpoints_x = np.column_stack([ax[nearby_ids], bx[nearby_ids]]).ravel()
        endpoints_y = np.column_stack([ay[nearby_ids], by[nearby_ids]]).ravel()
        endpoints_lon, endpoints_lat = projector.inverse(endpoints_x, endpoints_y)
        coords = np.column_stack([endpoints_lon, endpoints_lat]).reshape(-1, 2, 2)
        network_lonlat = [segment.astype(float).tolist() for segment in coords]

    infer_ms = (time.perf_counter() - infer_start) * 1000
    gps = [[float(lon), float(lat)] for lon, lat in zip(track["lon"], track["lat"])]
    total_ms = (time.perf_counter() - started) * 1000
    return {
        "trip_id": trip.trip_id,
        "gps": gps,
        "pred_arcs": pred_arcs,
        "route_arc_ids": route_arc_ids,
        "route_lonlat": route_lonlat,
        "network_lonlat": network_lonlat,
        "nearest_dist_m": nearest_dist,
        "timings_ms": {
            "preprocess": round(prep_ms, 3),
            "inference": round(infer_ms, 3),
            "total": round(total_ms, 3),
        },
        "model": "nearest_arc_baseline",
    }
