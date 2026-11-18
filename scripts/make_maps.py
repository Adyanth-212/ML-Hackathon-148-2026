"""Folium EDA maps: raw GPS (blue), nearby road network (grey), reference route (green).

    python scripts/make_maps.py                 # default showcase trips
    python scripts/make_maps.py 00000024 ...    # any train / student_test trip ids

Trips are chosen to cover the cases found in EDA (see docs/data_quality_log.md).
Only arcs with an end node within 200 m of the track are drawn so the HTML stays small.
Writes maps/eda_<partition>_<trip_id>.html.
"""
from __future__ import annotations

import sys

import numpy as np

from pathsnap.config import repo_path
from pathsnap.io import discover_trips, load_trip
from pathsnap.viz import (GPS_COLOR, NETWORK_COLOR, ROUTE_COLOR, add_gps_layer, add_legend, add_network_layer,
                          add_route_layer, arcs_near_track, base_map, fit_to)

import folium

SHOWCASE = {
    "00000072": ("train", "normal: 1 Hz, no gaps, no jumps, connected route"),
    "00000078": ("train", "gappy: 124 gaps > 5 s, largest 52,810 s"),
    "00000032": ("train", "fast: 358 steps > 60 m/s (sustained, motorway)"),
    "00000013": ("train", "largest network: 307,392 arcs"),
    "00000033": ("train", "route covers ~13% of the GPS path length"),
    "00000056": ("student_test", "student_test: longest student_test track"),
}


def make_map(trip_id: str, partition: str, note: str = "") -> str:
    trip = load_trip(discover_trips(partition)[trip_id], with_route=True)
    lon, lat = trip.track.lon.to_numpy(), trip.track.lat.to_numpy()
    m = base_map(float(np.median(lat)), float(np.median(lon)))
    arc_ids = arcs_near_track(trip.nodes, trip.arcs, lon, lat, corridor_m=200)
    add_network_layer(m, trip.nodes, trip.arcs, arc_ids)
    add_route_layer(m, trip.nodes, trip.arcs, trip.route, name=f"Reference route ({len(trip.route)} arcs)")
    add_gps_layer(m, trip.track, name=f"GPS observations")
    folium.LayerControl(collapsed=False).add_to(m)
    add_legend(m, [("Raw GPS", GPS_COLOR), ("Road network (within 200 m of track)", NETWORK_COLOR),
                   ("Reference route", ROUTE_COLOR)],
               title=f"{partition} trip {trip_id}<br><span style='font-weight:normal'>{note}</span>")
    fit_to(m, lon, lat)
    out = repo_path(f"maps/eda_{partition}_{trip_id}.html")
    out.parent.mkdir(exist_ok=True)
    m.save(str(out))
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, {len(arc_ids)} arcs drawn)")
    return str(out)


def main(argv: list[str]) -> int:
    if argv:
        found = {**{k: "train" for k in discover_trips("train")}, **{k: "student_test" for k in discover_trips("student_test")}}
        todo = {t: (found[t], "") for t in argv}
    else:
        todo = SHOWCASE
    for tid, (part, note) in todo.items():
        make_map(tid, part, note)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
