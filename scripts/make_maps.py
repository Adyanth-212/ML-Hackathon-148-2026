"""Folium EDA maps: raw GPS (blue), nearby road network (grey), reference route (green).

    python scripts/make_maps.py                 # default showcase trips
    python scripts/make_maps.py 00000024 ...    # any train / student_test trip ids
    python scripts/make_maps.py --labels        # label-check maps (needs scripts/build_labels.py first)
    python scripts/make_maps.py --predicted     # report maps: reference vs predicted route (student_test)

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
    if argv[:1] == ["--predicted"]:
        from pathsnap.pipeline import load_artifacts
        art = load_artifacts(use_decoder=True)
        for tid, note in PREDICTED_SHOWCASE.items():
            make_predicted_map(tid, art, note)
        return 0
    if argv[:1] == ["--labels"]:
        for tid, (part, note) in LABEL_SHOWCASE.items():
            make_label_map(tid, part, note)
        return 0
    if argv:
        found = {**{k: "train" for k in discover_trips("train")}, **{k: "student_test" for k in discover_trips("student_test")}}
        todo = {t: (found[t], "") for t in argv}
    else:
        todo = SHOWCASE
    for tid, (part, note) in todo.items():
        make_map(tid, part, note)
    return 0



LABEL_SHOWCASE = {
    "00000072": ("train", "normal"),
    "00000024": ("train", "speed outliers"),
    "00000078": ("train", "gappy"),
    "00000002": ("train", "route break + far points"),
    "00000054": ("train", "worst: route covers part of the track"),
    "00000033": ("train", "second worst"),
    "00000056": ("student_test", "student_test"),
}


def make_label_map(trip_id: str, partition: str, note: str = "") -> str:
    """GPS points coloured by their aligned route arc (labels.py), outliers in black, plus the route."""
    import pandas as pd

    from pathsnap.viz import distinct_colors

    trip = load_trip(discover_trips(partition)[trip_id], with_route=True)
    al = pd.read_parquet(repo_path(f"data/processed/labels/{partition}/{trip_id}.parquet"))
    lon, lat = trip.track.lon.to_numpy(), trip.track.lat.to_numpy()
    m = base_map(float(np.median(lat)), float(np.median(lon)))
    add_network_layer(m, trip.nodes, trip.arcs, arcs_near_track(trip.nodes, trip.arcs, lon, lat, 150), show=False)
    pal = distinct_colors(12)
    rp = al.route_pos.to_numpy()
    # colour the route arcs with the same cyclic palette so each point can be checked against its arc
    fg = folium.FeatureGroup(name="Reference route (coloured by position)").add_to(m)
    pos = np.arange(len(trip.route))
    for k in range(12):
        sel = pos[pos % 12 == k]
        if len(sel):
            folium.PolyLine(arc_segments(trip, trip.route[sel]), color=pal[k], weight=6, opacity=.55).add_to(fg)
    colors = [("#000000" if o else pal[p % 12]) for p, o in zip(rp, al.is_outlier)]
    tips = [f"obs {i} -> route pos {p}, arc {a}, {d:.1f} m" for i, p, a, d in
            zip(al.obs_idx, rp, al.arc_id, al.dist_m)]
    add_gps_layer(m, trip.track, name="GPS coloured by aligned arc (black = > 50 m)", colors=colors, tooltips=tips,
                  radius=3)
    add_route_layer(m, trip.nodes, trip.arcs, trip.route, name="Reference route (plain green)", show=False)
    folium.LayerControl(collapsed=False).add_to(m)
    add_legend(m, [("Point colour = colour of its aligned route arc", "#9467bd"), ("Outlier > 50 m", "#000000"),
                   ("Reference route (toggle)", ROUTE_COLOR), ("Road network (toggle)", NETWORK_COLOR)],
               title=f"Label check: {partition} {trip_id}<br><span style='font-weight:normal'>{note}; "
                     f"median {al.dist_m.median():.1f} m, {al.is_outlier.mean():.1%} > 50 m</span>")
    fit_to(m, lon, lat)
    out = repo_path(f"maps/labels_{partition}_{trip_id}.html")
    m.save(str(out))
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return str(out)


def arc_segments(trip, arc_ids):
    from pathsnap.viz import arc_segments_lonlat
    return arc_segments_lonlat(trip.nodes, trip.arcs, np.asarray(arc_ids))


PREDICTED_SHOWCASE = {  # student_test trips, picked from experiments/results/pipeline_student_test.csv
    "00000007": "best trip",
    "00000040": "typical trip",
    "00000056": "longest trip (3,958 points)",
    "00000077": "worst trip",
}


def make_predicted_map(trip_id: str, artifacts: dict, note: str = "", partition: str = "student_test") -> str:
    """Report map: raw GPS, nearby network, reference route (green) and predicted route (red).

    Offline only, on student_test: the reference route is drawn here for the report and is never
    sent to the UI. GPS points whose predicted arc differs from the derived label are orange.
    """
    import pandas as pd

    from pathsnap.pipeline import predict_trip
    from pathsnap.viz import OUTLIER_COLOR, PRED_COLOR

    trip = load_trip(discover_trips(partition)[trip_id], with_route=False)
    res = predict_trip(trip, artifacts)  # no route in the prediction
    route = load_trip(discover_trips(partition)[trip_id], with_route=True).route
    lab = pd.read_parquet(repo_path(f"data/processed/labels/{partition}/{trip_id}.parquet"))
    pred = np.asarray(res["pred_arcs"])
    ok = ~lab.is_outlier.to_numpy()
    right = pred == lab.arc_id.to_numpy()
    top1 = float(right[ok].mean())

    lon, lat = trip.track.lon.to_numpy(), trip.track.lat.to_numpy()
    m = base_map(float(np.median(lat)), float(np.median(lon)))
    add_network_layer(m, trip.nodes, trip.arcs, arcs_near_track(trip.nodes, trip.arcs, lon, lat, 200), show=True)
    add_route_layer(m, trip.nodes, trip.arcs, route, name=f"Reference route ({len(route)} arcs)", weight=8)
    fg = folium.FeatureGroup(name="Predicted route (classifier + HMM)").add_to(m)
    folium.PolyLine([(la, lo) for lo, la in res["route_lonlat"]], color=PRED_COLOR, weight=3, opacity=0.9).add_to(fg)
    colors = [GPS_COLOR if r else OUTLIER_COLOR for r in right]
    tips = [f"obs {i}: predicted arc {p}, label arc {a}" for i, (p, a) in enumerate(zip(pred, lab.arc_id))]
    add_gps_layer(m, trip.track, name="GPS (orange = predicted arc differs from label)", colors=colors,
                  tooltips=tips, radius=2.5)
    folium.LayerControl(collapsed=False).add_to(m)
    add_legend(m, [("Reference route", ROUTE_COLOR), ("Predicted route", PRED_COLOR),
                   ("GPS, predicted arc = label", GPS_COLOR), ("GPS, predicted arc differs", OUTLIER_COLOR),
                   ("Road network (within 200 m)", NETWORK_COLOR)],
               title=f"{partition} trip {trip_id}: {res['method']['scorer']}<br><span style='font-weight:normal'>"
                     f"{note}; Top-1 {top1:.3f} (reliable labels)</span>")
    fit_to(m, lon, lat)
    out = repo_path(f"maps/predicted_{partition}_{trip_id}.html")
    m.save(str(out))
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, Top-1 {top1:.3f})")
    return str(out)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
