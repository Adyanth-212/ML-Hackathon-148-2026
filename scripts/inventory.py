"""EDA step 1: dataset inventory and data-quality table.

Builds one row per trip for the TRAIN and STUDENT_TEST partitions (never the
evaluator partition) and writes experiments/results/inventory.csv.

    python scripts/inventory.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pathsnap.config import repo_path
from pathsnap.geo import LocalProjector, haversine_m
from pathsnap.io import discover_trips, load_trip
from pathsnap.network import network_summary, route_summary

DEV_PARTITIONS = ("train", "student_test")


def track_summary(track: pd.DataFrame) -> dict:
    lon, lat, t = track["lon"].to_numpy(), track["lat"].to_numpy(), track["t"].to_numpy()
    nan_rows = track.isna().any(axis=1)
    out_of_range = (np.abs(lon) > 180) | (np.abs(lat) > 90)
    zero = (lon == 0) & (lat == 0)
    ok = ~(nan_rows.to_numpy() | out_of_range | zero)
    dt = np.diff(t)
    row = {
        "n_obs": len(track),
        "n_nan_rows": int(nan_rows.sum()),
        "n_out_of_range": int(out_of_range.sum()),
        "n_zero_coords": int(zero.sum()),
        "t_monotonic": bool(np.all(dt > 0)) if len(dt) else True,
        "n_dup_timestamps": int((dt == 0).sum()),
        "n_backwards_time": int((dt < 0).sum()),
        "dt_median_s": float(np.median(dt)) if len(dt) else np.nan,
        "dt_frac_1s": float(np.mean(dt == 1)) if len(dt) else np.nan,
        "dt_max_s": float(dt.max()) if len(dt) else np.nan,
        "n_gaps_gt_5s": int((dt > 5).sum()),
        "duration_s": float(t[-1] - t[0]) if len(t) else np.nan,
        "lon_min": np.nanmin(lon), "lon_max": np.nanmax(lon),
        "lat_min": np.nanmin(lat), "lat_max": np.nanmax(lat),
    }
    if ok.sum() >= 2:
        step = haversine_m(lon[ok][:-1], lat[ok][:-1], lon[ok][1:], lat[ok][1:])
        dts = np.diff(t[ok])
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.where(dts > 0, step / dts, np.nan)
        row.update({
            "path_length_m": float(step.sum()),
            "speed_p95_mps": float(np.nanpercentile(speed, 95)),
            "speed_max_mps": float(np.nanmax(speed)),
            "n_jumps_gt_60mps": int(np.nansum(speed > 60)),  # > 216 km/h is implausible for a car
            "utm_epsg": LocalProjector.for_points(lon[ok], lat[ok]).epsg,
        })
    return row


def main() -> int:
    rows = []
    for part in DEV_PARTITIONS:
        trips = discover_trips(part)
        print(f"[{part}] {len(trips)} trips found")
        for trip_id, files in trips.items():
            trip = load_trip(files, with_route=True)
            row = {"partition": part, "trip_id": trip_id}
            for kind in ("track", "nodes", "arcs", "route"):
                p = getattr(files, kind)
                row[f"has_{kind}"] = p is not None
                row[f"{kind}_bytes"] = p.stat().st_size if p else 0
            row.update(track_summary(trip.track))
            if trip.nodes is not None and trip.arcs is not None:
                row.update(network_summary(trip.nodes, trip.arcs))
                if trip.route is not None:
                    row.update(route_summary(trip.route, trip.arcs))
            rows.append(row)

    if not rows:
        print("No trips found. Put the raw data under data/raw/<partition>/ (see README).")
        return 1
    df = pd.DataFrame(rows)
    out = repo_path("experiments/results/inventory.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(df)} trips)\n")
    print(df.groupby("partition")[["n_obs", "duration_s", "path_length_m", "n_nodes", "n_arcs"]]
          .describe().T.round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
