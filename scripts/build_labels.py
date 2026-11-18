"""Milestone 2: derive per-observation labels for TRAIN and STUDENT_TEST trips.

    python scripts/build_labels.py

Writes
  data/processed/labels/<partition>/<trip_id>.parquet   obs_idx, route_pos, arc_id, dist_m, is_outlier, after_gap
  experiments/results/label_quality.csv                 per-trip alignment quality
  experiments/figures/label_quality.png
"""
from __future__ import annotations

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

from pathsnap.config import repo_path
from pathsnap.geo import TripGeometry
from pathsnap.io import discover_trips, load_trip
from pathsnap.labels import AlignParams, align_route_to_observations, label_quality

DEV_PARTITIONS = ("train", "student_test")


def run(files, params: AlignParams) -> dict:
    t0 = time.perf_counter()
    trip = load_trip(files, with_route=True)
    geom = TripGeometry.from_trip(trip)
    al = align_route_to_observations(geom, trip.route, params)
    out = repo_path(f"data/processed/labels/{files.partition}/{files.trip_id}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    al.to_parquet(out, compression="zstd", index=False)
    q = label_quality(files.trip_id, files.partition, al, len(trip.route))
    q["n_outliers"] = int(al.is_outlier.sum())
    q["seconds"] = time.perf_counter() - t0
    return q


def main() -> int:
    params = AlignParams()
    jobs = [f for p in DEV_PARTITIONS for f in discover_trips(p).values()]
    jobs.sort(key=lambda f: -f.track.stat().st_size)  # big trips first for load balance
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=-1)(delayed(run)(f, params) for f in jobs)
    q = pd.DataFrame(rows).sort_values(["partition", "trip_id"])
    q.to_csv(repo_path("experiments/results/label_quality.csv"), index=False)
    print(f"Aligned {len(q)} trips in {time.perf_counter() - t0:.1f} s with {params}")
    for p, g in q.groupby("partition"):
        w = g.n_obs / g.n_obs.sum()
        print(f"[{p}] obs-weighted: >50 m {(g.frac_gt_50m * w).sum():.3%}, >100 m {(g.frac_gt_100m * w).sum():.3%}; "
              f"median of trip medians {g.dist_median_m.median():.1f} m; route coverage median {g.route_coverage.median():.3f}"
              f" min {g.route_coverage.min():.3f}")
    print(q.sort_values("frac_gt_50m", ascending=False).head(8).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for p, c in (("train", "#1f77b4"), ("student_test", "#d62728")):
        g = q[q.partition == p]
        ax[0].hist(g.dist_median_m, bins=30, alpha=.7, color=c, label=p)
        ax[1].scatter(g.dist_p95_m, g.frac_gt_50m, s=15, c=c, label=p)
        ax[2].scatter(g.n_obs, g.route_coverage, s=15, c=c, label=p)
    ax[0].set(title="Median distance to aligned arc per trip", xlabel="m")
    ax[1].set(title="Per trip: p95 distance vs share > 50 m", xlabel="p95 distance (m)", ylabel="share of points > 50 m")
    ax[2].set(title="Route arcs that received >= 1 point", xlabel="observations", ylabel="coverage", xscale="log")
    ax[0].legend(); fig.tight_layout(); fig.savefig(repo_path("experiments/figures/label_quality.png"), dpi=130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
