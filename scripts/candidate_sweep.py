"""Choose candidate radius R and cap K from data (TRAIN trips only).

    python scripts/candidate_sweep.py

For every train observation the aligned (label) arc's rank among all arcs
within 200 m is computed once; the miss rate for any (R <= 200, K) follows
directly. Two variants are reported: all observations, and observations whose
label is reliable (``is_outlier == False``, see data_quality_log section 7).

Writes experiments/results/candidate_miss_rate.csv and
experiments/figures/candidate_miss_rate.png.
"""
from __future__ import annotations

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap.candidates import ArcIndex, miss_rate_grid, true_arc_rank
from pathsnap.config import repo_path
from pathsnap.geo import TripGeometry
from pathsnap.io import discover_trips, load_trip

RADII = [15, 25, 40, 60, 80, 120, 200]
KS = [5, 10, 15, 20, 30]


def sweep_trip(files) -> list[dict]:
    trip = load_trip(files)  # no route needed: labels come from the parquet
    al = pd.read_parquet(repo_path(f"data/processed/labels/train/{files.trip_id}.parquet"))
    geom = TripGeometry.from_trip(trip)
    idx = ArcIndex(geom)
    aligned, tdist = al.arc_id.to_numpy(), al.dist_m.to_numpy()
    rank, obs, d = true_arc_rank(idx, aligned, tdist, max(RADII))
    rows = []
    for subset, use in (("all", np.ones(len(al), bool)), ("reliable", ~al.is_outlier.to_numpy())):
        for r in miss_rate_grid(rank, tdist, obs, d, use, RADII, KS):
            rows.append({"trip_id": files.trip_id, "subset": subset, **r})
    return rows


def main() -> int:
    t0 = time.perf_counter()
    jobs = sorted(discover_trips("train").values(), key=lambda f: -f.arcs.stat().st_size)
    rows = [r for rs in Parallel(n_jobs=-1)(delayed(sweep_trip)(f) for f in jobs) for r in rs]
    per_trip = pd.DataFrame(rows)
    g = per_trip.groupby(["subset", "radius_m", "max_candidates"])[["n_obs", "n_miss", "sum_cands", "n_empty"]].sum()
    g["miss_rate"] = g.n_miss / g.n_obs
    g["mean_cands"] = g.sum_cands / g.n_obs
    g["empty_rate"] = g.n_empty / g.n_obs
    # trip-averaged miss rate, so a few long trips do not dominate
    tm = per_trip.assign(m=per_trip.n_miss / per_trip.n_obs.clip(lower=1)).groupby(
        ["subset", "radius_m", "max_candidates"]).m.mean().rename("miss_rate_trip_mean")
    g = g.join(tm).reset_index()
    g.insert(0, "partition", "train")
    g.to_csv(repo_path("experiments/results/candidate_miss_rate.csv"), index=False)
    print(f"Sweep over {len(jobs)} train trips in {time.perf_counter() - t0:.1f} s")
    print(g[g.subset == "reliable"].pivot(index="radius_m", columns="max_candidates", values="miss_rate").round(4))
    print(g[g.subset == "reliable"].pivot(index="radius_m", columns="max_candidates", values="mean_cands").round(2))
    print(g[g.subset == "all"].pivot(index="radius_m", columns="max_candidates", values="miss_rate").round(4))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    rel = g[g.subset == "reliable"]
    for K, sub in rel.groupby("max_candidates"):
        ax[0].plot(sub.radius_m, sub.miss_rate * 100, marker="o", label=f"K={K}")
        ax[1].plot(sub.radius_m, sub.mean_cands, marker="o", label=f"K={K}")
        ax[2].plot(sub.mean_cands, sub.miss_rate * 100, marker="o", label=f"K={K}")
    ax[0].set(title="TRAIN (reliable labels): candidate miss rate", xlabel="radius R (m)", ylabel="miss rate (%)",
              yscale="log")
    ax[1].set(title="TRAIN: mean candidates per observation", xlabel="radius R (m)", ylabel="candidates")
    ax[2].set(title="Trade-off (each point is one R)", xlabel="mean candidates", ylabel="miss rate (%)", yscale="log")
    for a in ax:
        a.grid(alpha=.3)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(repo_path("experiments/figures/candidate_miss_rate.png"), dpi=130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
