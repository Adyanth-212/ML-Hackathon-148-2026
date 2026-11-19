"""Build the candidate tables for TRAIN and STUDENT_TEST.

    python scripts/build_labels.py       # first (labels are attached here)
    python scripts/build_candidates.py

Writes, per trip, data/processed/candidates/<partition>/<trip_id>.parquet (zstd) with the
candidate-table columns ``trip_id, obs_idx, arc_id, dist_m, proj_frac, label`` and
experiments/results/candidate_stats.csv. R, K and the fallback radius come from
configs/candidates.yaml. Observations with no candidate even after the fallback keep zero rows
and are counted in ``n_no_candidates``.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap.candidates import CONTRACT_COLUMNS, attach_labels, generate_candidates
from pathsnap.config import load_config, repo_path
from pathsnap.geo import TripGeometry
from pathsnap.io import discover_trips, load_trip

DEV_PARTITIONS = ("train", "student_test")


def build(files) -> dict:
    t0 = time.perf_counter()
    trip = load_trip(files)
    geom = TripGeometry.from_trip(trip)
    cands, missing = generate_candidates(geom, files.trip_id)
    t_gen = time.perf_counter() - t0
    al = pd.read_parquet(repo_path(f"data/processed/labels/{files.partition}/{files.trip_id}.parquet"))
    cands = attach_labels(cands, al.arc_id.to_numpy())[CONTRACT_COLUMNS]
    out = repo_path(f"data/processed/candidates/{files.partition}/{files.trip_id}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    cands.to_parquet(out, compression="zstd", index=False)

    n = geom.n_obs
    per_obs = np.bincount(cands.obs_idx, minlength=n)
    hit = np.zeros(n, bool)
    hit[cands.obs_idx[cands.label == 1].to_numpy()] = True
    rel = ~al.is_outlier.to_numpy()
    cfg = load_config("candidates")
    return {
        "partition": files.partition, "trip_id": files.trip_id, "n_obs": n, "n_rows": len(cands),
        "mean_candidates": float(per_obs.mean()), "max_candidates_seen": int(per_obs.max()),
        "n_no_candidates": int(len(missing)), "n_fallback_used": int(((per_obs > 0) & (
            np.bincount(cands.obs_idx[cands.dist_m <= cfg["radius_m"]], minlength=n) == 0)).sum()),
        "miss_rate": float(1 - hit.mean()), "n_reliable": int(rel.sum()),
        "miss_rate_reliable": float(1 - hit[rel].mean()) if rel.any() else np.nan,
        "n_arcs": geom.n_arcs, "gen_seconds": t_gen, "file_mb": out.stat().st_size / 1e6,
    }


def main() -> int:
    cfg = load_config("candidates")
    jobs = sorted([f for p in DEV_PARTITIONS for f in discover_trips(p).values()], key=lambda f: -f.arcs.stat().st_size)
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=-1)(delayed(build)(f) for f in jobs)
    st = pd.DataFrame(rows).sort_values(["partition", "trip_id"])
    st.to_csv(repo_path("experiments/results/candidate_stats.csv"), index=False)
    print(f"Built {len(st)} trips in {time.perf_counter() - t0:.1f} s with {cfg}")
    for p, g in st.groupby("partition"):
        print(f"[{p}] rows {g.n_rows.sum():,}  obs {g.n_obs.sum():,}  mean cands {g.n_rows.sum() / g.n_obs.sum():.2f}  "
              f"miss {(g.miss_rate * g.n_obs).sum() / g.n_obs.sum():.4%}  "
              f"miss(reliable) {(g.miss_rate_reliable * g.n_reliable).sum() / g.n_reliable.sum():.4%}  "
              f"no-cand obs {g.n_no_candidates.sum()}  fallback obs {g.n_fallback_used.sum()}  "
              f"size {g.file_mb.sum():.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
