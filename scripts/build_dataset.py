"""Build the shared feature table for TRAIN and STUDENT_TEST.

    python scripts/build_dataset.py        # a few seconds on 8 cores

Input: the committed candidate tables (data/processed/candidates/) and label flags
(data/processed/labels/). Output, per trip:
  data/processed_local/features/<partition>/<trip_id>.parquet
    trip_id, obs_idx, arc_id, label, obs_reliable, <FEATURE_COLUMNS>
plus artifacts/feature_list.json and experiments/results/feature_summary.csv.

The feature files (~175 MB) are gitignored; they are rebuilt from committed inputs in
seconds. ``obs_reliable`` is False for observations whose derived label is > 50 m from
the reference route (data_quality_log section 7); it is metadata, never a feature.
"""
from __future__ import annotations

import sys
import time

import pandas as pd
from joblib import Parallel, delayed

from pathsnap.config import repo_path
from pathsnap.features import FEATURE_COLUMNS, build_features, write_feature_list
from pathsnap.geo import TripGeometry
from pathsnap.io import discover_trips, load_trip

DEV_PARTITIONS = ("train", "student_test")
OUT_DIR = "data/processed_local/features"


def build(files) -> dict:
    t0 = time.perf_counter()
    geom = TripGeometry.from_trip(load_trip(files))
    c = pd.read_parquet(repo_path(f"data/processed/candidates/{files.partition}/{files.trip_id}.parquet"))
    lab = pd.read_parquet(repo_path(f"data/processed/labels/{files.partition}/{files.trip_id}.parquet"))
    f = build_features(geom, c)
    f.insert(4, "obs_reliable", ~lab.is_outlier.to_numpy()[f.obs_idx.to_numpy()])
    out = repo_path(f"{OUT_DIR}/{files.partition}/{files.trip_id}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out, compression="zstd", index=False)
    return {"partition": files.partition, "trip_id": files.trip_id, "rows": len(f),
            "seconds": time.perf_counter() - t0, "mb": out.stat().st_size / 1e6}


def main() -> int:
    jobs = sorted([f for p in DEV_PARTITIONS for f in discover_trips(p).values()], key=lambda f: -f.arcs.stat().st_size)
    t0 = time.perf_counter()
    stats = pd.DataFrame(Parallel(n_jobs=-1)(delayed(build)(f) for f in jobs))
    print(f"Built features for {len(stats)} trips in {time.perf_counter() - t0:.1f} s "
          f"({stats.rows.sum():,} rows, {stats.mb.sum():.0f} MB) -> {OUT_DIR}/")
    print("Feature list:", write_feature_list())

    # summary on TRAIN only: per-feature mean for positives / negatives (reliable observations)
    tr = pd.concat([pd.read_parquet(p) for p in sorted(repo_path(f"{OUT_DIR}/train").glob("*.parquet"))])
    tr = tr[tr.obs_reliable]
    summ = tr.groupby("label")[FEATURE_COLUMNS].mean().T
    summ.columns = ["mean_label0", "mean_label1"]
    summ["std_all"] = tr[FEATURE_COLUMNS].std()
    summ["std_effect"] = (summ.mean_label1 - summ.mean_label0) / summ.std_all
    summ.insert(0, "partition", "train")
    summ.to_csv(repo_path("experiments/results/feature_summary.csv"), index_label="feature")
    print(summ.sort_values("std_effect", key=abs, ascending=False).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
