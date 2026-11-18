"""Rule-based baseline (not ML): nearest arc, optionally penalised by heading mismatch.

    python scripts/eval_baseline.py

The heading weight is chosen on TRAIN (Top-1 over reliable-label observations); student_test is
reported for the chosen weight and the plain nearest-arc rule. Runs the real
``pathsnap.pipeline.predict_trip`` code path for student_test (candidates rebuilt from raw
files), so this also checks the pipeline end to end. Writes
experiments/results/baseline_heading_sweep.csv and experiments/results/baseline_student_test.csv.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap import pipeline
from pathsnap.config import repo_path
from pathsnap.geo import TripGeometry
from pathsnap.io import discover_trips, load_trip

WEIGHTS = [0, 2, 5, 10, 20, 40]


def continuity(trip, pred: np.ndarray) -> float:
    """Share of arc changes where the previous arc's to_node is the next arc's from_node."""
    p = pred[pred >= 0]
    ch = np.flatnonzero(p[1:] != p[:-1])
    if len(ch) == 0:
        return 1.0
    a, b = p[ch], p[ch + 1]
    return float(np.mean(trip.arcs["to_node"].to_numpy()[a] == trip.arcs["from_node"].to_numpy()[b]))


def sweep_trip(files) -> list[dict]:
    trip = load_trip(files)
    geom = TripGeometry.from_trip(trip)
    c = pd.read_parquet(repo_path(f"data/processed/candidates/{files.partition}/{files.trip_id}.parquet"))
    lab = pd.read_parquet(repo_path(f"data/processed/labels/{files.partition}/{files.trip_id}.parquet"))
    ok = ~lab.is_outlier.to_numpy()
    rows = []
    for w in WEIGHTS:
        pred = pipeline.argmax_per_obs(c, pipeline.baseline_scores(geom, c, w), geom.n_obs)
        rows.append({"partition": files.partition, "trip_id": files.trip_id, "heading_weight_m": w,
                     "n_obs": int(ok.sum()), "n_correct": int((pred == lab.arc_id.to_numpy())[ok].sum())})
    return rows


def pipeline_trip(files) -> dict:
    trip = load_trip(files)  # no route: exactly what the API does
    res = pipeline.predict_trip(trip, None)
    lab = pd.read_parquet(repo_path(f"data/processed/labels/{files.partition}/{files.trip_id}.parquet"))
    ok = ~lab.is_outlier.to_numpy()
    pred = np.asarray(res["pred_arcs"])
    return {"partition": files.partition, "trip_id": files.trip_id, "n_obs": int(ok.sum()),
            "top1": float((pred == lab.arc_id.to_numpy())[ok].mean()),
            "route_continuity": continuity(trip, pred), "n_no_candidate": len(res["no_candidate_obs"]),
            **res["timings_ms"]}


def main() -> int:
    jobs = [f for p in ("train", "student_test") for f in discover_trips(p).values()]
    sw = pd.DataFrame([r for rs in Parallel(n_jobs=-1)(delayed(sweep_trip)(f) for f in jobs) for r in rs])
    g = sw.groupby(["partition", "heading_weight_m"])[["n_obs", "n_correct"]].sum()
    g["top1"] = g.n_correct / g.n_obs
    g = g.reset_index()
    g.to_csv(repo_path("experiments/results/baseline_heading_sweep.csv"), index=False)
    print(g.pivot(index="heading_weight_m", columns="partition", values="top1").round(4))
    best = int(g[g.partition == "train"].sort_values("top1", ascending=False).heading_weight_m.iloc[0])
    print(f"best heading weight on TRAIN: {best} m (pipeline default {pipeline.BASELINE_HEADING_WEIGHT_M:g} m)")

    st = pd.DataFrame(Parallel(n_jobs=-1)(delayed(pipeline_trip)(f) for f in discover_trips("student_test").values()))
    st.to_csv(repo_path("experiments/results/baseline_student_test.csv"), index=False)
    print(st[["trip_id", "n_obs", "top1", "route_continuity", "total_ms"]].round(3).to_string(index=False))
    print(f"student_test via predict_trip: Top-1 {(st.top1 * st.n_obs).sum() / st.n_obs.sum():.4f}, "
          f"continuity mean {st.route_continuity.mean():.3f}, total ms/trip mean {st.total_ms.mean():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
