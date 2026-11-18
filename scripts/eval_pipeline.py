"""End-to-end check of pathsnap.pipeline.predict_trip on STUDENT_TEST (the API code path).

    python scripts/export_artifacts.py   # first
    python scripts/eval_pipeline.py

Runs predict_trip exactly as the API does (trip loaded without its route), once with the
exported classifier (load_artifacts) and once with the rule-based baseline, then scores the
returned arcs against the derived labels (reliable observations only). Writes
experiments/results/pipeline_student_test.csv (per trip) and pipeline_summary.csv.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pathsnap.config import repo_path
from pathsnap.io import discover_trips, load_trip
from pathsnap.pipeline import load_artifacts, predict_trip


def continuity(arcs: pd.DataFrame, pred: np.ndarray) -> tuple[float, int]:
    p = pred[pred >= 0]
    ch = np.flatnonzero(p[1:] != p[:-1])
    if not len(ch):
        return 1.0, 0
    ok = arcs.to_node.to_numpy()[p[ch]] == arcs.from_node.to_numpy()[p[ch + 1]]
    return float(ok.mean()), int((~ok).sum())


def main() -> int:
    variants = {"model": load_artifacts(use_decoder=False), "baseline_rule": {}}
    model_name = variants["model"].get("metadata", {}).get("model")
    if not model_name:
        raise SystemExit("artifacts/model.joblib missing: run scripts/export_artifacts.py")
    rows = []
    for tid, files in discover_trips("student_test").items():
        trip = load_trip(files)  # with_route=False: what the API does
        lab = pd.read_parquet(repo_path(f"data/processed/labels/student_test/{tid}.parquet"))
        ok = ~lab.is_outlier.to_numpy()
        for vname, art in variants.items():
            res = predict_trip(trip, art)
            pred = np.asarray(res["pred_arcs"])
            cont, breaks = continuity(trip.arcs, pred)
            rows.append({"partition": "student_test", "trip_id": tid, "variant": vname,
                         "scorer": res["method"]["scorer"], "n_obs": int(ok.sum()),
                         "n_correct": int((pred == lab.arc_id.to_numpy())[ok].sum()),
                         "top1": float((pred == lab.arc_id.to_numpy())[ok].mean()),
                         "route_continuity": cont, "n_breaks": breaks,
                         "n_no_candidate": len(res["no_candidate_obs"]), **res["timings_ms"]})
    df = pd.DataFrame(rows)
    df.to_csv(repo_path("experiments/results/pipeline_student_test.csv"), index=False)
    summ = df.groupby("variant").agg(scorer=("scorer", "first"), trips=("trip_id", "count"), n_obs=("n_obs", "sum"),
                                     n_correct=("n_correct", "sum"), continuity_mean=("route_continuity", "mean"),
                                     breaks_total=("n_breaks", "sum"), total_ms_mean=("total_ms", "mean"),
                                     total_ms_max=("total_ms", "max"), scoring_ms_mean=("scoring_ms", "mean"))
    summ.insert(3, "top1", summ.n_correct / summ.n_obs)
    summ.to_csv(repo_path("experiments/results/pipeline_summary.csv"))
    print(df[df.variant == "model"][["trip_id", "n_obs", "top1", "route_continuity", "total_ms"]].round(3).to_string(index=False))
    print(summ.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
