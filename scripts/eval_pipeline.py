"""End-to-end check of pathsnap.pipeline.predict_trip on STUDENT_TEST (the API code path).

    python scripts/export_artifacts.py   # first
    python scripts/eval_pipeline.py

Runs predict_trip exactly as the API does (trip loaded without its route), with the exported
classifier + HMM decoder (``model_hmm``, the API path), the classifier alone (``model``) and the
rule-based baseline, then scores the
returned arcs against the derived labels (reliable observations only). Writes
experiments/results/pipeline_student_test.csv (per trip), pipeline_summary.csv and
pipeline_error_types.csv (each wrong observation classified as: reverse twin of the label arc,
duplicate arc, adjacent arc sharing a node, or other).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pathsnap.config import repo_path
from pathsnap.geo import TripGeometry
from pathsnap.io import discover_trips, load_trip
from pathsnap.pipeline import load_artifacts, predict_trip


def continuity(arcs: pd.DataFrame, pred: np.ndarray) -> tuple[float, int]:
    p = pred[pred >= 0]
    ch = np.flatnonzero(p[1:] != p[:-1])
    if not len(ch):
        return 1.0, 0
    ok = arcs.to_node.to_numpy()[p[ch]] == arcs.from_node.to_numpy()[p[ch + 1]]
    return float(ok.mean()), int((~ok).sum())


def error_types(arcs: pd.DataFrame, pred: np.ndarray, label: np.ndarray, use: np.ndarray, geom=None) -> dict:
    """Classify wrong observations. For reverse-twin errors with a trusted GPS heading, also count
    whether the predicted or the label arc points the way the vehicle is moving (label-noise check)."""
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    p = np.where(pred >= 0, pred, 0)
    wrong = use & (pred != label)
    twin = wrong & (fn[p] == tn[label]) & (tn[p] == fn[label])
    dup = wrong & (fn[p] == fn[label]) & (tn[p] == tn[label])
    adj = wrong & ~twin & ~dup & ((tn[p] == fn[label]) | (fn[p] == tn[label]) | (fn[p] == fn[label]) | (tn[p] == tn[label]))
    out = {"wrong": int(wrong.sum()), "reverse_twin": int(twin.sum()), "duplicate_arc": int(dup.sum()),
           "adjacent_arc": int(adj.sum()), "other": int((wrong & ~twin & ~dup & ~adj).sum())}
    if geom is not None:
        from pathsnap.labels import obs_heading

        ux, uy, valid = obs_heading(geom, 5.0)

        def cos(a):
            vx, vy = geom.bx[a] - geom.ax[a], geom.by[a] - geom.ay[a]
            return (ux * vx + uy * vy) / np.maximum(np.hypot(vx, vy), 1e-9)

        m = twin & valid
        out["twin_with_heading"] = int(m.sum())
        out["twin_pred_agrees_heading"] = int((cos(p)[m] > 0).sum())
        out["twin_label_agrees_heading"] = int((cos(label)[m] > 0).sum())
    return out


def main() -> int:
    variants = {"model_hmm": load_artifacts(), "model": load_artifacts(use_decoder=False), "baseline_rule": {}}
    if "decoder" not in variants["model_hmm"]:
        del variants["model_hmm"]  # no decoder available: classifier-only, as before
    model_name = variants["model"].get("metadata", {}).get("model")
    if not model_name:
        raise SystemExit("artifacts/model.joblib missing: run scripts/export_artifacts.py")
    rows, errs = [], []
    for tid, files in tqdm(discover_trips("student_test").items(), desc="student_test trips", unit="trip"):
        trip = load_trip(files)  # with_route=False: what the API does
        lab = pd.read_parquet(repo_path(f"data/processed/labels/student_test/{tid}.parquet"))
        ok = ~lab.is_outlier.to_numpy()
        geom = TripGeometry.from_trip(trip)
        for vname, art in variants.items():
            res = predict_trip(trip, art)
            pred = np.asarray(res["pred_arcs"])
            cont, breaks = continuity(trip.arcs, pred)
            errs.append({"trip_id": tid, "variant": vname,
                         **error_types(trip.arcs, pred, lab.arc_id.to_numpy(), ok, geom)})
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
    e = pd.DataFrame(errs)
    e.to_csv(repo_path("experiments/results/pipeline_error_types.csv"), index=False)
    et = e.groupby("variant")[["wrong", "reverse_twin", "duplicate_arc", "adjacent_arc", "other"]].sum()
    print(et.assign(**{f"{c}_share": et[c] / et.wrong for c in ["reverse_twin", "adjacent_arc", "other"]}).round(3).to_string())
    print(df[df.variant == ("model_hmm" if "model_hmm" in variants else "model")][["trip_id", "n_obs", "top1", "route_continuity", "total_ms"]].round(3).to_string(index=False))
    print(summ.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
