"""Classifier-only vs classifier+HMM, with a small decoder grid tuned on train trips.

    PYTHONPATH=src python scripts/eval_hmm.py
Writes experiments/results/hmm_grid.csv and hmm_comparison.csv. Evaluator data is never touched.
"""
from __future__ import annotations
import sys, copy
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, str(Path(__file__).parent / "models"))
from _sudhe_models import FEATURES, load_partition
from pathsnap import hmm, pipeline
from pathsnap.config import repo_path
from pathsnap.geo import TripGeometry
from pathsnap.io import discover_trips, load_trip
from pathsnap.candidates import generate_candidates

rf = RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=2, class_weight="balanced", random_state=148, n_jobs=-1)
tr = load_partition("train")
rf.fit(tr[FEATURES].to_numpy(float), tr.label.to_numpy(int))


def scorer(geom, c):
    c = c.copy(); g = c.groupby("obs_idx", sort=False)
    c["dist_rank"] = g.cumcount().astype(float)
    c["dist_minus_min"] = c.dist_m - g.dist_m.transform("min")
    c["proj_outside"] = ((c.proj_frac < 0) | (c.proj_frac > 1)).astype(float)
    c["n_candidates"] = g.arc_id.transform("size").astype(float)
    return rf.predict_proba(c[FEATURES].to_numpy(float))[:, 1]


def prep(part, n=None):
    out = []
    for f in list(discover_trips(part).values())[:n]:
        trip = load_trip(f); geom = TripGeometry.from_trip(trip)
        c, _ = generate_candidates(geom, trip.trip_id)
        lab = pd.read_parquet(repo_path(f"data/processed/labels/{part}/{f.trip_id}.parquet"))
        out.append((geom, c.reset_index(drop=True), lab.arc_id.to_numpy(), ~lab.is_outlier.to_numpy()))
    return out


def evaluate(data, cfg):
    hit = tot = 0
    for geom, c, y, ok in data:
        s = scorer(geom, c)
        pred = pipeline.argmax_per_obs(c, s, geom.n_obs) if cfg is None else hmm.decode_candidates(geom, c, s, cfg)[0]
        hit += int((pred == y)[ok].sum()); tot += int(ok.sum())
    return hit / tot


def cfgof(w, beta):
    cfg = copy.deepcopy(hmm.load_hmm_config()); cfg["emission"]["classifier_weight"] = w; cfg["transition"]["beta_m"] = beta
    return cfg


tune, test = prep("train", 10), prep("student_test")
rows = [{"w": w, "beta_m": b, "train_top1": evaluate(tune, cfgof(w, b))} for w in (0.0, 0.5, 1.0) for b in (10, 20, 50)]
grid = pd.DataFrame(rows); grid.to_csv(repo_path("experiments/results/hmm_grid.csv"), index=False); print(grid)
best = grid.sort_values("train_top1", ascending=False).iloc[0]
comp = pd.DataFrame([
    {"method": "classifier argmax", "student_test_top1": evaluate(test, None)},
    {"method": "HMM distance-only", "student_test_top1": evaluate(test, cfgof(0.0, 20))},
    {"method": f"classifier+HMM (w={best.w}, beta={best.beta_m:g}, tuned on train)", "student_test_top1": evaluate(test, cfgof(best.w, best.beta_m))},
])
comp.to_csv(repo_path("experiments/results/hmm_comparison.csv"), index=False); print(comp)
