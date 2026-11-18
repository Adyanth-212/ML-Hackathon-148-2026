"""Shared training/evaluation helpers for Sudhe's tree ensembles."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import GroupKFold
from pathsnap.config import repo_path

FEATURES = ["dist_m", "proj_frac", "dist_rank", "dist_minus_min", "proj_outside", "n_candidates"]


def load_partition(partition):
    root = repo_path(f"data/processed/candidates/{partition}")
    parts = []
    for f in sorted(root.glob("*.parquet")):
        c = pd.read_parquet(f)
        lab = pd.read_parquet(repo_path(f"data/processed/labels/{partition}/{f.stem}.parquet"))
        aligned = lab.arc_id.to_numpy()
        outlier = lab.is_outlier.to_numpy()
        obs = c.obs_idx.to_numpy()
        c = c.copy()
        c["label"] = (c.arc_id.to_numpy() == aligned[obs]).astype(np.int8)
        c = c.loc[~outlier[obs]].copy()
        if c.empty:
            continue
        g = c.groupby("obs_idx", sort=False)
        c["dist_rank"] = g.cumcount().astype(np.float32)
        c["dist_minus_min"] = (c.dist_m - g.dist_m.transform("min")).astype(np.float32)
        c["proj_outside"] = ((c.proj_frac < 0) | (c.proj_frac > 1)).astype(np.float32)
        c["n_candidates"] = g.arc_id.transform("size").astype(np.float32)
        c["trip_id"] = f.stem
        parts.append(c)
    if not parts:
        raise FileNotFoundError(f"No usable candidate/label pairs for {partition}.")
    return pd.concat(parts, ignore_index=True)


def top1(df, probabilities):
    d = df[["trip_id", "obs_idx", "label"]].copy()
    d["p"] = probabilities
    best = d.loc[d.groupby(["trip_id", "obs_idx"], sort=False).p.idxmax()]
    return float(best.label.mean())


def run_model(name, estimator, config):
    start = time.perf_counter()
    train = load_partition("train")
    test = load_partition("student_test")
    x, y = train[FEATURES].to_numpy(float), train.label.to_numpy(int)
    groups = train.trip_id.to_numpy()
    folds = []
    splitter = GroupKFold(n_splits=5)
    for fold, (tr, va) in enumerate(splitter.split(x, y, groups), 1):
        model = clone(estimator)
        t0 = time.perf_counter()
        model.fit(x[tr], y[tr])
        fit_s = time.perf_counter() - t0
        p = model.predict_proba(x[va])[:, 1]
        folds.append({"fold": fold, "macro_f1": float(f1_score(y[va], p >= .5, average="macro", zero_division=0)),
                      "top1": top1(train.iloc[va].reset_index(drop=True), p), "fit_seconds": fit_s})
    model = clone(estimator)
    t0 = time.perf_counter()
    model.fit(x, y)
    fit_s = time.perf_counter() - t0
    p = model.predict_proba(test[FEATURES].to_numpy(float))[:, 1]
    pred = p >= .5
    result = {
        "name": name, "family": "tree ensemble", "config": config, "seed": 148,
        "status": "development evaluation; 5-fold GroupKFold by trip_id and held-out student_test",
        "features": FEATURES, "train_rows": len(train), "student_test_rows": len(test),
        "cv": {"folds": folds, "macro_f1_mean": float(np.mean([f["macro_f1"] for f in folds])),
               "macro_f1_std": float(np.std([f["macro_f1"] for f in folds]))},
        "student_test": {"top1_accuracy": top1(test, p),
                         "precision": float(precision_score(test.label, pred, zero_division=0)),
                         "recall": float(recall_score(test.label, pred, zero_division=0)),
                         "macro_f1": float(f1_score(test.label, pred, average="macro", zero_division=0)),
                         "log_loss": float(log_loss(test.label, np.c_[1-p, p], labels=[0,1]))},
        "fit_seconds": fit_s, "total_seconds": time.perf_counter() - start,
    }
    out = repo_path(f"experiments/results/models/{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"{name}: CV macro-F1={result['cv']['macro_f1_mean']:.4f}; student-test Top-1={result['student_test']['top1_accuracy']:.4f}; saved {out}")
    return result
