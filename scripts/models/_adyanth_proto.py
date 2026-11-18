"""Pre-harness prototype evaluation for Adyanth's models (Logistic Regression, Gaussian NB).

Used only until Advitha's ``features.py`` / ``harness.py`` land on main; then the model
scripts switch to ``harness.evaluate_model`` and these results are superseded.

Protocol (mirrors SPLIT.md section 4 so results stay comparable):
* data: committed candidate tables (R=60 m, K=30) + simple per-candidate features computed
  here from the candidate table only (no labels, no route):
  ``dist_m, dist_rank, dist_minus_min, proj_frac, proj_outside, n_candidates``
* observations whose derived label is unreliable (``is_outlier``, > 50 m from the route) are
  dropped from training and evaluation (count recorded); misses stay in (all-zero groups)
* CV: GroupKFold(5, shuffle=True, random_state=148) by trip on the 80 train trips
* held-out: the 10 student_test trips; evaluator never touched
* diagnostics: Top-1 counting the reverse/duplicate twin as correct, and the nearest-arc rule
* candidate-level metrics at threshold 0.5 (macro over the two classes); Top-1 = argmax of
  predict_proba per observation equals the label arc (observations with no positive count as wrong)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from pathsnap.config import load_config, repo_path

SEED = 148
FEATURES = ["dist_m", "dist_rank", "dist_minus_min", "proj_frac", "proj_outside", "n_candidates"]


def add_simple_features(c: pd.DataFrame) -> pd.DataFrame:
    """Per-candidate features from the candidate table alone (rows are sorted by obs, dist)."""
    g = c.groupby(["trip_id", "obs_idx"], sort=False)
    c = c.copy()
    c["dist_rank"] = g.cumcount().astype(np.float32)
    c["dist_minus_min"] = (c["dist_m"] - g["dist_m"].transform("min")).astype(np.float32)
    c["proj_outside"] = ((c["proj_frac"] < 0) | (c["proj_frac"] > 1)).astype(np.float32)
    c["n_candidates"] = g["arc_id"].transform("size").astype(np.float32)
    return c


def load_partition(partition: str) -> tuple[pd.DataFrame, dict]:
    files = sorted(repo_path(f"data/processed/candidates/{partition}").glob("*.parquet"))
    parts, n_obs, n_drop, trips = [], 0, 0, {}
    for f in files:
        c = pd.read_parquet(f)
        lab = pd.read_parquet(repo_path(f"data/processed/labels/{partition}/{f.stem}.parquet"))
        ok = ~lab.is_outlier.to_numpy()
        n_obs += int(ok.sum())
        n_drop += int((~ok).sum())
        trips[f.stem] = int(ok.sum())
        parts.append(c[ok[c.obs_idx.to_numpy()]])
    df = add_simple_features(pd.concat(parts, ignore_index=True))
    return df, {"n_obs_used": n_obs, "n_obs_dropped_unreliable": n_drop, "obs_per_trip": trips}


def top1(df: pd.DataFrame, proba: np.ndarray, n_obs: int) -> float:
    """Share of (used) observations whose highest-probability candidate is the label arc."""
    d = df[["trip_id", "obs_idx", "label"]].assign(p=proba)
    best = d.loc[d.groupby(["trip_id", "obs_idx"], sort=False)["p"].idxmax()]
    return float(best["label"].sum() / n_obs)


def top1_diagnostics(df: pd.DataFrame, proba: np.ndarray, partition: str) -> dict:
    """Top-1 counting a geometric twin (same unordered node pair: reverse or duplicate arc) as correct,
    plus the nearest-arc rule (rank 0) for reference. Reads arcs only, never routes."""
    from pathsnap.io import discover_trips, read_arcs

    trips = discover_trips(partition)
    d = df[["trip_id", "obs_idx", "arc_id", "label", "dist_rank"]].assign(p=proba)
    best = d.loc[d.groupby(["trip_id", "obs_idx"], sort=False)["p"].idxmax()]
    truth = d[d.label == 1].set_index(["trip_id", "obs_idx"])["arc_id"]
    nearest = d[d.dist_rank == 0].set_index(["trip_id", "obs_idx"])["arc_id"]
    n_obs = d.groupby(["trip_id", "obs_idx"]).ngroups
    out = {"top1_undirected": 0, "nearest_arc_top1": 0, "nearest_arc_top1_undirected": 0}
    for tid, b in best.groupby("trip_id"):
        arcs = read_arcs(trips[tid].arcs)
        key = np.minimum(arcs.from_node.to_numpy(), arcs.to_node.to_numpy()) * 10**7 + \
            np.maximum(arcs.from_node.to_numpy(), arcs.to_node.to_numpy())
        b = b.set_index(["trip_id", "obs_idx"])
        tr = truth.reindex(b.index)
        nr = nearest.reindex(b.index)
        ok = tr.notna().to_numpy()
        t_arc = tr.to_numpy()[ok].astype(int)
        out["top1_undirected"] += int((key[b.arc_id.to_numpy()[ok]] == key[t_arc]).sum())
        out["nearest_arc_top1"] += int((nr.to_numpy()[ok].astype(int) == t_arc).sum())
        out["nearest_arc_top1_undirected"] += int((key[nr.to_numpy()[ok].astype(int)] == key[t_arc]).sum())
    return {k: v / n_obs for k, v in out.items()}


def candidate_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "precision_pos": float(precision_score(y, pred, zero_division=0)),
        "recall_pos": float(recall_score(y, pred, zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


def evaluate(name: str, family: str, estimator, config: dict) -> dict:
    t_all = time.perf_counter()
    tr, tr_info = load_partition("train")
    te, te_info = load_partition("student_test")
    X, y, groups = tr[FEATURES].to_numpy(np.float64), tr["label"].to_numpy(), tr["trip_id"].to_numpy()

    folds = []
    cv = GroupKFold(n_splits=load_config("data")["cv"]["n_splits"], shuffle=True, random_state=SEED)
    for k, (a, b) in enumerate(cv.split(X, y, groups)):
        est = clone(estimator).fit(X[a], y[a])
        p = est.predict_proba(X[b])[:, 1]
        val = tr.iloc[b]
        n_val = sum(tr_info["obs_per_trip"][t] for t in np.unique(val.trip_id))
        p_tr = est.predict_proba(X[a])[:, 1]
        folds.append({"fold": k, "n_val_trips": int(len(np.unique(groups[b]))),
                      "val_trips": sorted(map(str, np.unique(groups[b]))),
                      "train_macro_f1": candidate_metrics(y[a], p_tr)["macro_f1"],
                      **candidate_metrics(y[b], p), "top1": top1(val, p, n_val)})

    t0 = time.perf_counter()
    final = clone(estimator).fit(X, y)
    fit_s = time.perf_counter() - t0

    # held-out: per-trip inference timing including feature computation
    proba, per_trip_ms = np.empty(len(te)), []
    for tid, idx in te.groupby("trip_id", sort=False).indices.items():
        raw = te.iloc[idx][["trip_id", "obs_idx", "arc_id", "dist_m", "proj_frac"]]
        t0 = time.perf_counter()
        f = add_simple_features(raw)[FEATURES].to_numpy(np.float64)
        proba[idx] = final.predict_proba(f)[:, 1]
        per_trip_ms.append((time.perf_counter() - t0) * 1000)
    yt = te["label"].to_numpy()
    st = candidate_metrics(yt, proba)

    cand_stats = pd.read_csv(repo_path("experiments/results/candidate_stats.csv"), dtype={"trip_id": str})
    cs = cand_stats[cand_stats.partition == "student_test"]
    miss = float((cs.miss_rate_reliable * cs.n_reliable).sum() / cs.n_reliable.sum())

    cv_f1 = [f["macro_f1"] for f in folds]
    out = {
        "name": name, "family": family, "config": config, "seed": SEED,
        "status": "prototype (pre-harness): simple features from the candidate table; rerun via harness when it lands",
        "features": FEATURES,
        "data": {"candidates": load_config("candidates"), "train": {k: v for k, v in tr_info.items() if k != "obs_per_trip"},
                 "student_test": {k: v for k, v in te_info.items() if k != "obs_per_trip"},
                 "train_rows": int(len(tr)), "student_test_rows": int(len(te)),
                 "train_pos_rate": float(y.mean())},
        "cv": {"scheme": "GroupKFold(5, shuffle=True, random_state=148) by trip_id, 80 train trips",
               "macro_f1_folds": cv_f1, "macro_f1_mean": float(np.mean(cv_f1)), "macro_f1_std": float(np.std(cv_f1)),
               "top1_mean": float(np.mean([f["top1"] for f in folds])),
               "train_macro_f1_mean": float(np.mean([f["train_macro_f1"] for f in folds])), "folds": folds},
        "student_test": {"top1": top1(te, proba, te_info["n_obs_used"]), **st,
                         **top1_diagnostics(te, proba, "student_test"),
                         "candidate_miss_rate": miss,
                         "top1_upper_bound": 1 - miss},
        "route_continuity": None,
        "route_continuity_reason": "classifier-only argmax, no decoder; computed in the HMM stage (metrics.py not on main yet)",
        "fit_seconds": fit_s,
        "inference_ms_per_trip": float(np.mean(per_trip_ms)),
        "inference_ms_per_point": float(np.sum(per_trip_ms) / te_info["n_obs_used"]),
        "total_seconds": time.perf_counter() - t_all,
    }
    path = repo_path(f"experiments/results/models/{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"{name}: CV macro F1 {out['cv']['macro_f1_mean']:.4f} +/- {out['cv']['macro_f1_std']:.4f} | "
          f"student_test Top-1 {out['student_test']['top1']:.4f}, macro F1 {st['macro_f1']:.4f}, "
          f"log loss {st['log_loss']:.4f} | fit {fit_s:.1f} s, {out['inference_ms_per_trip']:.1f} ms/trip -> {path}")
    return out
