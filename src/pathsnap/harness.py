"""Shared evaluation harness: every model is evaluated the same way (docs/REFERENCE.md).

    from pathsnap.harness import evaluate_model
    evaluate_model("advitha_lgbm", "gradient boosting", estimator, config)

Protocol
--------
* data: the shared feature table (``scripts/build_dataset.py``), columns from
  ``artifacts/feature_list.json`` unless ``features=`` is given (ablations)
* observations whose derived label is unreliable (``obs_reliable == False``) are dropped
  from training and evaluation; the counts are recorded
* CV: ``GroupKFold(5, shuffle=True, random_state=148)`` by trip_id on the 80 train trips.
  The fold assignment is fixed by the seed and saved in experiments/results/cv_folds.csv.
* held-out: the 10 student_test trips (the evaluator partition is never read)
* candidate-level metrics at threshold 0.5 (macro over the two classes), log loss on
  ``predict_proba[:, 1]``; observation-level Top-1 = the highest-probability candidate is
  the label arc (observations whose label arc is not among the candidates count as wrong)
* route continuity of the classifier-only argmax sequence (no decoder)
* ``max_train_obs``: optional random subsample of training *observations* (all their
  candidate rows kept) for slow models such as KNN / SVM; recorded in the JSON. The
  validation folds and the student_test set are never subsampled.

Writes ``experiments/results/models/<name>.json`` and the student_test probabilities to
``data/processed_local/preds/<name>_student_test.parquet`` (for HMM experiments).
"""
from __future__ import annotations

import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

from .config import load_config, repo_path
from .features import FEATURE_COLUMNS

SEED = 148
FEATURE_DIR = "data/processed_local/features"


@lru_cache(maxsize=2)
def _load(partition: str) -> pd.DataFrame:
    files = sorted(repo_path(f"{FEATURE_DIR}/{partition}").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No feature files in {FEATURE_DIR}/{partition}; run scripts/build_dataset.py first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["trip_id"] = df["trip_id"].astype(str)
    return df


def load_split(partition: str) -> tuple[pd.DataFrame, dict]:
    """Feature rows of reliable observations + bookkeeping (n observations used / dropped)."""
    df = _load(partition)
    obs = df.groupby(["trip_id", "obs_idx"], sort=False)["obs_reliable"].first()
    # observations with no candidate at all are not in the table; count them from the labels
    n_total = sum(len(pd.read_parquet(p)) for p in sorted(repo_path(f"data/processed/labels/{partition}").glob("*.parquet")))
    n_unrel = sum(int(pd.read_parquet(p).is_outlier.sum())
                  for p in sorted(repo_path(f"data/processed/labels/{partition}").glob("*.parquet")))
    used = df[df["obs_reliable"]].reset_index(drop=True)
    info = {"n_obs_total": n_total, "n_obs_dropped_unreliable": n_unrel, "n_obs_used": n_total - n_unrel,
            "n_obs_with_candidates": int(obs[obs].size), "n_rows": int(len(used)), "n_trips": int(used.trip_id.nunique())}
    return used, info


def cv_folds(trip_ids, n_splits: int | None = None) -> dict[str, int]:
    """Fixed trip -> fold assignment (same for every model)."""
    n_splits = n_splits or load_config("data")["cv"]["n_splits"]
    trips = np.array(sorted(set(trip_ids)))
    out = {}
    for k, (_, b) in enumerate(GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(trips, groups=trips)):
        out.update({t: k for t in trips[b]})
    return out


def top1(df: pd.DataFrame, proba: np.ndarray, n_obs: int) -> float:
    d = pd.DataFrame({"trip_id": df["trip_id"].to_numpy(), "obs_idx": df["obs_idx"].to_numpy(),
                      "label": df["label"].to_numpy(), "p": proba})
    best = d.loc[d.groupby(["trip_id", "obs_idx"], sort=False)["p"].idxmax()]
    return float(best["label"].sum() / n_obs)


def argmax_arcs(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"trip_id": df["trip_id"].to_numpy(), "obs_idx": df["obs_idx"].to_numpy(),
                      "arc_id": df["arc_id"].to_numpy(), "p": proba})
    return d.loc[d.groupby(["trip_id", "obs_idx"], sort=False)["p"].idxmax()].sort_values(["trip_id", "obs_idx"])


def route_continuity(best: pd.DataFrame, partition: str) -> float:
    """Mean over trips of: share of predicted-arc changes where prev.to_node == next.from_node."""
    from .io import discover_trips, read_arcs

    trips, vals = discover_trips(partition), []
    for tid, g in best.groupby("trip_id"):
        arcs = read_arcs(trips[tid].arcs)
        p = g["arc_id"].to_numpy()
        ch = np.flatnonzero(p[1:] != p[:-1])
        if len(ch):
            vals.append(float(np.mean(arcs.to_node.to_numpy()[p[ch]] == arcs.from_node.to_numpy()[p[ch + 1]])))
    return float(np.mean(vals)) if vals else float("nan")


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


def _proba(est, X) -> np.ndarray:
    return est.predict_proba(X)[:, 1]


def _subsample(df: pd.DataFrame, max_obs: int | None, rng: np.random.Generator) -> np.ndarray:
    """Row indices of a random subset of whole observations."""
    if not max_obs:
        return np.arange(len(df))
    key = df["trip_id"].to_numpy() + ":" + df["obs_idx"].astype(str).to_numpy()
    uniq = np.unique(key)
    if len(uniq) <= max_obs:
        return np.arange(len(df))
    keep = rng.choice(uniq, size=max_obs, replace=False)
    return np.flatnonzero(np.isin(key, keep))


def evaluate_model(name: str, family: str, estimator, config: dict, features: list[str] | None = None,
                   max_train_obs: int | None = None, train_score: bool = True, verbose: bool = True) -> dict:
    """Cross-validate on train, refit on all train, score student_test, write the JSON."""
    t_all = time.perf_counter()
    feats = list(features or FEATURE_COLUMNS)
    rng = np.random.default_rng(SEED)
    tr, tr_info = load_split("train")
    te, te_info = load_split("student_test")
    X, y = tr[feats].to_numpy(np.float32), tr["label"].to_numpy()
    fold_of = cv_folds(tr["trip_id"])
    fold = tr["trip_id"].map(fold_of).to_numpy()
    obs_per_trip = tr.groupby("trip_id")["obs_idx"].nunique()
    # observations without any candidate still count in the Top-1 denominator
    lab_n = {p.stem: int((~pd.read_parquet(p).is_outlier).sum())
             for p in repo_path("data/processed/labels/train").glob("*.parquet")}

    folds = []
    fold_ids = sorted(set(fold))
    bar = tqdm(total=len(fold_ids) + 1, desc=name, unit="fit", disable=not verbose, dynamic_ncols=True)
    for k in fold_ids:
        bar.set_postfix_str(f"fitting fold {k}")
        tr_idx = np.flatnonzero(fold != k)
        va_idx = np.flatnonzero(fold == k)
        tr_idx = tr_idx[_subsample(tr.iloc[tr_idx], max_train_obs, rng)]
        t0 = time.perf_counter()
        est = clone(estimator).fit(X[tr_idx], y[tr_idx])
        fit_s = time.perf_counter() - t0
        p = _proba(est, X[va_idx])
        va = tr.iloc[va_idx]
        row = {"fold": int(k), "val_trips": sorted(va.trip_id.unique().tolist()), "fit_seconds": fit_s,
               **candidate_metrics(y[va_idx], p), "top1": top1(va, p, sum(lab_n[t] for t in va.trip_id.unique()))}
        if train_score:
            row["train_macro_f1"] = candidate_metrics(y[tr_idx], _proba(est, X[tr_idx]))["macro_f1"]
        folds.append(row)
        bar.update(1)
        bar.set_postfix_str(f"fold {k}: F1 {row['macro_f1']:.4f} Top-1 {row['top1']:.4f} ({fit_s:.0f} s)")
        if verbose:
            tqdm.write(f"  [{name}] fold {k}: macro F1 {row['macro_f1']:.4f}, Top-1 {row['top1']:.4f}, fit {fit_s:.1f} s")

    full_idx = _subsample(tr, max_train_obs, rng)
    bar.set_postfix_str("final fit on all train")
    t0 = time.perf_counter()
    final = clone(estimator).fit(X[full_idx], y[full_idx])
    fit_s = time.perf_counter() - t0
    bar.update(1)
    bar.close()

    # student_test: per-trip inference timing (feature rows -> probabilities)
    proba, per_trip_ms = np.empty(len(te)), []
    for tid, idx in te.groupby("trip_id", sort=False).indices.items():
        Xt = te.iloc[idx][feats].to_numpy(np.float32)
        t0 = time.perf_counter()
        proba[idx] = _proba(final, Xt)
        per_trip_ms.append((time.perf_counter() - t0) * 1000)
    yt = te["label"].to_numpy()
    st = candidate_metrics(yt, proba)
    best = argmax_arcs(te, proba)
    cs = pd.read_csv(repo_path("experiments/results/candidate_stats.csv"), dtype={"trip_id": str})
    cs = cs[cs.partition == "student_test"]
    miss = float((cs.miss_rate_reliable * cs.n_reliable).sum() / cs.n_reliable.sum())

    cv_f1 = [f["macro_f1"] for f in folds]
    out = {
        "name": name, "family": family, "config": config, "seed": SEED,
        "features": feats, "n_features": len(feats),
        "data": {"candidates": load_config("candidates"), "train": tr_info, "student_test": te_info,
                 "train_pos_rate": float(y.mean()), "max_train_obs": max_train_obs,
                 "train_rows_used_final_fit": int(len(full_idx))},
        "cv": {"scheme": "GroupKFold(n_splits=5, shuffle=True, random_state=148) by trip_id on 80 train trips",
               "macro_f1_folds": cv_f1, "macro_f1_mean": float(np.mean(cv_f1)), "macro_f1_std": float(np.std(cv_f1)),
               "top1_mean": float(np.mean([f["top1"] for f in folds])),
               "top1_std": float(np.std([f["top1"] for f in folds])),
               "log_loss_mean": float(np.mean([f["log_loss"] for f in folds])),
               "train_macro_f1_mean": float(np.mean([f["train_macro_f1"] for f in folds])) if train_score else None,
               "folds": folds},
        "student_test": {"top1": top1(te, proba, te_info["n_obs_used"]), **st,
                         "confusion_matrix": confusion_matrix(yt, (proba >= 0.5).astype(int)).tolist(),
                         "route_continuity_argmax": route_continuity(best, "student_test"),
                         "candidate_miss_rate": miss, "top1_upper_bound": 1 - miss},
        "route_continuity_note": "classifier-only argmax per observation; the HMM decoder is evaluated separately",
        "fit_seconds": fit_s,
        "inference_ms_per_trip": float(np.mean(per_trip_ms)),
        "inference_ms_per_point": float(np.sum(per_trip_ms) / te_info["n_obs_with_candidates"]),
        "inference_note": "predict_proba on prepared feature rows; feature building adds ~1-20 ms per trip",
        "total_seconds": time.perf_counter() - t_all,
    }
    path = repo_path(f"experiments/results/models/{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    pred_path = repo_path(f"data/processed_local/preds/{name}_student_test.parquet")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    te[["trip_id", "obs_idx", "arc_id", "label"]].assign(proba=proba.astype(np.float32)).to_parquet(pred_path, index=False)
    if verbose:
        print(f"{name}: CV macro F1 {out['cv']['macro_f1_mean']:.4f} +/- {out['cv']['macro_f1_std']:.4f}, "
              f"CV Top-1 {out['cv']['top1_mean']:.4f} | student_test Top-1 {out['student_test']['top1']:.4f}, "
              f"macro F1 {st['macro_f1']:.4f}, log loss {st['log_loss']:.4f} | fit {fit_s:.1f} s, "
              f"{out['inference_ms_per_trip']:.1f} ms/trip -> {path.relative_to(repo_path('.'))}")
    return out


def save_cv_folds() -> str:
    tr = _load("train")
    folds = cv_folds(tr["trip_id"])
    path = repo_path("experiments/results/cv_folds.csv")
    pd.DataFrame(sorted(folds.items()), columns=["trip_id", "fold"]).to_csv(path, index=False)
    return str(path)
