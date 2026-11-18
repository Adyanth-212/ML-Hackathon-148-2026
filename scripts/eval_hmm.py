"""Tune the HMM decoder on train and compare classifier-only vs classifier + HMM on student_test.

    python scripts/eval_hmm.py              # tune on train (out-of-fold scores), evaluate, write results
    python scripts/eval_hmm.py --no-write-config   # same, but leave configs/hmm.yaml unchanged

Classifier: the exported model (artifacts/metadata.json, currently the tuned LightGBM).

1. Train scores are OUT-OF-FOLD: for each of the harness's 5 trip folds the classifier is refitted
   on the other 4 folds and scores the held-out trips. In-sample scores would be over-confident
   and would bias the decoder towards trusting the classifier.
2. Grid over classifier_weight x beta_m (configs/hmm.yaml) on all 80 train trips; the objective
   is observation-level Top-1 on reliable points. The best setting is written to configs/hmm.yaml.
3. student_test (final exported model, never used for any choice): classifier argmax, HMM with the
   old placeholder config, distance-only HMM, and the tuned HMM. Top-1, route continuity, route
   breaks, sequence precision/recall/F1 against the reference route, decode time.

The evaluator partition is never read (pathsnap.io refuses without PATHSNAP_MODE=serve).

Writes experiments/results/hmm_grid.csv, hmm_comparison.csv, hmm_student_test.csv (per trip)
and experiments/figures/hmm_comparison.png.
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from export_artifacts import build_from_script  # noqa: E402

from pathsnap import hmm  # noqa: E402
from pathsnap.candidates import generate_candidates  # noqa: E402
from pathsnap.config import CONFIG_DIR, repo_path  # noqa: E402
from pathsnap.features import FEATURE_COLUMNS  # noqa: E402
from pathsnap.geo import TripGeometry  # noqa: E402
from pathsnap.harness import cv_folds, load_split  # noqa: E402
from pathsnap.io import discover_trips, load_trip  # noqa: E402
from pathsnap.metrics import route_continuity, sequence_metrics  # noqa: E402
from pathsnap.pipeline import argmax_per_obs, classifier_scorer  # noqa: E402

GRID = {"classifier_weight": [0.5, 0.75, 1.0], "beta_m": [5.0, 10.0, 20.0, 50.0]}
N_JOBS = max(1, (os.cpu_count() or 2) - 1)
INK, MUTED, BLUE, ORANGE = "#1f2328", "#6e7781", "#2f6fb0", "#d9822b"


def cfg_with(base: dict, w: float | None = None, beta: float | None = None) -> dict:
    c = copy.deepcopy(base)
    if w is not None:
        c["emission"]["classifier_weight"] = float(w)
    if beta is not None:
        c["transition"]["beta_m"] = float(beta)
    return c


def prepare_trip(partition: str, files, scorer) -> dict:
    trip = load_trip(files)  # no route: the scorer and decoder never see it
    geom = TripGeometry.from_trip(trip)
    route = load_trip(files, with_route=True).route  # for scoring only, after decoding
    cands, _ = generate_candidates(geom, trip.trip_id)
    cands = cands.reset_index(drop=True)
    lab = pd.read_parquet(repo_path(f"data/processed/labels/{partition}/{trip.trip_id}.parquet"))
    return {"trip_id": trip.trip_id, "geom": geom, "cands": cands,
            "scores": np.asarray(scorer(geom, cands)), "label": lab.arc_id.to_numpy(),
            "ok": ~lab.is_outlier.to_numpy(), "arcs": trip.arcs[["from_node", "to_node"]],
            "route": None if route is None else np.asarray(route).ravel()}


@lru_cache(maxsize=8)
def _load(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def decode(path: str, cfg: dict | None) -> tuple[np.ndarray, float]:
    """cfg None = classifier argmax; otherwise the HMM. Returns (pred, decode ms)."""
    d = _load(path)
    t0 = time.perf_counter()
    if cfg is None:
        pred = argmax_per_obs(d["cands"], d["scores"], d["geom"].n_obs)
    else:
        pred = hmm.decode_candidates(d["geom"], d["cands"], d["scores"], cfg)[0]
    return np.asarray(pred), (time.perf_counter() - t0) * 1000


def score(path: str, pred: np.ndarray, ms: float) -> dict:
    d = _load(path)
    ok = d["ok"]
    p = pred[pred >= 0]
    seq = p[np.r_[True, p[1:] != p[:-1]]] if len(p) else p
    breaks = int(np.sum(d["arcs"].to_node.to_numpy()[p[:-1]] != d["arcs"].from_node.to_numpy()[p[1:]])) if len(p) > 1 else 0
    out = {"trip_id": d["trip_id"], "n_obs": int(ok.sum()), "n_correct": int((pred == d["label"])[ok].sum()),
           "continuity": route_continuity(pred, d["arcs"]), "breaks": breaks, "decode_ms": ms}
    if d["route"] is not None:
        out["seq"] = sequence_metrics([d["route"]], [seq])
    return out


def run(paths: list[str], cfgs: dict[str, dict | None], desc: str) -> pd.DataFrame:
    tasks = [(key, p) for key in cfgs for p in paths]
    # biggest trips first so the progress bar does not stall at the end
    tasks.sort(key=lambda t: -os.path.getsize(t[1]))

    def job(key, p):
        pred, ms = decode(p, cfgs[key])
        return {"variant": key, **score(p, pred, ms)}

    rows = []
    gen = Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(delayed(job)(k, p) for k, p in tasks)
    for r in tqdm(gen, total=len(tasks), desc=desc, unit="trip", dynamic_ncols=True):
        rows.append(r)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("variant", sort=False)
    out = pd.DataFrame({"top1": g.n_correct.sum() / g.n_obs.sum(), "n_correct": g.n_correct.sum(),
                        "n_obs": g.n_obs.sum(), "continuity_mean": g.continuity.mean(), "breaks_total": g.breaks.sum(),
                        "decode_ms_mean": g.decode_ms.mean(), "decode_ms_max": g.decode_ms.max()})
    if "seq" in df:
        for k in ("precision", "recall", "f1"):
            out[f"route_{k}_mean"] = df.assign(v=df.seq.map(lambda s: s[k])).groupby("variant", sort=False).v.mean()
    return out


def main(argv: list[str]) -> int:
    t_all = time.perf_counter()
    base = hmm.load_hmm_config()
    meta = json.loads(repo_path("artifacts/metadata.json").read_text())
    name = meta["model"]
    tmp = Path(tempfile.mkdtemp(prefix="hmm_eval_"))

    # 1. out-of-fold train scores
    tr, _ = load_split("train")
    fold_of = cv_folds(tr["trip_id"])
    trips = discover_trips("train")
    train_paths = []
    folds = sorted(set(fold_of.values()))
    with tqdm(total=len(folds) + len(trips), desc=f"OOF scores ({name})", unit="step", dynamic_ncols=True) as bar:
        for k in folds:
            bar.set_postfix_str(f"fit without fold {k}")
            est, _ = build_from_script(name)
            m = tr.trip_id.map(fold_of).to_numpy() != k
            est.fit(tr.loc[m, FEATURE_COLUMNS].to_numpy(np.float32), tr.loc[m, "label"].to_numpy())
            bar.update(1)
            scorer = classifier_scorer(est)
            for tid in sorted(t for t, f in fold_of.items() if f == k):
                bar.set_postfix_str(f"score trip {tid}")
                p = tmp / f"train_{tid}.pkl"
                p.write_bytes(pickle.dumps(prepare_trip("train", trips[tid], scorer)))
                train_paths.append(str(p))
                bar.update(1)

    # 2. grid on train
    cfgs = {f"w={w:g},beta={b:g}": cfg_with(base, w, b) for w in GRID["classifier_weight"] for b in GRID["beta_m"]}
    cfgs["argmax"] = None
    tr_res = run(train_paths, cfgs, "HMM grid on 80 train trips")
    grid = summarise(tr_res).reset_index()
    grid.to_csv(repo_path("experiments/results/hmm_grid.csv"), index=False)
    hmm_rows = grid[grid.variant != "argmax"]
    best = hmm_rows.sort_values(["top1", "continuity_mean"], ascending=False).iloc[0]
    bw, bb = (float(x.split("=")[1]) for x in best.variant.split(","))
    print("\nGrid (train, out-of-fold):\n" + grid.round(4).to_string(index=False))
    print(f"\nBest on train: classifier_weight={bw:g}, beta_m={bb:g}, Top-1 {best.top1:.4f}")

    # 3. student_test with the exported (final) model
    model = joblib.load(repo_path("artifacts/model.joblib"))
    scorer = classifier_scorer(model)
    st_paths = []
    for tid, f in tqdm(sorted(discover_trips("student_test").items()), desc="score student_test", unit="trip"):
        p = tmp / f"st_{tid}.pkl"
        p.write_bytes(pickle.dumps(prepare_trip("student_test", f, scorer)))
        st_paths.append(str(p))
    variants = {"classifier argmax (no HMM)": None,
                f"HMM placeholder config (w={base['emission']['classifier_weight']:g}, beta={base['transition']['beta_m']:g})": base,
                f"HMM distance-only (w=0, beta={bb:g})": cfg_with(base, 0.0, bb),
                f"HMM tuned on train (w={bw:g}, beta={bb:g})": cfg_with(base, bw, bb)}
    st = run(st_paths, variants, "student_test variants")
    comp = summarise(st).reindex(list(variants)).reset_index()
    comp.to_csv(repo_path("experiments/results/hmm_comparison.csv"), index=False)
    per_trip = st.drop(columns=["seq"]).assign(
        route_f1=st.seq.map(lambda s: s["f1"]), top1=st.n_correct / st.n_obs)
    per_trip = per_trip.sort_values(["trip_id", "variant"])
    per_trip.to_csv(repo_path("experiments/results/hmm_student_test.csv"), index=False)
    print("\nstudent_test:\n" + comp.round(4).to_string(index=False))
    piv = per_trip.pivot_table(index="trip_id", columns="variant", values="top1")[list(variants)]
    print("\nper-trip Top-1:\n" + piv.round(3).to_string())

    # figure: per-trip Top-1, argmax vs tuned HMM
    a, h = list(variants)[0], list(variants)[-1]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(piv))
    ax.scatter(x, piv[a], color=MUTED, s=30, label="classifier argmax", zorder=3)
    ax.scatter(x, piv[h], color=BLUE, s=30, label="classifier + HMM (tuned on train)", zorder=3)
    for i in x:
        ax.plot([i, i], [piv[a].iloc[i], piv[h].iloc[i]], color=ORANGE if piv[h].iloc[i] < piv[a].iloc[i] else "#9fb9d6", lw=2)
    ax.set_xticks(x, piv.index, rotation=45, fontsize=8)
    ax.set_ylabel("Top-1 (STUDENT_TEST)", color=INK)
    ax.set_title("Per-trip Top-1: classifier only vs + HMM (orange = HMM worse)", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", color="#d0d7de", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(repo_path("experiments/figures/hmm_comparison.png"), dpi=140)

    if "--no-write-config" not in argv:
        text = (CONFIG_DIR / "hmm.yaml").read_text()
        cfg = yaml.safe_load(text)
        cfg["emission"]["classifier_weight"], cfg["transition"]["beta_m"] = bw, bb
        header = ("# HMM / Viterbi decoder parameters. classifier_weight and beta_m were tuned by\n"
                  "# scripts/eval_hmm.py: grid on the 80 TRAIN trips with out-of-fold classifier scores\n"
                  "# (experiments/results/hmm_grid.csv). Never tuned on evaluator trips.\n")
        (CONFIG_DIR / "hmm.yaml").write_text(header + yaml.safe_dump(cfg, sort_keys=False))
        print(f"configs/hmm.yaml updated: classifier_weight={bw:g}, beta_m={bb:g}")
    print(f"total {(time.perf_counter() - t_all) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
