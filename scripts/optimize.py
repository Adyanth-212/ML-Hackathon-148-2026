"""Bayesian hyperparameter optimisation of LightGBM with Optuna (TPE), configs/bayes_opt.yaml.

    python scripts/optimize.py               # run (or resume) the study
    python scripts/optimize.py --report      # only rewrite the CSV / JSON / figures from the saved study

Objective: mean cross-validated observation-level Top-1 on the 80 train trips, using the
harness's fixed trip folds (pathsnap.harness.cv_folds) and the harness's Top-1 definition.
student_test and the evaluator partition are not touched here. After the search, the best
parameters go to scripts/models/advitha_lgbm_tuned.py, which runs through the normal harness.

Writes (experiments/results/bayes_opt/):
  study.db           Optuna SQLite storage: every trial, its parameters, per-fold values, state
  trials.csv         one row per trial: params, CV Top-1, CV macro F1, CV log loss, fold values, time
  best_params.json   the best configuration (read by scripts/models/advitha_lgbm_tuned.py)
  summary.json       search space, sampler, n trials (complete / pruned), wall time, best value
and experiments/figures/bayes_opt_history.png, bayes_opt_importance.png.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier

from pathsnap.config import load_config, repo_path
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.harness import SEED, candidate_metrics, cv_folds, load_split, top1

OUT = repo_path("experiments/results/bayes_opt")
FIG = repo_path("experiments/figures")
INK, MUTED, BAR, GREY = "#1f2328", "#6e7781", "#2f6fb0", "#afb8c1"


def suggest(trial: optuna.Trial, space: dict) -> dict:
    p = {}
    for name, s in space.items():
        if s["type"] == "int":
            p[name] = trial.suggest_int(name, s["low"], s["high"], step=s.get("step", 1), log=s.get("log", False))
        else:
            p[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
    return p


class CVData:
    """Train features, labels and folds, loaded once and shared by every trial."""

    def __init__(self):
        tr, _ = load_split("train")
        self.df = tr
        self.X = tr[FEATURE_COLUMNS].to_numpy(np.float32)
        self.y = tr["label"].to_numpy()
        fold_of = cv_folds(tr["trip_id"])
        self.fold = tr["trip_id"].map(fold_of).to_numpy()
        # same Top-1 denominator as the harness: every reliable observation of the fold's trips
        lab_n = {p.stem: int((~pd.read_parquet(p).is_outlier).sum())
                 for p in repo_path("data/processed/labels/train").glob("*.parquet")}
        self.splits = []
        for k in sorted(set(self.fold)):
            va = np.flatnonzero(self.fold == k)
            trips = tr.iloc[va].trip_id.unique()
            self.splits.append((k, np.flatnonzero(self.fold != k), va, sum(lab_n[t] for t in trips)))


def objective_factory(data: CVData, cfg: dict):
    def objective(trial: optuna.Trial) -> float:
        params = {**cfg["fixed"], **suggest(trial, cfg["search_space"])}
        t0 = time.perf_counter()
        top, f1, ll = [], [], []
        for step, (k, tr_idx, va_idx, n_obs) in enumerate(data.splits):
            est = LGBMClassifier(**params, random_state=SEED, n_jobs=-1, verbose=-1)
            est.fit(data.X[tr_idx], data.y[tr_idx])
            p = est.predict_proba(data.X[va_idx])[:, 1]
            m = candidate_metrics(data.y[va_idx], p)
            top.append(top1(data.df.iloc[va_idx], p, n_obs))
            f1.append(m["macro_f1"])
            ll.append(m["log_loss"])
            trial.set_user_attr("folds_done", step + 1)
            trial.set_user_attr("fold_top1", top)
            trial.set_user_attr("fold_macro_f1", f1)
            trial.set_user_attr("fold_log_loss", ll)
            trial.set_user_attr("seconds", time.perf_counter() - t0)
            trial.report(float(np.mean(top)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_macro_f1_mean", float(np.mean(f1)))
        trial.set_user_attr("cv_log_loss_mean", float(np.mean(ll)))
        trial.set_user_attr("cv_top1_std", float(np.std(top)))
        return float(np.mean(top))
    return objective


def make_study(cfg: dict) -> optuna.Study:
    s, pr = cfg["sampler"], cfg["pruner"]
    OUT.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=cfg["study_name"], storage=f"sqlite:///{OUT / 'study.db'}", load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=s["seed"], n_startup_trials=s["n_startup_trials"],
                                           multivariate=s.get("multivariate", False)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=pr["n_startup_trials"], n_warmup_steps=pr["n_warmup_steps"]))


def baseline_params(cfg: dict) -> dict:
    """The untuned advitha_lgbm configuration, restricted to the searched parameters.

    reg_alpha is 0 in the untuned model (LightGBM default); 0.001 is used as its stand-in
    because the search range is on a log scale. Its effect at this size is nil."""
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("advitha_lgbm", repo_path("scripts/models/advitha_lgbm.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = {**mod.CONFIG, "reg_alpha": cfg["search_space"]["reg_alpha"]["low"]}
    return {k: base[k] for k in cfg["search_space"]}


def report(study: optuna.Study, cfg: dict, wall_s: float | None) -> None:
    rows = []
    for t in study.trials:
        ua = t.user_attrs
        rows.append({"trial": t.number, "state": t.state.name, "cv_top1_mean": t.value,
                     "cv_top1_running_mean_last": (list(t.intermediate_values.values()) or [None])[-1],
                     "folds_done": ua.get("folds_done"), "cv_macro_f1_mean": ua.get("cv_macro_f1_mean"),
                     "cv_log_loss_mean": ua.get("cv_log_loss_mean"), "cv_top1_std": ua.get("cv_top1_std"),
                     "seconds": ua.get("seconds"),
                     **{f"param_{k}": v for k, v in t.params.items()},
                     "fold_top1": json.dumps([round(v, 5) for v in ua.get("fold_top1", [])]),
                     "fold_macro_f1": json.dumps([round(v, 5) for v in ua.get("fold_macro_f1", [])]),
                     "is_baseline": t.number == 0 and cfg.get("enqueue_baseline", False),
                     "started": t.datetime_start.isoformat(timespec="seconds") if t.datetime_start else None})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "trials.csv", index=False)

    done = df[df.state == "COMPLETE"]
    best = study.best_trial
    params = {**cfg["fixed"], **best.params}
    (OUT / "best_params.json").write_text(json.dumps(
        {"study": cfg["study_name"], "best_trial": best.number, "cv_top1_mean": best.value,
         "cv_macro_f1_mean": best.user_attrs.get("cv_macro_f1_mean"),
         "cv_log_loss_mean": best.user_attrs.get("cv_log_loss_mean"), "params": params}, indent=2))
    prev = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    total_s = float(df.seconds.fillna(0).sum())
    summary = {
        "objective": "maximise mean CV observation-level Top-1 (train, GroupKFold(5) by trip, harness folds)",
        "model": "LightGBM (LGBMClassifier)", "n_features": len(FEATURE_COLUMNS),
        "sampler": cfg["sampler"], "pruner": cfg["pruner"], "search_space": cfg["search_space"], "fixed": cfg["fixed"],
        "n_trials": int(len(df)), "n_complete": int(len(done)), "n_pruned": int((df.state == "PRUNED").sum()),
        "n_failed": int((df.state == "FAIL").sum()),
        "trial_compute_seconds_total": total_s,
        "wall_seconds_last_run": wall_s if wall_s is not None else prev.get("wall_seconds_last_run"),
        "baseline_trial_cv_top1": float(df.loc[0, "cv_top1_mean"]) if cfg.get("enqueue_baseline") and len(df) else None,
        "best_trial": best.number, "best_cv_top1": best.value, "best_params": params,
        "hardware": f"{__import__('os').cpu_count()} CPU threads, n_jobs=-1",
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # history: every trial's value, best-so-far line, pruned trials at their last running mean
    fig, ax = plt.subplots(figsize=(8, 4))
    pr = df[df.state == "PRUNED"]
    ax.scatter(pr.trial, pr.cv_top1_running_mean_last, s=18, color=GREY, label="pruned (running mean at stop)")
    ax.scatter(done.trial, done.cv_top1_mean, s=22, color=BAR, label="complete (5-fold mean)")
    ax.plot(done.trial, done.cv_top1_mean.cummax(), color=INK, lw=1.2, drawstyle="steps-post", label="best so far")
    if cfg.get("enqueue_baseline"):
        ax.axhline(df.loc[0, "cv_top1_mean"], color=MUTED, lw=0.8, ls="--", label="untuned (trial 0)")
    ax.set_xlabel("trial", color=INK)
    ax.set_ylabel("CV Top-1 (TRAIN, 5 folds)", color=INK)
    ax.set_title(f"Optuna TPE on LightGBM: {len(done)} complete, {len(pr)} pruned", loc="left", fontsize=10, color=INK)
    lo = np.nanpercentile(pd.concat([done.cv_top1_mean, pr.cv_top1_running_mean_last]).astype(float), 5)
    ax.set_ylim(lo - 0.002, done.cv_top1_mean.max() + 0.002)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#d0d7de", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "bayes_opt_history.png", dpi=140)
    plt.close(fig)

    if len(done) >= 5:
        imp = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=SEED))
        (OUT / "param_importance.json").write_text(json.dumps(imp, indent=2))
        names = list(imp)[::-1]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.barh(names, [imp[n] for n in names], color=BAR, height=0.6)
        for y, n in enumerate(names):
            ax.text(imp[n], y, f"  {imp[n]:.2f}", va="center", fontsize=8, color=INK)
        ax.set_title("Hyperparameter importance (fANOVA on complete trials)", loc="left", fontsize=10, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK, labelsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "bayes_opt_importance.png", dpi=140)
        plt.close(fig)
    print(f"{len(df)} trials ({len(done)} complete, {len(pr)} pruned); best #{best.number} CV Top-1 {best.value:.4f}")
    print(json.dumps(params, indent=2))


def main(argv: list[str]) -> int:
    cfg = load_config("bayes_opt")
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = make_study(cfg)
    wall = None
    if "--report" not in argv:
        if cfg.get("enqueue_baseline") and not study.trials:
            study.enqueue_trial(baseline_params(cfg), user_attrs={"baseline": True})
        remaining = cfg["n_trials"] - sum(t.state.is_finished() for t in study.trials)
        if remaining > 0:
            data = CVData()
            t0 = time.perf_counter()
            study.optimize(objective_factory(data, cfg), n_trials=remaining,
                           timeout=cfg["timeout_minutes"] * 60, gc_after_trial=True)
            wall = time.perf_counter() - t0
    report(study, cfg, wall)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
