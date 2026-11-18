"""Untuned vs Bayesian-tuned LightGBM, same partitions, folds, features and seed.

    python scripts/compare_tuning.py

Reads experiments/results/models/advitha_lgbm.json (untuned) and advitha_lgbm_tuned.json, and
experiments/results/bayes_opt/summary.json (search cost). Writes
experiments/results/tuning_comparison.csv.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from pathsnap.config import repo_path

ROWS = [  # (label, getter, partition)
    ("CV macro F1 (mean)", lambda d: d["cv"]["macro_f1_mean"], "train CV"),
    ("CV macro F1 (std over folds)", lambda d: d["cv"]["macro_f1_std"], "train CV"),
    ("CV Top-1 (mean)", lambda d: d["cv"]["top1_mean"], "train CV"),
    ("CV log loss (mean)", lambda d: d["cv"]["log_loss_mean"], "train CV"),
    ("train-fold macro F1 (mean)", lambda d: d["cv"]["train_macro_f1_mean"], "train CV"),
    ("Top-1", lambda d: d["student_test"]["top1"], "student_test"),
    ("macro F1", lambda d: d["student_test"]["macro_f1"], "student_test"),
    ("log loss", lambda d: d["student_test"]["log_loss"], "student_test"),
    ("route continuity (argmax)", lambda d: d["student_test"]["route_continuity_argmax"], "student_test"),
    ("final fit time (s)", lambda d: d["fit_seconds"], "train"),
    ("inference (ms per trip)", lambda d: d["inference_ms_per_trip"], "student_test"),
]


def main() -> int:
    res = repo_path("experiments/results")
    a = json.loads((res / "models/advitha_lgbm.json").read_text())
    b = json.loads((res / "models/advitha_lgbm_tuned.json").read_text())
    rows = [{"metric": m, "partition": part, "untuned": f(a), "tuned": f(b), "change": f(b) - f(a)}
            for m, f, part in ROWS]
    df = pd.DataFrame(rows)
    df.to_csv(res / "tuning_comparison.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    s = json.loads((res / "bayes_opt/summary.json").read_text())
    print(f"\nSearch cost: {s['n_trials']} trials ({s['n_complete']} complete, {s['n_pruned']} pruned), "
          f"{s['trial_compute_seconds_total'] / 60:.1f} min of trial compute, {s['hardware']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
