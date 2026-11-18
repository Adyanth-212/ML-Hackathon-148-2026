"""Collect every model result (experiments/results/models/*.json) into one comparison table.

    python scripts/cv_compare.py

Writes experiments/results/model_comparison.csv and experiments/figures/model_comparison.png
(small multiples, one measure per panel, models sorted by student_test Top-1). Rule-based
baselines from scripts/eval_baseline.py are listed for reference but are not models.
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pathsnap.config import repo_path

INK, MUTED, BAR, ABL = "#1f2328", "#6e7781", "#2f6fb0", "#9fb9d6"


def main() -> int:
    rows = []
    for p in sorted(repo_path("experiments/results/models").glob("*.json")):
        d = json.loads(p.read_text())
        s, cv = d["student_test"], d["cv"]
        rows.append({
            "model": d["name"], "family": d["family"],
            "key_config": ", ".join(f"{k}={v}" for k, v in d["config"].items() if k not in ("feature_groups",))[:120],
            "n_features": d["n_features"],
            "cv_macro_f1_mean": cv["macro_f1_mean"], "cv_macro_f1_std": cv["macro_f1_std"],
            "cv_top1_mean": cv["top1_mean"], "cv_top1_std": cv.get("top1_std"),
            "cv_log_loss_mean": cv.get("log_loss_mean"), "train_macro_f1_mean": cv.get("train_macro_f1_mean"),
            "student_test_top1": s["top1"], "student_test_macro_f1": s["macro_f1"],
            "student_test_precision_macro": s["precision_macro"], "student_test_recall_macro": s["recall_macro"],
            "student_test_log_loss": s["log_loss"],
            "route_continuity_argmax": s.get("route_continuity_argmax"),
            "candidate_miss_rate": s["candidate_miss_rate"],
            "fit_seconds": d["fit_seconds"], "inference_ms_per_trip": d["inference_ms_per_trip"],
            "inference_ms_per_point": d["inference_ms_per_point"],
            "is_ablation": d["name"].startswith("ablation_"), "seed": d["seed"],
        })
    df = pd.DataFrame(rows).sort_values("student_test_top1", ascending=False)

    base = repo_path("experiments/results/baseline_heading_sweep.csv")
    if base.exists():
        b = pd.read_csv(base)
        b = b[b.partition == "student_test"].set_index("heading_weight_m").top1
        extra = [{"model": "rule_nearest_arc", "family": "rule (not ML)", "key_config": "heading_weight_m=0",
                  "student_test_top1": b.get(0)},
                 {"model": "rule_nearest_heading", "family": "rule (not ML)", "key_config": "heading_weight_m=20 (chosen on train)",
                  "student_test_top1": b.get(20)}]
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    df.to_csv(repo_path("experiments/results/model_comparison.csv"), index=False)
    show = ["model", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean", "student_test_top1",
            "student_test_macro_f1", "student_test_log_loss", "route_continuity_argmax", "fit_seconds",
            "inference_ms_per_trip"]
    print(df[show].round(4).to_string(index=False))

    m = df[df.cv_macro_f1_mean.notna()].iloc[::-1]  # best at the top of horizontal bars
    colors = [ABL if a else BAR for a in m.is_ablation]
    fig, axs = plt.subplots(1, 3, figsize=(15, 0.45 * len(m) + 1.8), sharey=True)
    panels = [("cv_macro_f1_mean", "cv_macro_f1_std", "CV macro F1 (mean ± std, 5 folds, TRAIN)"),
              ("student_test_top1", None, "Top-1 accuracy (STUDENT_TEST)"),
              ("student_test_log_loss", None, "Log loss (STUDENT_TEST, lower is better)")]
    for ax, (col, err, title) in zip(axs, panels):
        ax.barh(m.model, m[col], xerr=m[err] if err else None, color=colors, height=0.6,
                error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
        ends = m[col] + (m[err] if err else 0)
        for y, (v, e) in enumerate(zip(m[col], ends)):
            ax.text(e, y, f"  {v:.3f}", va="center", fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.grid(axis="x", color="#d0d7de", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    axs[0].tick_params(axis="y", colors=INK, labelsize=9)
    for ax in axs[:2]:
        ax.set_xlim(0, 1.08)
    fig.suptitle("Model comparison (light bars = feature-group ablations)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(repo_path("experiments/figures/model_comparison.png"), dpi=140)
    return 0


if __name__ == "__main__":
    sys.exit(main())
