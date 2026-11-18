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
import numpy as np
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
    plt.close(fig)
    results = {d["name"]: d for d in (json.loads(p.read_text())
                                      for p in sorted(repo_path("experiments/results/models").glob("*.json")))}
    models = [m for m in df[df.cv_macro_f1_mean.notna() & ~df.is_ablation.astype(bool)].model]
    plot_cv_f1(models, results)
    plot_confusion(models, results)
    return 0


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_cv_f1(models: list[str], results: dict) -> None:
    """CV macro F1 per model: large dot = mean, whisker = std over the 5 folds, small dots = each fold.
    (A dot plot, not bars, because the axis does not start at 0.)"""
    order = sorted(models, key=lambda n: results[n]["cv"]["macro_f1_mean"])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(order) + 1.4))
    for y, n in enumerate(order):
        cv = results[n]["cv"]
        ax.scatter(cv["macro_f1_folds"], [y] * len(cv["macro_f1_folds"]), s=12, color=ABL, zorder=2)
        ax.errorbar(cv["macro_f1_mean"], y, xerr=cv["macro_f1_std"], fmt="o", color=BAR, ms=7,
                    ecolor=INK, elinewidth=1, capsize=3, zorder=3)
        ax.text(max(cv["macro_f1_folds"]) + 0.004, y, f"{cv['macro_f1_mean']:.3f} ± {cv['macro_f1_std']:.3f}",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(order)), order)
    ax.tick_params(axis="y", colors=INK, labelsize=9)
    lo = min(min(results[n]["cv"]["macro_f1_folds"]) for n in order)
    ax.set_xlim(max(0, lo - 0.03), 1.0)
    ax.set_xlabel("macro F1 at threshold 0.5 (candidate level)", color=MUTED, fontsize=8)
    ax.set_title("CV macro F1, TRAIN, GroupKFold(5) by trip: mean ± std (blue), each fold (light)", loc="left",
                 fontsize=10, color=INK)
    ax.grid(axis="x", color="#d0d7de", lw=0.6)
    ax.set_axisbelow(True)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(repo_path("experiments/figures/cv_macro_f1.png"), dpi=140)
    plt.close(fig)


def plot_confusion(models: list[str], results: dict) -> None:
    """Student-test confusion matrices (threshold 0.5), row-normalised colour, counts printed."""
    order = sorted(models, key=lambda n: -results[n]["student_test"]["top1"])
    cols = min(4, len(order))
    rows = -(-len(order) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.0 * rows), squeeze=False)
    for ax in axs.flat[len(order):]:
        ax.axis("off")
    for ax, n in zip(axs.flat, order):
        cm = np.array(results[n]["student_test"]["confusion_matrix"], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j]):,}\n{norm[i, j]:.1%}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.tick_params(labelsize=8, colors=MUTED, length=0)
        ax.set_title(f"{n}\nTop-1 {results[n]['student_test']['top1']:.3f}", fontsize=9, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Confusion matrices, STUDENT_TEST candidates, threshold 0.5 (row-normalised; 1 = true arc)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(repo_path("experiments/figures/confusion_matrices.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
