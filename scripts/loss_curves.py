"""Train/validation loss curves for the models that have a training history.

    python scripts/loss_curves.py

Only two kinds of model in this project are trained iteratively:
- boosting (LightGBM, XGBoost) adds one tree per iteration;
- SGD makes one pass over the data per epoch.
Random Forest, Extra Trees and the Decision Tree are grown once, Logistic Regression is solved by
L-BFGS to convergence and Gaussian NB is fitted in closed form, so they have no loss history and are
not plotted.

For each model we hold out CV fold 0 of the shared harness folds (whole trips), fit on the other four
folds with the same configuration as in scripts/models/, and record the binary log loss on the
training and the validation rows after every iteration (boosting) or epoch (SGD). SGD is fitted with
partial_fit, one shuffled epoch at a time, with the balanced class weights passed as sample weights
(partial_fit does not accept class_weight='balanced'); its log loss is the unweighted one.

Writes experiments/results/loss_curves.csv, loss_curves_summary.json and
experiments/figures/loss_curves.png.
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from pathsnap.config import repo_path
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.harness import SEED, cv_folds, load_split

sys.path.insert(0, str(repo_path("scripts/models")))
import advitha_lgbm  # noqa: E402
import advitha_lgbm_tuned  # noqa: E402
import advitha_xgb  # noqa: E402
import adyanth_sgd  # noqa: E402

VAL_FOLD = 0
SGD_EPOCHS = 30


def boosting_curve(est, Xa, ya, Xb, yb):
    if isinstance(est, LGBMClassifier):
        est.fit(Xa, ya, eval_set=[(Xa, ya), (Xb, yb)], eval_metric="binary_logloss")
        tr_l, va_l = (v["binary_logloss"] for v in est.evals_result_.values())
    else:
        est.fit(Xa, ya, eval_set=[(Xa, ya), (Xb, yb)], verbose=False)
        r = est.evals_result()
        tr_l, va_l = r["validation_0"]["logloss"], r["validation_1"]["logloss"]
    return list(tr_l), list(va_l)


def sgd_curve(Xa, ya, Xb, yb):
    c = adyanth_sgd.CONFIG
    scaler = StandardScaler().fit(Xa)
    Xa, Xb = scaler.transform(Xa), scaler.transform(Xb)
    sw = compute_sample_weight("balanced", ya)
    clf = SGDClassifier(loss=c["loss"], penalty=c["penalty"], alpha=c["alpha"], learning_rate=c["learning_rate"],
                        random_state=SEED)
    rng = np.random.default_rng(SEED)
    tr_l, va_l = [], []
    for _ in range(SGD_EPOCHS):
        order = rng.permutation(len(ya))
        clf.partial_fit(Xa[order], ya[order], classes=np.array([0, 1]), sample_weight=sw[order])
        tr_l.append(log_loss(ya, clf.predict_proba(Xa)[:, 1]))
        va_l.append(log_loss(yb, clf.predict_proba(Xb)[:, 1]))
    return tr_l, va_l


def main() -> int:
    tr, _ = load_split("train")
    X, y = tr[FEATURE_COLUMNS].to_numpy(np.float32), tr["label"].to_numpy()
    fold = tr["trip_id"].map(cv_folds(tr["trip_id"])).to_numpy()
    a, b = fold != VAL_FOLD, fold == VAL_FOLD
    print(f"fit rows {a.sum():,}, validation rows {b.sum():,} (fold {VAL_FOLD}, {tr.trip_id[b].nunique()} trips)")

    runs = {
        "LightGBM (Bayesian-tuned)": lambda: boosting_curve(
            LGBMClassifier(**advitha_lgbm_tuned.CONFIG, random_state=SEED, n_jobs=-1, verbose=-1), X[a], y[a], X[b], y[b]),
        "LightGBM (untuned)": lambda: boosting_curve(
            LGBMClassifier(**advitha_lgbm.CONFIG, random_state=SEED, n_jobs=-1, verbose=-1), X[a], y[a], X[b], y[b]),
        "XGBoost": lambda: boosting_curve(
            XGBClassifier(**advitha_xgb.CONFIG, random_state=SEED, n_jobs=-1), X[a], y[a], X[b], y[b]),
        "SGD (logistic loss)": lambda: sgd_curve(X[a], y[a], X[b], y[b]),
    }
    rows, summary = [], {}
    for name, run in runs.items():
        tr_l, va_l = run()
        best = int(np.argmin(va_l))
        unit = "epoch" if name.startswith("SGD") else "iteration"
        summary[name] = {"unit": unit, "n": len(va_l), "final_train": tr_l[-1], "final_val": va_l[-1],
                         "best_val": va_l[best], "best_at": best + 1}
        print(f"{name}: final train {tr_l[-1]:.4f}, val {va_l[-1]:.4f}; lowest val {va_l[best]:.4f} at {unit} {best + 1}")
        rows += [{"model": name, "unit": unit, "step": i + 1, "train_log_loss": t, "val_log_loss": v}
                 for i, (t, v) in enumerate(zip(tr_l, va_l))]

    df = pd.DataFrame(rows)
    df.to_csv(repo_path("experiments/results/loss_curves.csv"), index=False)
    repo_path("experiments/results/loss_curves_summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, len(runs), figsize=(4.0 * len(runs), 3.3))
    for ax, (name, g) in zip(axes, df.groupby("model", sort=False)):
        g = g.reset_index(drop=True)
        ax.plot(g.step, g.train_log_loss, color="#2a6fdb", lw=1.6, label="train (folds 1-4)")
        ax.plot(g.step, g.val_log_loss, color="#e8590c", lw=1.6, label=f"validation (fold {VAL_FOLD})")
        ax.axvline(g.step[g.val_log_loss.idxmin()], color="#888", lw=0.8, ls="--")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("boosting iteration (trees)" if g.unit[0] == "iteration" else "epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("binary log loss")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = repo_path("experiments/figures/loss_curves.png")
    fig.savefig(out, dpi=160)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
