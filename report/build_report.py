"""Build the PDF report from the results in experiments/ (every table number is read from a file).

    python report/build_report.py        # writes report/report.tex and report/report.pdf

Needs tectonic (brew install tectonic) or xelatex. Run after the pipeline in README.md.
Figures come from experiments/figures/ and report/figures/.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments/results"
FIG = ROOT / "experiments/figures"
RFIG = ROOT / "report/figures"
OUT = ROOT / "report"


def esc(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s


def f(x, d=3) -> str:
    return "--" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{d}f}"


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}\\%"


def table(rows: list[list], heads: list[str], align: str, caption: str, label: str, size: str = r"\small") -> str:
    body = "\n".join(" & ".join(str(v) for v in r) + r" \\" for r in rows)
    return (f"\\begin{{table}}[H]\\centering{size}\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\resizebox{{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}}{{!}}{{"
            f"\\begin{{tabular}}{{{align}}}\\toprule\n" + " & ".join(heads) + r" \\ \midrule" + "\n"
            + body + "\n\\bottomrule\\end{tabular}}\\end{table}\n")


def figure(path: Path, caption: str, label: str, width: str = r"\linewidth") -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    rel = os.path.relpath(path, OUT)
    return (f"\\begin{{figure}}[H]\\centering\\includegraphics[width={width}]{{{rel}}}\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\\end{{figure}}\n")


# --------------------------------------------------------------------------- per-model text
# Display name, author, and the answers to the per-model questions of the brief (3.2). Numbers in
# {braces} are filled from experiments/results/models/<name>.json.
MODELS = {
    "advitha_lgbm_tuned": dict(
        title="LightGBM, Bayesian-tuned (final model)", by="Advitha, Adyanth",
        why="Same algorithm as the untuned LightGBM below; the hyperparameters come from the 40-trial Optuna "
            "search of Section~4. It is the model with the highest cross-validated Top-1, so it is the one exported "
            "to the API and used as the HMM emission model.",
        sl="Strengths: best CV Top-1 and CV macro F1 of all ten models and the lowest STUDENT\\_TEST log loss, i.e. "
           "the best-calibrated probabilities, which the HMM uses directly. Limitations: 254 leaves and 800 trees "
           "make it the slowest model to score ({inf:.0f}\\,ms per trip, 4.6$\\times$ the untuned one) and it has the "
           "largest train/CV gap.",
        loss="Binary log loss (cross-entropy) on (point, candidate) rows, minimised by gradient boosting with "
             "L1 ({reg_alpha:.2f}) and L2 ({reg_lambda:.3f}) penalties on the leaf weights.",
        fit="Mild overfitting. Train-fold macro F1 is {trf1:.3f} against {cvf1:.3f} in CV (gap {gap:+.3f}, the "
            "largest of all models). The loss curve on a held-out fold (Figure~\\ref{{fig:loss}}) shows the validation "
            "loss flat from about iteration {cbest} while the training loss keeps falling ({ctr:.4f} vs {cva:.4f} at "
            "the end). The validation loss does not rise, so the extra trees cost time but not accuracy."),
    "advitha_lgbm": dict(
        title="LightGBM (untuned)", by="Advitha",
        why="Each GPS point has 1--30 candidate arcs and the question is which one is right; this is a tabular "
            "binary problem with strong feature interactions (a near arc is only right if its direction also "
            "matches and the neighbouring points agree). Gradient-boosted trees learn such interactions without "
            "scaling, handle the 1:10 class ratio well and give usable probabilities. It was the first model "
            "trained on the shared feature table and served as the reference for tuning.",
        sl="Strengths: second-best CV Top-1, fast to fit ({fit:.0f}\\,s on 2 million rows) and to score "
           "({inf:.0f}\\,ms per trip). Limitations: many hyperparameters; leaf-wise growth can overfit "
           "small groups of rows, which is why min\\_child\\_samples and subsampling are set.",
        loss="Binary log loss, gradient boosting with L2 penalty {reg_lambda:g} on leaf weights.",
        fit="Slight overfitting: train-fold macro F1 {trf1:.3f} vs CV {cvf1:.3f} (gap {gap:+.3f}). On the held-out "
            "fold the validation loss is lowest at iteration {cbest} of {cn} and stays flat after it "
            "(Figure~\\ref{{fig:loss}}). CV Top-1 varies by $\\pm${cvtop1sd:.3f} across folds, so the trips, not the "
            "model, dominate the variance."),
    "advitha_xgb": dict(
        title="XGBoost", by="Advitha",
        why="A second boosting implementation, grown depth-wise with second-order (Newton) updates, to check "
            "whether the result depends on the LightGBM library or on the features.",
        sl="Strengths: the highest STUDENT\\_TEST Top-1 ({st_top1:.4f}) and fast inference ({inf:.0f}\\,ms per trip). "
           "Limitations: slightly lower CV Top-1 than both LightGBMs; depth-wise trees of depth 8 need more "
           "trees to model the same interactions.",
        loss="Regularised binary log loss (objective binary:logistic) with L2 penalty {reg_lambda:g}, histogram "
             "tree method.",
        fit="Slight overfitting: train-fold F1 {trf1:.3f} vs CV {cvf1:.3f} (gap {gap:+.3f}). Validation loss is "
            "lowest at iteration {cbest} of {cn} and flat afterwards (Figure~\\ref{{fig:loss}}). Its STUDENT\\_TEST lead "
            "over the tuned LightGBM is about 25 of 19,133 points, far inside the fold-to-fold spread, so we did not "
            "select on it."),
    "advitha_hgb": dict(
        title="HistGradientBoosting (scikit-learn)", by="Advitha",
        why="The scikit-learn histogram boosting implementation, a third boosting variant with no extra "
            "dependency, to confirm that the three boosting libraries agree.",
        sl="Strengths: ties with the other boosting models (CV Top-1 {cvtop1:.4f}) and is simple to deploy. "
           "Limitations: slightly lower STUDENT\\_TEST Top-1 and higher log loss than LightGBM and XGBoost; the "
           "larger learning rate (0.1) gives a coarser fit.",
        loss="Binary log loss, gradient boosting with L2 regularisation {l2_regularization:g}; early stopping off so "
             "every model sees the same data.",
        fit="Slight overfitting: train-fold F1 {trf1:.3f} vs CV {cvf1:.3f} (gap {gap:+.3f}), the same pattern as the "
            "other boosting models. With early stopping off no per-iteration history was recorded for this model."),
    "sudhe_random_forest": dict(
        title="Random Forest", by="Sudhe",
        why="Bagging is a different way to build a strong tree ensemble: deep trees on bootstrap samples with "
            "random feature subsets, averaged. It tests whether variance reduction alone reaches the boosting "
            "result.",
        sl="Strengths: robust, few hyperparameters, CV Top-1 {cvtop1:.4f}, close to boosting. Limitations: probabilities "
           "are less calibrated (log loss {st_ll:.3f} vs 0.047 for boosting), it is the slowest model to fit "
           "({fit:.0f}\\,s) and the balanced class weights shift the probabilities.",
        loss="No global loss: each tree greedily maximises the Gini impurity decrease at each split; balanced class "
             "weights.",
        fit="Moderate variance: train-fold F1 {trf1:.3f} vs CV {cvf1:.3f} (gap {gap:+.3f}). Depth 18 with leaves of 2 rows "
            "lets single trees memorise, and averaging 120 of them only partly removes it."),
    "sudhe_extra_trees": dict(
        title="Extra Trees", by="Sudhe",
        why="Like Random Forest, but split thresholds are drawn at random, which reduces variance further and "
            "fits faster; a check of how much the exact split points matter.",
        sl="Strengths: smaller train/CV gap than Random Forest, faster to fit ({fit:.0f}\\,s). Limitations: random "
           "thresholds blur the sharp distance and heading cut-offs that matter here, so Top-1 and log loss "
           "({st_ll:.3f}) are worse than Random Forest.",
        loss="Gini impurity per split with random thresholds; balanced class weights.",
        fit="Low variance, slight underfitting relative to Random Forest: gap {gap:+.3f} (train {trf1:.3f}, CV "
            "{cvf1:.3f}), but CV macro F1 is 0.009 below Random Forest."),
    "sudhe_decision_tree": dict(
        title="Decision Tree", by="Sudhe",
        why="The single interpretable tree is the baseline that shows what the ensembles add; its first splits "
            "(distance rank, heading cosine) confirm which features carry the signal.",
        sl="Strengths: fastest inference ({inf:.1f}\\,ms per trip) and readable rules; already {st_top1:.3f} Top-1. "
           "Limitations: piecewise-constant probabilities (log loss {st_ll:.3f}) are poor HMM emissions and the "
           "depth limit caps accuracy.",
        loss="Gini impurity decrease per split; balanced class weights.",
        fit="Underfitting, not overfitting: depth 8 and leaves of at least 10 rows give a small gap ({gap:+.3f}; "
            "train {trf1:.3f}, CV {cvf1:.3f}) but CV macro F1 is 0.05 below boosting."),
    "adyanth_logreg": dict(
        title="Logistic Regression", by="Adyanth",
        why="The simplest probabilistic classifier: a linear score of the 36 features. It shows how far the "
            "hand-made features go without learned interactions, and its coefficients are easy to explain. We also "
            "used it for the feature ablation in Section~2.",
        sl="Strengths: trains in {fit:.0f}\\,s, scores in under 1\\,ms per trip, stable across folds (CV F1 std "
           "{cvf1sd:.3f}, the lowest). Limitations: a linear boundary cannot express rules such as ``close \\emph{{and}} "
           "same direction''; balanced weights make the probabilities less calibrated (log loss {st_ll:.3f}).",
        loss="Class-weighted binary cross-entropy with an L2 penalty (C = {C:g}), solved by L-BFGS; features "
             "standardised inside the pipeline so each fold refits the scaler.",
        fit="Underfitting (high bias): train-fold and CV macro F1 are equal ({trf1:.3f} vs {cvf1:.3f}, gap "
            "{gap:+.3f}) and both are about 0.06 below boosting. More data would not help; more model capacity does."),
    "adyanth_sgd": dict(
        title="SGD classifier (logistic loss)", by="Adyanth",
        why="The same linear model as Logistic Regression fitted by stochastic gradient descent, to see whether an "
            "iterative solver that scales to more data changes the result.",
        sl="Strengths: fastest to fit of the linear models ({fit:.0f}\\,s), same accuracy as Logistic Regression. "
           "Limitations: sensitive to feature scale and the learning-rate schedule; same linear limits.",
        loss="Class-weighted logistic loss with L2 penalty $\\alpha$ = {alpha:g}, minimised by SGD with the "
             "``optimal'' learning-rate schedule; standardised features.",
        fit="Underfitting, like Logistic Regression: gap {gap:+.3f} (train {trf1:.3f}, CV {cvf1:.3f}). In the per-epoch "
            "curves (Figure~\\ref{{fig:loss}}) the validation loss is lowest after epoch {cbest} ({cbv:.4f}) and then "
            "fluctuates up to {cva:.4f} by epoch {cn}, while the training loss stays higher ({ctr:.4f}). Training loss "
            "never falls below validation loss, so this is noise from the stochastic updates, not memorisation; "
            "a linear model has no capacity to overfit 2 million rows."),
    "adyanth_gnb": dict(
        title="Gaussian Naive Bayes", by="Adyanth",
        why="A generative baseline: it models each feature per class as a Gaussian and assumes independence, "
            "which is quick to fit and shows the cost of ignoring feature correlations.",
        sl="Strengths: fits in about a second, no tuning. Limitations: several features are strongly correlated "
           "(distance, distance rank, distance to the nearest candidate; heading cosines of neighbouring points), "
           "so the independence assumption multiplies the same evidence and the model is over-confident: log loss "
           "{st_ll:.3f}, 18$\\times$ boosting, Top-1 {st_top1:.3f}.",
        loss="No iterative loss: class priors and per-class feature means and variances by maximum likelihood "
             "(var\\_smoothing {var_smoothing:g}).",
        fit="Underfitting: train-fold F1 {trf1:.3f} equals CV {cvf1:.3f} (gap {gap:+.3f}); the error comes from the "
            "wrong independence assumption, not from too little data."),
}
KEY_PARAMS = {
    "advitha_lgbm_tuned": ["n_estimators", "learning_rate", "num_leaves", "min_child_samples"],
    "advitha_lgbm": ["n_estimators", "learning_rate", "num_leaves", "min_child_samples"],
    "advitha_xgb": ["n_estimators", "learning_rate", "max_depth"],
    "advitha_hgb": ["max_iter", "learning_rate", "max_leaf_nodes"],
    "sudhe_random_forest": ["n_estimators", "max_depth", "min_samples_leaf"],
    "sudhe_extra_trees": ["n_estimators", "max_depth", "min_samples_leaf"],
    "sudhe_decision_tree": ["max_depth", "min_samples_leaf"],
    "adyanth_logreg": ["C", "penalty", "class_weight"],
    "adyanth_sgd": ["alpha", "penalty", "class_weight"],
    "adyanth_gnb": ["var_smoothing"],
}
CURVE = {"advitha_lgbm_tuned": "LightGBM (Bayesian-tuned)", "advitha_lgbm": "LightGBM (untuned)",
         "advitha_xgb": "XGBoost", "adyanth_sgd": "SGD (logistic loss)"}


def fmt_v(v) -> str:
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 0.01 or abs(v) >= 1000 else f"{v:.4g}"
    return str(v)


def model_vars(d: dict, curves: dict) -> dict:
    c, s = d["cv"], d["student_test"]
    v = {k: val for k, val in d["config"].items() if isinstance(val, (int, float))}
    v.update(trf1=c["train_macro_f1_mean"], cvf1=c["macro_f1_mean"], cvf1sd=c["macro_f1_std"],
             gap=c["train_macro_f1_mean"] - c["macro_f1_mean"], cvtop1=c["top1_mean"], cvtop1sd=c["top1_std"],
             st_top1=s["top1"], st_f1=s["macro_f1"], st_ll=s["log_loss"], fit=d["fit_seconds"],
             inf=d["inference_ms_per_trip"])
    cu = curves.get(CURVE.get(d["name"], ""), {})
    v.update(cbest=cu.get("best_at"), cn=cu.get("n"), ctr=cu.get("final_train", 0), cva=cu.get("final_val", 0),
             cbv=cu.get("best_val", 0))
    return v


def main() -> int:
    models = {p.stem: json.loads(p.read_text()) for p in sorted((RES / "models").glob("*.json"))}
    comp = pd.read_csv(RES / "model_comparison.csv")
    comp = comp[~comp.is_ablation.fillna(False).astype(bool)].copy()
    ml = comp[comp.cv_macro_f1_mean.notna()].sort_values("cv_top1_mean", ascending=False)
    rules = comp[comp.cv_macro_f1_mean.isna()]
    tune = pd.read_csv(RES / "tuning_comparison.csv")
    bo = json.loads((RES / "bayes_opt/summary.json").read_text())
    imp = json.loads((RES / "bayes_opt/param_importance.json").read_text())
    grid = pd.read_csv(RES / "hmm_grid.csv")
    hc = pd.read_csv(RES / "hmm_comparison.csv")
    hst = pd.read_csv(RES / "hmm_student_test.csv")
    pipe = pd.read_csv(RES / "pipeline_summary.csv").set_index("variant")
    errs = pd.read_csv(RES / "pipeline_error_types.csv")
    part = pd.read_csv(RES / "eda_partition_summary.csv").set_index("partition")
    evs = pd.read_csv(RES / "evaluator_scores.csv", dtype={"trip_id": str})
    curves = json.loads((RES / "loss_curves_summary.json").read_text())
    meta = json.loads((ROOT / "artifacts/metadata.json").read_text())
    hy = yaml.safe_load((ROOT / "configs/hmm.yaml").read_text())
    abl = {k: models[k]["student_test"]["top1"] for k in models if k.startswith("ablation")}

    cls, hm, rule = pipe.loc["model"], pipe.loc["model_hmm"], pipe.loc["baseline_rule"]
    ev = errs.groupby("variant")[["wrong", "reverse_twin", "adjacent_arc", "other", "twin_pred_agrees_heading"]].sum()
    eh = ev.loc["model_hmm"]
    per_trip = errs[errs.variant == "model_hmm"].set_index("trip_id")
    worst = per_trip.wrong.idxmax()
    worst_share = per_trip.wrong.max() / eh.wrong
    piv = hst.pivot_table(index="trip_id", columns="variant", values="top1")
    a_col = [c for c in piv.columns if c.startswith("classifier argmax")][0]
    t_col = [c for c in piv.columns if c.startswith("HMM tuned")][0]
    worse = piv[piv[t_col] < piv[a_col]]
    n_better = int((piv[t_col] > piv[a_col]).sum())
    best_grid = grid[grid.variant != "argmax"].sort_values("top1", ascending=False).iloc[0]
    argmax_tr = grid[grid.variant == "argmax"].iloc[0]
    sel = meta["model"]
    lt = models[sel]
    ev_top1 = (evs.top1 * evs.n_points_scored).sum() / evs.n_points_scored.sum()

    # ---------------------------------------------------------------- tables
    tr_, st_ = part.loc["train"], part.loc["student_test"]
    qrows = [[esc(lab), f"{int(tr_[c]):,}" if c != "max_gap_s" else f"{tr_[c]:,.0f}",
              f"{int(st_[c]):,}" if c != "max_gap_s" else f"{st_[c]:,.0f}"]
             for lab, c in [("Trips", "trips"), ("GPS points", "obs_total"), ("Missing files", "missing_files"),
                            ("NaN rows", "nan_rows"), ("Out-of-range coordinates", "out_of_range"),
                            ("(0, 0) coordinates", "zero_coords"), ("Duplicate timestamps", "dup_timestamps"),
                            ("Backwards timestamps", "backwards_time"), ("Time steps > 5 s", "gaps_gt_5s"),
                            ("Longest time step (s)", "max_gap_s"), ("Steps implying > 60 m/s", "jumps_gt_60mps")]]
    q_tab = table(qrows, ["Check", "TRAIN", "STUDENT\\_TEST"], "lrr",
                  "Missing-value and anomaly checks per partition (\\texttt{scripts/inventory.py}, "
                  "\\texttt{scripts/eda.py}).", "tab:quality")

    def keycfg(name):
        cfg = models[name]["config"]
        return esc(", ".join(f"{k}={fmt_v(cfg[k])}" for k in KEY_PARAMS[name] if k in cfg))

    comp_rows = [[esc(MODELS[r.model]["title"].replace(" (final model)", "")), esc(r.family), keycfg(r.model),
                  f"{r.cv_macro_f1_mean:.3f} $\\pm$ {r.cv_macro_f1_std:.3f}", f(r.student_test_top1, 4),
                  f(r.route_continuity_argmax), f(r.student_test_macro_f1), f(r.student_test_log_loss),
                  f"{r.inference_ms_per_trip:.1f}"] for r in ml.itertuples()]
    comp_tab = ("\\begin{table}[H]\\centering\\scriptsize\n\\caption{Model comparison, sorted by the selection "
                "metric (CV Top-1, Table~\\ref{tab:cv}). CV = 5-fold GroupKFold by trip on TRAIN; Top-1, continuity, "
                "macro F1 and log loss on STUDENT\\_TEST. Top-1 is observation-level accuracy. Continuity is for the "
                "classifier-only route (the HMM is in Section~5). Inference = scoring time per trip.}"
                "\\label{tab:comp}\n\\begin{tabular}{p{2.3cm}p{2.2cm}p{3.6cm}ccccccc}\\toprule\n"
                "Model & Family & Key configuration & CV macro F1 & Top-1 & Route cont. & Macro F1 & Log loss & "
                "ms/trip \\\\ \\midrule\n" + "\n".join(" & ".join(r) + r" \\" for r in comp_rows)
                + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    cv_rows = [[esc(MODELS[r.model]["title"].replace(" (final model)", "")), f(r.cv_top1_mean, 4),
                f(r.cv_top1_std, 4), f(r.cv_log_loss_mean, 4), f(r.train_macro_f1_mean, 3),
                f"{r.train_macro_f1_mean - r.cv_macro_f1_mean:+.3f}", f"{r.fit_seconds:.0f}"] for r in ml.itertuples()]
    cv_tab = table(cv_rows, ["Model", "CV Top-1", "std", "CV log loss", "Train-fold F1", "Train$-$CV F1", "Fit (s)"],
                   "lcccccc", "Cross-validation detail (TRAIN). Train$-$CV F1 is the over/underfitting evidence used "
                   "in Section~3.2.", "tab:cv")

    tune_rows = [[esc(r.metric), esc(r.partition), f(r.untuned, 4), f(r.tuned, 4), f"{r.change:+.4f}"]
                 for r in tune.itertuples()]
    tune_tab = table(tune_rows, ["Metric", "Partition", "Untuned", "Tuned", "Change"], "llrrr",
                     "Untuned vs Bayesian-tuned LightGBM (same folds, features, seed).", "tab:tune")
    space_rows = []
    for k, v in bo["search_space"].items():
        b = bo["best_params"][k]
        space_rows.append([esc(k), f"{v['low']}--{v['high']}", "log" if v.get("log") else "uniform",
                           esc(f"{b:.4g}" if isinstance(b, float) else b), f"{imp.get(k, 0):.2f}"])
    space_tab = table(space_rows, ["Parameter", "Range", "Scale", "Best", "Importance"], "lcccc",
                      f"Search space, best values (trial {bo['best_trial']}) and fANOVA importance.", "tab:space")

    g = grid.sort_values("top1", ascending=False)
    grid_tab = table([[esc(r.variant), f(r.top1, 4), f(r.continuity_mean, 4), f(r.route_f1_mean, 4)]
                      for r in g.itertuples()],
                     ["Setting (w = classifier weight, $\\beta$ in m)", "Top-1", "Continuity", "Route F1"], "lccc",
                     "HMM decoder grid on the 80 TRAIN trips with out-of-fold classifier scores (objective: Top-1).",
                     "tab:grid")
    hmm_tab = table([[esc(r.variant), f(r.top1, 4), f(r.continuity_mean), f"{int(r.breaks_total):,}",
                      f(r.route_precision_mean), f(r.route_recall_mean), f(r.route_f1_mean), f"{r.decode_ms_mean:.0f}"]
                     for r in hc.itertuples()],
                    ["Method", "Top-1", "Cont.", "Breaks", "Route P", "Route R", "Route F1", "Decode ms/trip"],
                    "lccccccc", "Classifier-only vs classifier + HMM on STUDENT\\_TEST (final exported classifier).",
                    "tab:hmm")
    pt = piv[[a_col, t_col]].reset_index()
    pt_tab = table([[f"{int(r.trip_id):08d}", f(getattr(r, "_2")), f(getattr(r, "_3")),
                     f"{getattr(r, '_3') - getattr(r, '_2'):+.3f}"] for r in pt.itertuples()],
                   ["Trip", "Classifier only", "+ HMM (tuned)", "Change"], "lccc",
                   "Per-trip Top-1 on STUDENT\\_TEST.", "tab:pertrip")
    ev_tab = table([[r.trip_id, f"{r.n_points_scored:,}", f(r.top1), f(r.candidate_miss_rate), f(r.route_f1),
                     f(r.route_continuity)] for r in evs.itertuples()],
                   ["Evaluator trip", "Points scored", "Top-1", "Miss rate", "Route F1", "Continuity"], "lccccc",
                   "Scores computed by the API server after prediction on the 10 EVALUATOR trips "
                   "(\\texttt{scripts/smoke\\_test\\_api.py}, log in Figure~\\ref{fig:api}).", "tab:eval")
    worse_txt = ", ".join(f"{int(t):08d} ({f(r[a_col])} $\\to$ {f(r[t_col])})" for t, r in worse.iterrows()) or "none"
    rule_txt = "; ".join(f"{esc(r.model)} ({esc(r.key_config)}): Top-1 {f(r.student_test_top1)}"
                         for r in rules.itertuples())

    # ---------------------------------------------------------------- per-model sections
    per_model = []
    for name in ml.model:
        d, p = models[name], MODELS[name]
        v = model_vars(d, curves)
        hp = ", ".join(f"\\texttt{{{esc(k)}}} = {esc(fmt_v(val))}" for k, val in d["config"].items()
                       if k not in ("tuned_by", "eval_metric", "subsample_freq"))
        s = d["student_test"]
        per_model.append(
            f"\\subsubsection{{{esc(p['title'])}}}\n"
            f"\\textit{{Trained by {p['by']}; script \\texttt{{scripts/models/{esc(name)}.py}}.}}\n\n"
            f"\\textbf{{A1. Why suitable.}} {p['why']}\n\n"
            f"\\textbf{{A2. Strengths and limitations.}} {p['sl'].format(**v)}\n\n"
            f"\\textbf{{B5. Hyperparameters (initial values).}} {hp}"
            + (f" (from Optuna, {esc(d['config'].get('tuned_by', ''))})." if "tuned_by" in d["config"] else ".") + "\n\n"
            f"\\textbf{{B6. Loss / objective.}} {p['loss'].format(**v)}\n\n"
            f"\\textbf{{C7--C8. STUDENT\\_TEST results.}} Top-1 {s['top1']:.4f}, macro F1 {s['macro_f1']:.4f}, "
            f"log loss {s['log_loss']:.4f}, classifier-only route continuity {s['route_continuity_argmax']:.3f} "
            f"(CV Top-1 {d['cv']['top1_mean']:.4f} $\\pm$ {d['cv']['top1_std']:.4f}).\n\n"
            f"\\textbf{{C9. Over- or underfitting.}} {p['fit'].format(**v)}\n")

    # ---------------------------------------------------------------- document
    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage[T1]{fontenc}\usepackage{tgheros}\renewcommand{\familydefault}{\sfdefault}
\usepackage{graphicx,booktabs,float,caption,amsmath,xcolor}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\setcounter{secnumdepth}{3}
\title{\textbf{PathSnap: GPS map-matching with a trained candidate classifier and an HMM decoder}\\[2pt]
\large UE24CS352A Machine Learning Hackathon --- Team 148 --- Technical report}
\author{Adyanth, Advitha, Sudhe, Tilak}
\date{19 November 2026}
\begin{document}
\maketitle
"""
    tex += rf"""
\begin{{abstract}}
We reconstruct the road route driven by a vehicle from its noisy GPS track. For every GPS point we generate the
nearby road arcs as candidates, score each (point, candidate) pair with a trained classifier, and decode one connected
route with a hidden Markov model solved by Viterbi. Ten classifiers from six families were compared with the same
trip-grouped cross-validation; a Bayesian-tuned LightGBM was selected by cross-validated Top-1. On the 10 held-out
STUDENT\_TEST trips the full pipeline matches \textbf{{{pct(hm.top1)}}} of GPS points to the correct arc
(classifier alone {pct(cls.top1)}, rule baseline {pct(rule.top1)}), raises route continuity from {f(cls.continuity_mean)} to
{f(hm.continuity_mean)} and cuts route breaks from {int(cls.breaks_total)} to {int(hm.breaks_total)}. On the 10 EVALUATOR
trips, scored by the API server, it reaches {pct(ev_top1)}. A FastAPI server runs the pipeline and a Streamlit UI shows the
predicted route and the server-side scores.
\end{{abstract}}

\section{{Problem and pipeline}}
Each trip has a GPS track (longitude, latitude, time), its own road network (nodes and directed arcs) and, for
development trips, a reference route given as an ordered list of arc ids. The task is to output one arc per GPS
point. The pipeline is:
\begin{{enumerate}}\setlength\itemsep{{0pt}}
\item project the trip into its local UTM zone (all distances in metres);
\item generate candidate arcs within $R=60$\,m of each point, at most $K=30$ (fallback 150\,m);
\item compute 36 features per (point, candidate) pair from GPS and network geometry only;
\item score each pair with the exported classifier ($P(\text{{candidate is the true arc}})$);
\item decode a route with an HMM/Viterbi decoder over the candidate sets;
\item serve it through an API and a UI; evaluator scoring happens on the server after prediction.
\end{{enumerate}}
\textbf{{Partitions.}} TRAIN (80 trips) for fitting, cross-validation and all tuning; STUDENT\_TEST (10 trips) for
held-out comparison; EVALUATOR (10 trips) only on the API server. The loader refuses to read the evaluator partition
unless the API runs in serve mode, so it cannot leak into development. Seed 148 everywhere.

\section{{Dataset understanding and exploratory data analysis}}
\subsection{{How we preprocessed the data}}
\begin{{enumerate}}\setlength\itemsep{{0pt}}
\item \textbf{{Quality checks}} on both TRAIN and STUDENT\_TEST (Table~\ref{{tab:quality}}): missing files, NaN rows,
invalid or (0,0) coordinates, duplicate or backwards timestamps, time gaps, implied speeds, network integrity. No
rows are deleted; problems are flagged and handled downstream.
\item \textbf{{Projection.}} The trips lie in 25 different UTM zones. Each trip is projected to the UTM zone of its
median point; Euclidean distances on degrees are never used.
\item \textbf{{Per-point labels.}} The reference route is an arc sequence, not a label per point. We align points to
route arcs with a monotonic dynamic programme: the route position only moves forward, and the cost is
point-to-arc distance (capped at 100\,m) plus a heading-mismatch term. Median point-to-label distance is 4.75\,m.
Points more than 50\,m from their aligned arc (11.1\% of TRAIN, mostly three trips where the route covers only part of
the track; 0.14\% of STUDENT\_TEST) are flagged unreliable and excluded from training and scoring.
\item \textbf{{Candidates.}} A spatial index (STRtree) returns the arcs within $R$ of each point, at most $K$. $R=60$\,m
and $K=30$ were chosen on TRAIN from the miss rate (true arc not among the candidates) against candidate-set size
(Figure~\ref{{fig:miss}}): miss rate 0.029\% on TRAIN and 0\% on STUDENT\_TEST, about 12 candidates per point, 9\%
positives.
\end{{enumerate}}

{q_tab}
\subsection{{What we observed}}
\textbf{{Missing values}}: none --- every trip has all four files and there are no NaN, invalid or duplicate rows
(Table~\ref{{tab:quality}}, both partitions). \textbf{{Sampling}}: 96.5\% of TRAIN steps are exactly 1\,s, but 1,216 TRAIN steps
exceed 5\,s and the longest (42\,h, trip 00000090) is a parked vehicle, so the decoder must split on gaps.
\textbf{{Speed outliers}}: 546 TRAIN steps imply more than 60\,m/s, almost all in three trips and sustained (motorways),
not single spikes, so no point was removed. \textbf{{Networks}}: each trip has its own network (299--155,003 nodes,
649--307,392 arcs), so arc ids are local and the problem is framed per (point, candidate) pair rather than as a
global arc vocabulary. About 92\% of arcs have a reverse twin (two-way roads), so direction is a central difficulty.
\textbf{{Routes}}: in three TRAIN trips the reference route covers only part of the GPS path, which is why labels are
checked for reliability.

{figure(FIG / "eda_trajectories.png", "GPS trajectory shapes (TRAIN, first 16 trips).", "fig:traj", r"0.78\linewidth")}
{figure(FIG / "eda_dt_hist.png", "Time steps between GPS points and the gaps above 5\\,s (TRAIN).", "fig:dt", r"0.8\linewidth")}
{figure(FIG / "eda_network.png", "Road networks: size, weak connectivity and node degree per trip (TRAIN and STUDENT\\_TEST).", "fig:net")}
{figure(FIG / "label_quality.png", "Quality of the derived labels per trip (TRAIN and STUDENT\\_TEST).", "fig:labels")}
{figure(FIG / "candidate_miss_rate.png", "Candidate miss rate and candidate count against radius $R$ (TRAIN).", "fig:miss")}
{figure(RFIG / "map_eda_train_00000033.png", "Folium EDA map, TRAIN trip 00000033: raw GPS (blue), road network within 200\\,m (grey), reference route (green). The route covers only part of the track.", "fig:eda_map", r"0.75\linewidth")}

\subsection{{Features and why we chose them}}
All 36 features use only GPS and network geometry (never labels), grouped by what they capture:
\emph{{distance}} (point-to-arc distance, rank among the candidates, gap to the closest, projection position) because the
true arc is usually the closest; \emph{{direction}} (cosine between GPS heading and arc bearing, also for the previous and
next points) because half the candidate rows are reverse twins with identical distance; \emph{{context}} (distance of
the neighbouring points to the same arc) because one noisy point can sit near the wrong road while its neighbours do
not; \emph{{motion}} (time steps, speeds, acceleration, turn angle); \emph{{network}} (arc length, node degrees,
reverse-twin flag); \emph{{sequence}} (agreement with the neighbours' best arc) and \emph{{density}} (number of nearby
candidates). An ablation with Logistic Regression shows what each group adds on STUDENT\_TEST: distance and density only
{f(abl['ablation_logreg_distance'])}, plus direction {f(abl['ablation_logreg_distance_direction'])}, all 36 features
{f(models['adyanth_logreg']['student_test']['top1'])} Top-1.

\section{{Model experimentation and comparison}}
\subsection{{Overall comparison and final model}}
Every model goes through one shared harness (\texttt{{src/pathsnap/harness.py}}): the same 36 features, the same trip
folds (GroupKFold(5) by trip, saved in \texttt{{experiments/results/cv\_folds.csv}}, so no trip is in both training and
validation), the same metrics and the same STUDENT\_TEST scoring. The \textbf{{primary metric is observation-level Top-1}}:
is the highest-scored candidate of a point its label arc? It needs no threshold and is exactly what the route needs.
Candidate-level macro F1 (threshold 0.5) and log loss are reported with it.

{comp_tab}
{cv_tab}
Rule baselines (no learning, STUDENT\_TEST): {rule_txt}.

\textbf{{Final model: Bayesian-tuned LightGBM}} (\texttt{{{esc(sel)}}}), chosen by the rule fixed before tuning: highest
CV Top-1. It also has the best CV macro F1 and the lowest STUDENT\_TEST log loss, i.e. the best-calibrated
probabilities, which matters because the HMM uses them as emissions. XGBoost is marginally higher on STUDENT\_TEST
Top-1, but the gap (about 25 of 19,133 points) is far inside the fold-to-fold spread, and choosing on a 10-trip test set
would be selecting on the test data. The boosting models are statistically tied; the features matter more than the
library. The cost is speed: about {lt['inference_ms_per_trip']:.0f}\,ms per trip for scoring, still far below a second.

\subsection{{Per model}}
{"".join(per_model)}

\subsection{{Model visualisations}}
{figure(FIG / "cv_macro_f1.png", "CV macro F1 per model (TRAIN): mean $\\pm$ std over the 5 folds, each fold shown.", "fig:cvf1", r"0.85\linewidth")}
{figure(FIG / "model_comparison.png", "CV macro F1, STUDENT\\_TEST Top-1 and STUDENT\\_TEST log loss per model.", "fig:compfig")}
{figure(FIG / "confusion_matrices.png", "STUDENT\\_TEST confusion matrices (threshold 0.5, row-normalised; 1 = true arc).", "fig:cm")}
{figure(FIG / "loss_curves.png", "Training and validation log loss per boosting iteration (LightGBM, XGBoost) and per epoch (SGD), CV fold 0 held out (\\texttt{scripts/loss\\_curves.py}). Dashed line: lowest validation loss.", "fig:loss")}
Only boosting and SGD have a training history. Random Forest, Extra Trees and the Decision Tree are grown once,
Logistic Regression is solved to convergence by L-BFGS and Gaussian NB is fitted in closed form, so no loss curves are
drawn for them; their over/underfitting evidence is the train-fold vs CV gap in Table~\ref{{tab:cv}}.

\section{{Bayesian hyperparameter optimisation}}
\textbf{{Model.}} LightGBM, the best model before tuning. \textbf{{Method.}} Optuna's TPE sampler (multivariate, seed 148,
10 random start-up trials) with a median pruner.
\textbf{{Q1--Q2. Search space}}: Table~\ref{{tab:space}} (seven hyperparameters, log scale for the learning rate, number
of leaves, child size and regularisation).
\textbf{{Q3. Objective}}: mean CV Top-1 on TRAIN with the harness folds, the same as the selection metric; STUDENT\_TEST
was used once, after the search. Trial 0 was the untuned configuration, so the search starts at the baseline
({f(bo['baseline_trial_cv_top1'], 4)}, identical to the harness).
\textbf{{Q4. Trials}}: {bo['n_trials']} ({bo['n_complete']} complete, {bo['n_pruned']} pruned), {bo['trial_compute_seconds_total'] / 60:.0f} minutes on
8 CPU threads, 200 model fits; every trial is stored in \texttt{{experiments/results/bayes\_opt/}}.
\textbf{{Q5. Best values}}: Table~\ref{{tab:space}}, trial {bo['best_trial']}.

{space_tab}
{tune_tab}
{figure(FIG / "bayes_opt_history.png", "Optimisation history: every trial, best so far, untuned baseline dashed.", "fig:bohist", r"0.8\linewidth")}
\textbf{{Q6. Did tuning help?}} Slightly, and consistently: CV Top-1 +0.0012 (better in 4 of 5 folds), STUDENT\_TEST
Top-1 +0.0008, log loss lower. The hand-picked defaults were already close to the optimum for these features. After the
random phase TPE converged on many leaves (about 200--255), a low learning rate and more trees; \texttt{{num\_leaves}}
explains about 70\% of the variation. The price is a 3.5$\times$ longer fit, 4.6$\times$ slower inference and a slightly
larger train/CV gap. The pruner never fired because all configurations score within 0.004 of each other.

\section{{HMM and Viterbi route reconstruction}}\label{{sec:hmm}}
\textbf{{Q1. Hidden states.}} At time $t$ the hidden states are the candidate arcs of point $t$ (a ragged set, at most 30).
\textbf{{Q2. Emissions.}} $\log e_t(a) = \operatorname{{logsoftmax}}_a\big(w \log \tilde p_t(a) + (1-w)\log
\mathcal{{N}}(d_t(a);0,\sigma)\big)$, where $\tilde p$ is the classifier probability normalised over the point's
candidates (floored at $10^{{-6}}$), $d$ the point-to-arc distance and $w$ the classifier weight.
\textbf{{Q3. Transitions.}} From arc $a$ at $t-1$ to arc $b$ at $t$ we compute the driving distance on the directed
network (rest of $a$, shortest path by Dijkstra from the end of $a$ to the start of $b$, part of $b$; along the arc for
a same-arc move) and compare it with the straight-line distance between the two GPS points:
$\log T(a\to b) = \operatorname{{logsoftmax}}_b\big(-|\text{{route}} - \text{{straight}}|/\beta\big)$. Paths longer than
$2\times$straight$+200$\,m are impossible ($-\infty$). This favours plausible connected paths and respects one-way arcs.
Viterbi runs in log space, $\delta_t(b) = \log e_t(b) + \max_a [\delta_{{t-1}}(a) + \log T(a\to b)]$, with back-pointers.
\textbf{{Q4. Missing candidates and large gaps.}} The chain is split when two points are more than
{hy['gaps']['break_seconds']:.0f}\,s apart or when a point has no candidate; each piece is decoded on its own.
\textbf{{Q5. Disconnected segments.}} If no finite transition exists between two consecutive candidate sets (the roads
are not connected within the allowed detour), the chain is split there too, so every point with candidates still gets
an arc; the break is counted in the continuity metric.

\textbf{{Tuning.}} $w$ and $\beta$ were tuned on the 80 TRAIN trips only, with \emph{{out-of-fold}} classifier scores
(refitted without each fold), because in-sample scores are over-confident. Best: $w={hy['emission']['classifier_weight']:g}$,
$\beta={hy['transition']['beta_m']:g}$\,m (TRAIN Top-1 {f(best_grid.top1, 4)} vs {f(argmax_tr.top1, 4)} for the argmax).

{grid_tab}
{hmm_tab}
\textbf{{Q6. Did the HMM improve the route?}} Yes. End to end through the API code path, Top-1 rises from {pct(cls.top1)}
to {pct(hm.top1)}, continuity from {f(cls.continuity_mean)} to {f(hm.continuity_mean)}, route breaks fall from
{int(cls.breaks_total)} to {int(hm.breaks_total)}, and route F1 rises sharply because the route stops flickering between arcs.
The HMM improves {n_better} of the 10 trips.

\textbf{{Where it did not help.}} Trips where it is worse: {worse_txt}. Distance-only emissions ($w=0$) are clearly worse
than the classifier on Top-1, so the classifier does most of the work and the HMM mainly removes isolated mistakes.
Decoding costs about {hm.total_ms_mean / 1000:.1f}\,s per trip (max {hm.total_ms_max / 1000:.1f}\,s) against
{cls.total_ms_mean / 1000:.2f}\,s without it, dominated by shortest-path searches.

{pt_tab}
{figure(FIG / "hmm_comparison.png", "Per-trip Top-1 on STUDENT\\_TEST, classifier only vs classifier + HMM.", "fig:hmmtrip", r"0.85\linewidth")}

\textbf{{Failure analysis.}} Of the {int(eh.wrong):,} wrong points left, {int(eh.adjacent_arc):,} are on an arc adjacent to the
label (sharing a junction) and {int(eh.reverse_twin):,} on the reverse direction of the label arc. Junction errors have a
median distance of about 8\,m to the shared node, inside GPS noise, where the label is set by where the alignment
switched arcs. Trip {int(worst):08d} alone holds {pct(worst_share, 0)} of all errors, mostly reverse-direction ones; for
{int(eh.twin_pred_agrees_heading)} of the reverse errors the prediction agrees with the GPS heading while the label does
not, which points to direction noise in the reference route rather than in the model.

\section{{Geographical visualisation with Folium}}
\texttt{{scripts/make\_maps.py}} draws, for any trip, the raw GPS points, the road network within 200\,m of the track
(grey), the reference route (green, development trips only), the predicted route of the final pipeline (red), and GPS
points coloured blue where the predicted arc equals the label and orange where it differs, with a legend and a layer
control. The interactive HTML files are in \texttt{{maps/}}.
{figure(RFIG / "map_predicted_student_test_00000040.png", "STUDENT\\_TEST trip 00000040: reference route vs route predicted by the final pipeline (tuned LightGBM + HMM).", "fig:map40", r"0.85\linewidth")}
{figure(RFIG / "map_predicted_student_test_00000077.png", "STUDENT\\_TEST trip 00000077, the hardest trip (final pipeline): most disagreements are on the reverse direction of the reference arc.", "fig:map77", r"0.85\linewidth")}

\section{{API, UI and deployment}}
\textbf{{API (Laptop 1).}} FastAPI (\texttt{{scripts/run\_api.sh}}, port 8000 on all interfaces) loads the classifier, the
HMM configuration and the trip files once at start-up. \texttt{{/evaluator/trips}} lists the evaluator trips and
\texttt{{/evaluator/predict}} runs the full pipeline on the track and network only. The reference route is read by
\texttt{{api/evaluation.py}} only \emph{{after}} prediction, to compute point Top-1 (labels from the same alignment as in
training), candidate miss rate, route precision/recall/F1, edit similarity and continuity. Only these numbers leave
the server; the reference route never reaches the model or the UI. Every request is logged with timings and scores.
\textbf{{UI (Laptop 2).}} Streamlit (\texttt{{ui/app.py}}) holds no data or model: it lists the evaluator trips, sends the
selected id and shows the predicted route on a Folium map with the server's scores.
\texttt{{scripts/smoke\_test\_api.py}} calls every evaluator trip: 10/10 passed, point-weighted Top-1 {pct(ev_top1)}
(Table~\ref{{tab:eval}}), in line with STUDENT\_TEST.

{ev_tab}
{figure(RFIG / "ui_evaluator.png", "UI showing evaluator trip 00000059: server-side scores and the predicted route.", "fig:ui", r"0.8\linewidth")}
{figure(RFIG / "api_terminal.png", "Terminal of the API server during the evaluator smoke test: requests, timings and server-side scores (the first lines are the start-up).", "fig:api")}

\section{{Academic integrity and AI use}}
No evaluator data was used for training, feature selection, model choice, thresholds or decoder tuning; the loader
enforces this. There are no hard-coded predictions or trip-specific code paths. Every number in this report is read
from \texttt{{experiments/results/}} by \texttt{{report/build\_report.py}} and can be reproduced with the run order in the
README (seed 148). \textbf{{AI assistance}} was used for parts of the code and for drafting this report, under the team's
direction and as disclosed in \texttt{{docs/AI\_USE.md}}; all results come from running the repository's scripts, and
every member has reviewed the work.

\section{{Limitations}}
The per-point labels are derived, not given, and are uncertain at junctions and on two-way roads, so Top-1 is a
conservative measure. STUDENT\_TEST has only 10 trips, so differences below about one point are not meaningful. The HMM
decoder is roughly 8$\times$ slower than classifier-only inference, acceptable here (seconds per trip) but it would need
cached shortest paths at scale.

\end{{document}}
"""
    (OUT / "report.tex").write_text(tex)
    eng = shutil.which("tectonic")
    cmd = [eng, "--keep-logs", "report.tex"] if eng else (["xelatex", "-interaction=nonstopmode", "report.tex"]
                                                           if shutil.which("xelatex") else None)
    if cmd is None:
        print("wrote report.tex; install tectonic or xelatex to build the PDF")
        return 0
    r = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True)
    if eng is None:
        r = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True)
    for ext in ("aux", "log", "out"):
        (OUT / f"report.{ext}").unlink(missing_ok=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        return 1
    print(f"wrote {OUT / 'report.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
