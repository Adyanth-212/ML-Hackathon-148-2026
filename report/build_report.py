"""Build the PDF report from the results in experiments/ (every table number is read from a file).

    python report/build_report.py        # writes report/report.tex and report/report.pdf (xelatex)

Run after the pipeline in README.md. Figures come from experiments/figures/ and report/figures/.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

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


def table(df: pd.DataFrame, cols: list[str], heads: list[str], align: str, caption: str, label: str,
          small: bool = True) -> str:
    rows = [" & ".join(str(v) for v in r) + r" \\" for r in df[cols].itertuples(index=False)]
    size = r"\small" if small else ""
    return (f"\\begin{{table}}[H]\\centering{size}\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\resizebox{{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}}{{!}}{{"
            f"\\begin{{tabular}}{{{align}}}\\toprule\n" + " & ".join(heads) + r" \\ \midrule" + "\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")


def figure(path: Path, caption: str, label: str, width: str = r"\linewidth") -> str:
    if not path.exists():
        return (f"\\fbox{{\\parbox{{0.9\\linewidth}}{{\\centering\\textit{{[{caption} --- add the screenshot as "
                f"\\texttt{{{esc(path.relative_to(ROOT))}}} and re-run report/build\\_report.py]}}}}}}\n\n")
    return (f"\\begin{{figure}}[H]\\centering\\includegraphics[width={width}]{{{path}}}\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\\end{{figure}}\n")


def load_models() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted((RES / "models").glob("*.json"))}


def main() -> int:
    models = load_models()
    comp = pd.read_csv(RES / "model_comparison.csv")
    comp = comp[~comp.is_ablation.fillna(False).astype(bool)].copy()
    ml = comp[comp.cv_macro_f1_mean.notna()].copy()
    rules = comp[comp.cv_macro_f1_mean.isna()]
    tune = pd.read_csv(RES / "tuning_comparison.csv")
    bo = json.loads((RES / "bayes_opt/summary.json").read_text())
    imp = json.loads((RES / "bayes_opt/param_importance.json").read_text())
    grid = pd.read_csv(RES / "hmm_grid.csv")
    hc = pd.read_csv(RES / "hmm_comparison.csv")
    hst = pd.read_csv(RES / "hmm_student_test.csv")
    pipe = pd.read_csv(RES / "pipeline_summary.csv").set_index("variant")
    errs = pd.read_csv(RES / "pipeline_error_types.csv")
    meta = json.loads((ROOT / "artifacts/metadata.json").read_text())
    hmm_cfg = (ROOT / "configs/hmm.yaml").read_text()
    import yaml
    hy = yaml.safe_load(hmm_cfg)

    # ---------- derived numbers used in the text ----------
    cls, hm, rule = pipe.loc["model"], pipe.loc["model_hmm"], pipe.loc["baseline_rule"]
    ev = errs.groupby("variant")[["wrong", "reverse_twin", "adjacent_arc", "other", "twin_with_heading",
                                  "twin_pred_agrees_heading"]].sum()
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
    lt = models["advitha_lgbm_tuned"]
    lu = models["advitha_lgbm"]

    # ---------- tables ----------
    ml["name"] = ml.model.map(esc)
    ml["fam"] = ml.family.map(lambda s: esc(s.replace("gradient boosting", "boosting")))
    ml["cvf1"] = [f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(ml.cv_macro_f1_mean, ml.cv_macro_f1_std)]
    for c, d in [("cv_top1_mean", 3), ("student_test_top1", 3), ("student_test_macro_f1", 3),
                 ("student_test_log_loss", 3), ("route_continuity_argmax", 3)]:
        ml[c + "_s"] = ml[c].map(lambda v, d=d: f(v, d))
    ml["fit_s"] = ml.fit_seconds.map(lambda v: f"{v:.0f}")
    ml["inf_s"] = ml.inference_ms_per_trip.map(lambda v: f"{v:.1f}")
    ml = ml.sort_values("cv_top1_mean", ascending=False)
    comp_tab = table(ml, ["name", "fam", "cvf1", "cv_top1_mean_s", "student_test_top1_s", "student_test_macro_f1_s",
                          "student_test_log_loss_s", "route_continuity_argmax_s", "fit_s", "inf_s"],
                     ["Model", "Family", "CV macro F1", "CV Top-1", "ST Top-1", "ST macro F1", "ST log loss",
                      "ST cont.", "Fit (s)", "ms/trip"], "llcccccccc",
                     "Model comparison, sorted by the selection metric (CV Top-1). CV = 5-fold GroupKFold by trip on "
                     "TRAIN; ST = STUDENT\\_TEST. Continuity is for the classifier-only argmax route (no HMM). "
                     "Inference = predict\\_proba per trip on prepared features.", "tab:comp")
    rule_rows = " ".join(f"{esc(r.model)} ({esc(r.key_config)}): ST Top-1 {f(r.student_test_top1)}."
                         for r in rules.itertuples())

    per_model_rows = []
    for name in ml.model:
        d = models[name]
        cfg = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in d["config"].items() if k not in ("tuned_by", "objective", "eval_metric",
                                                                              "tree_method", "subsample_freq"))
        gap = (d["cv"]["train_macro_f1_mean"] or 0) - d["cv"]["macro_f1_mean"]
        per_model_rows.append({"m": esc(name), "cfg": esc(cfg[:150]), "gap": f"{gap:+.3f}",
                               "cm": esc(str(d["student_test"]["confusion_matrix"]))})
    pm = pd.DataFrame(per_model_rows)
    pm_tab = ("\\begin{table}[H]\\centering\\scriptsize\\caption{Key hyperparameters (initial values; LightGBM-tuned "
              "from Bayesian optimisation), train-fold minus CV macro F1 (over/underfitting), and the STUDENT\\_TEST "
              "confusion matrix [[TN, FP], [FN, TP]] at threshold 0.5.}\\label{tab:permodel}\n"
              "\\begin{tabular}{p{2.6cm}p{8.2cm}cp{3.4cm}}\\toprule Model & Hyperparameters & Train$-$CV F1 & "
              "Confusion matrix \\\\ \\midrule\n"
              + "\n".join(f"{r.m} & {r.cfg} & {r.gap} & {r.cm} \\\\" for r in pm.itertuples())
              + "\n\\bottomrule\\end{tabular}\\end{table}\n")

    tune["u"] = tune.untuned.map(lambda v: f(v, 4))
    tune["t"] = tune.tuned.map(lambda v: f(v, 4))
    tune["c"] = tune.change.map(lambda v: f"{v:+.4f}")
    tune["m"] = tune.metric.map(esc)
    tune["p"] = tune.partition.map(esc)
    tune_tab = table(tune, ["m", "p", "u", "t", "c"], ["Metric", "Partition", "Untuned", "Tuned", "Change"],
                     "llrrr", "Untuned vs Bayesian-tuned LightGBM (same folds, features, seed).", "tab:tune")
    space = pd.DataFrame([{"p": esc(k), "r": f"{v['low']}--{v['high']}", "s": "log" if v.get("log") else "linear",
                           "b": esc(f"{bo['best_params'][k]:.4g}" if isinstance(bo["best_params"][k], float)
                                    else bo["best_params"][k]),
                           "i": f"{imp.get(k, 0):.2f}"} for k, v in bo["search_space"].items()])
    space_tab = table(space, ["p", "r", "s", "b", "i"], ["Parameter", "Range", "Scale", "Best", "Importance"],
                      "lcccc", "Search space, best values (trial %d) and fANOVA importance." % bo["best_trial"],
                      "tab:space")

    g = grid.copy()
    g["v"] = g.variant.map(esc)
    for c in ("top1", "continuity_mean", "route_f1_mean"):
        g[c + "_s"] = g[c].map(lambda v: f(v, 4))
    g = g.sort_values("top1", ascending=False)
    grid_tab = table(g, ["v", "top1_s", "continuity_mean_s", "route_f1_mean_s"],
                     ["Setting (w = classifier weight, $\\beta$ in m)", "Top-1", "Continuity", "Route F1"], "lccc",
                     "HMM decoder grid on the 80 TRAIN trips with out-of-fold classifier scores "
                     "(the objective is Top-1).", "tab:grid")
    h = hc.copy()
    h["v"] = h.variant.map(esc)
    for c, d in [("top1", 4), ("continuity_mean", 3), ("route_precision_mean", 3), ("route_recall_mean", 3),
                 ("route_f1_mean", 3)]:
        h[c + "_s"] = h[c].map(lambda v, d=d: f(v, d))
    h["br"] = h.breaks_total.map(lambda v: f"{int(v):,}")
    h["ms"] = h.decode_ms_mean.map(lambda v: f"{v:.0f}")
    hmm_tab = table(h, ["v", "top1_s", "continuity_mean_s", "br", "route_precision_mean_s", "route_recall_mean_s",
                        "route_f1_mean_s", "ms"],
                    ["Method", "Top-1", "Cont.", "Breaks", "Route P", "Route R", "Route F1", "Decode ms/trip"],
                    "lccccccc", "Classifier-only vs classifier + HMM on STUDENT\\_TEST (final exported classifier).",
                    "tab:hmm")
    pt = piv[[a_col, t_col]].reset_index()
    pt["a"] = pt[a_col].map(lambda v: f(v))
    pt["t"] = pt[t_col].map(lambda v: f(v))
    pt["d"] = (pt[t_col] - pt[a_col]).map(lambda v: f"{v:+.3f}")
    pt["id"] = pt.trip_id.map(lambda v: f"{int(v):08d}")
    pt_tab = table(pt, ["id", "a", "t", "d"], ["Trip", "Classifier only", "+ HMM (tuned)", "Change"], "lccc",
                   "Per-trip Top-1 on STUDENT\\_TEST.", "tab:pertrip")

    worse_txt = ", ".join(f"{int(t):08d} ({f(r[a_col])} $\\to$ {f(r[t_col])})" for t, r in worse.iterrows()) or "none"

    tex = rf"""
\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2cm]{{geometry}}
\usepackage{{fontspec}}
\setmainfont{{TeX Gyre Heros}}
\usepackage{{adjustbox}}\usepackage{{graphicx,booktabs,float,caption,amsmath,xcolor,fancyvrb}}
\usepackage[hidelinks]{{hyperref}}
\captionsetup{{font=small,labelfont=bf}}
\setlength{{\parskip}}{{4pt}}\setlength{{\parindent}}{{0pt}}
\title{{\textbf{{PathSnap: GPS map-matching with a trained candidate classifier and an HMM decoder}}\\[2pt]
\large ML Hackathon 148 --- final report}}
\author{{Team 148: Adyanth, Advitha, Sudhe, Tilak}}
\date{{18 November 2026}}
\begin{{document}}
\maketitle

\begin{{abstract}}
We reconstruct the road route driven by a vehicle from its noisy GPS track. For every GPS point we generate the
nearby road arcs as candidates, score each (point, candidate) pair with a trained classifier, and decode one connected
route with a hidden Markov model solved by Viterbi. Ten classifiers from six families were compared with the same
trip-grouped cross-validation; a Bayesian-tuned LightGBM was selected by cross-validated Top-1. On the 10 held-out
STUDENT\_TEST trips the full pipeline matches \textbf{{{pct(hm.top1)}}} of GPS points to the correct arc
(classifier alone {pct(cls.top1)}, rule baseline {pct(rule.top1)}), raises route continuity from {f(cls.continuity_mean)} to
{f(hm.continuity_mean)} and cuts route breaks from {int(cls.breaks_total)} to {int(hm.breaks_total)}. Most remaining errors sit within GPS noise of
a junction or on the reverse direction of the correct road, where the derived labels themselves are uncertain.
A FastAPI server runs the pipeline on Laptop~1 and a Streamlit UI on Laptop~2 lists the 10 evaluator trips and shows
the predicted route and server-side scores.
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
Partitions: TRAIN (80 trips) for fitting, cross-validation and all tuning; STUDENT\_TEST (10 trips) for held-out
comparison; EVALUATOR (10 trips) only on the demo server. A loader guard refuses to read the evaluator partition
unless the API runs in serve mode, so it cannot leak into development. Seed 148 everywhere.

\section{{Data and exploratory analysis (2.2)}}
\textbf{{Inventory.}} TRAIN has 80 trips and 208,417 GPS points (361--19,313 per trip); STUDENT\_TEST has 10 trips and
19,160 points. Every trip has all four files. Each trip has its own network (299--155,003 nodes, 649--307,392 arcs),
so arc ids are local and the learning problem is framed per (point, candidate) pair rather than as a global
arc vocabulary.

\textbf{{Missing values and anomalies (both partitions).}} There are no NaN rows, out-of-range or (0,0) coordinates,
duplicate or backwards timestamps. 96.5\% of TRAIN steps are exactly 1\,s; 1,216 TRAIN steps exceed 5\,s and 8 exceed
10 minutes (the longest, 42\,h in trip 00000090, is a parked vehicle). 546 TRAIN steps imply speeds above 60\,m/s, almost
all in three trips and sustained rather than single spikes, so no points were deleted. The networks contain 24
self-loops and 1,903 duplicate arcs, and about 92\% of arcs have a reverse twin (two-way roads), which makes
direction a central difficulty.

\textbf{{Coordinates.}} The trips lie in 25 different UTM zones. Each trip is projected to the UTM zone of its median
point; Euclidean distances on degrees are never used.

\textbf{{Labels.}} The reference route is an arc sequence, not a label per point, so per-point labels are derived by a
monotonic dynamic-programming alignment: each point is assigned to one route arc, the route position can only move
forward, and the cost is point-to-arc distance plus a heading-mismatch term. The median distance from a point to its
aligned arc is 4.75\,m. Points more than 50\,m from their aligned arc (11.1\% of TRAIN, mostly three trips where the route
covers only part of the track; 0.14\% of STUDENT\_TEST) are flagged unreliable and excluded from training and scoring.
These labels are an assumption: at junctions and on two-way roads they are uncertain, which matters in Section~\ref{{sec:hmm}}.

\textbf{{Candidates.}} $R$ and $K$ were chosen on TRAIN by the candidate miss rate (true arc not among the candidates)
against candidate-set size. With $R=60$\,m, $K=30$ the miss rate on reliable points is 0.029\% on TRAIN and 0\% on
STUDENT\_TEST, with about 12 candidates per point and a positive rate of 9\%.

\textbf{{Features and why.}} All 36 features use only GPS and network geometry: \emph{{distance}} (point-to-arc distance,
rank and gap to the closest candidate, projection position) because the true arc is usually closest;
\emph{{direction}} (cosine between GPS heading and arc bearing, also for the previous and next points) because about half
of all candidate rows are reverse twins with identical distance; \emph{{context}} (distances of the neighbouring points to
the same arc) because one noisy point can sit near the wrong road while its neighbours do not; \emph{{motion}} (time steps,
speeds, acceleration, turn angle); \emph{{network}} (arc length, node degrees, reverse-twin flag); \emph{{sequence}}
(agreement with the neighbours' best arc under a label-free rule) and \emph{{density}} (number of nearby candidates). An
ablation with logistic regression shows the jump: distance features alone give STUDENT\_TEST Top-1 0.546, adding
direction gives 0.843, all features 0.858.

{figure(FIG / "eda_trajectories.png", "GPS trajectories (TRAIN sample).", "fig:traj", r"0.8\linewidth")}
{figure(FIG / "eda_network.png", "Road-network size and connectivity per trip.", "fig:net", r"0.8\linewidth")}
{figure(FIG / "candidate_miss_rate.png", "Candidate miss rate against radius $R$ and cap $K$ (TRAIN).", "fig:miss", r"0.75\linewidth")}
{figure(RFIG / "map_eda_train_00000033.png", "Folium EDA map (TRAIN trip 00000033): GPS points and road network.", "fig:eda_map", r"0.75\linewidth")}

\section{{Models (3)}}
\subsection{{Comparison and final model (3.1)}}
Every model is evaluated by one shared harness: the same 36 features, the same trip folds
(GroupKFold(5, shuffle, seed 148) by trip, so no trip appears in both training and validation), the same metrics and the
same STUDENT\_TEST scoring. The \textbf{{primary metric is observation-level Top-1}}: for each point, is the highest-scored
candidate the label arc? It needs no threshold and is exactly what the route requires. Candidate-level macro F1
(threshold 0.5) and log loss are reported alongside it.

{comp_tab}
Rule baselines (no learning): {rule_rows}

\textbf{{Final model: {esc(sel)}}}, selected by the rule fixed before tuning: the highest cross-validated Top-1. It
also has the best CV macro F1 and the lowest STUDENT\_TEST log loss, i.e. the best-calibrated probabilities, which matters
because the HMM uses them directly as emissions. XGBoost is marginally higher on STUDENT\_TEST Top-1, but the gap
(about 25 of 19,133 points) is far inside the fold-to-fold spread, and choosing on a 10-trip test set would be selecting on
the test data. The boosting models are statistically tied; the features matter more than the library. Its cost is
speed: about {lt['inference_ms_per_trip']:.0f}\,ms per trip for scoring, still far below one second.

\subsection{{Per model (3.2)}}
\textbf{{Gradient boosting (LightGBM, XGBoost, HistGradientBoosting).}} Tree ensembles fitted to the gradient of the
binary log loss. Suitable because they need no scaling, learn interactions such as distance $\times$ heading $\times$
context, and give good probabilities. LightGBM grows trees leaf-wise, XGBoost depth-wise with second-order updates,
HistGB is the scikit-learn implementation. No class weighting: Top-1 only needs the ranking within a point, and
unweighted log loss keeps the probabilities calibrated. Limitation: mild overfitting (train-fold F1 about 0.03 above CV).

\textbf{{Bagged trees (Random Forest, Extra Trees) and a single Decision Tree.}} Random Forest averages deep trees grown
on bootstrap samples with random feature subsets; Extra Trees also randomises split thresholds; the single tree is the
interpretable baseline that shows what the ensembles add. All use balanced class weights. They rank below boosting,
and their probabilities are less calibrated (higher log loss).

\textbf{{Linear and probabilistic (Logistic Regression, SGD, Gaussian NB).}} Logistic regression (L2, balanced, scaler
inside the pipeline) and the same model fitted by SGD are fast and interpretable but underfit: train and CV F1 are equal,
about 0.06 below boosting. Gaussian NB assumes independent features, which is violated by the several correlated
distance and heading features, so it is over-confident (log loss 0.84), a poor emission model.

{pm_tab}

\subsection{{Plots (3.3)}}
{figure(FIG / "cv_macro_f1.png", "CV macro F1 per model (TRAIN): mean $\\pm$ std over the 5 folds, each fold shown.", "fig:cvf1", r"0.85\linewidth")}
{figure(FIG / "model_comparison.png", "CV macro F1, STUDENT\\_TEST Top-1 and STUDENT\\_TEST log loss per model.", "fig:compfig")}
{figure(FIG / "confusion_matrices.png", "STUDENT\\_TEST confusion matrices (threshold 0.5, row-normalised; 1 = true arc).", "fig:cm")}
No model in the final comparison is trained by epochs (trees are added one by one, linear models converge in a solver),
so there are no train/validation loss curves; we do not plot substitutes.

\section{{Bayesian hyperparameter optimisation (4)}}
LightGBM, the best model before tuning, was tuned with Optuna's TPE sampler (multivariate, seed 148, 10 random
start-up trials) and a median pruner. The objective is the same as the selection metric: mean CV Top-1 on TRAIN with
the harness's folds; STUDENT\_TEST was used only once, after the search. Trial 0 was the untuned configuration, so the
search record starts at the baseline ({f(bo['baseline_trial_cv_top1'], 4)}, identical to the harness).
{bo['n_trials']} trials ran ({bo['n_complete']} complete, {bo['n_pruned']} pruned), {bo['trial_compute_seconds_total'] / 60:.0f} minutes on 8 CPU threads,
200 model fits in total. Every trial is stored (\texttt{{experiments/results/bayes\_opt/}}).

{space_tab}
{tune_tab}
{figure(FIG / "bayes_opt_history.png", "Optimisation history: every trial, best so far, untuned baseline dashed.", "fig:bohist", r"0.8\linewidth")}
The gain is small but consistent: CV Top-1 +0.0012 (better in 4 of 5 folds) and STUDENT\_TEST Top-1 +0.0008. The
hand-picked defaults were already near the optimum for these features. After the random start-up phase TPE converged
on many leaves (about 200--255), a low learning rate and more trees; \texttt{{num\_leaves}} explains about 70\% of the
variation. The price is a 3.5$\times$ longer fit, 4.6$\times$ slower inference and a slightly larger train/CV gap.
The pruner never fired because all configurations score within 0.004 of each other.

\section{{HMM and Viterbi decoding (5)}}\label{{sec:hmm}}
\textbf{{States.}} At time $t$ the hidden states are the candidate arcs of point $t$ (a ragged set, at most 30), each
mapped to its arc id. \textbf{{Emission.}} $\log e_t(a) = \operatorname{{logsoftmax}}_a\big(w \log \tilde p_t(a) + (1-w)\log
\mathcal{{N}}(d_t(a);0,\sigma)\big)$, where $\tilde p$ is the classifier probability normalised over the point's
candidates (floored at $10^{{-6}}$), $d$ the point-to-arc distance and $w$ the classifier weight.
\textbf{{Transition.}} From arc $a$ at $t-1$ to arc $b$ at $t$ we compute the driving distance on the directed network
(rest of $a$, shortest path by Dijkstra from the end of $a$ to the start of $b$, part of $b$; same-arc moves are the
distance along the arc) and compare it with the straight-line distance between the two GPS points:
$\log T(a\to b) = \operatorname{{logsoftmax}}_b\big(-|\text{{route}} - \text{{straight}}|/\beta\big)$. Paths longer than
$2\times$straight$+200$\,m are impossible ($-\infty$). This encodes that a vehicle drives a plausible connected path
and respects one-way arcs. \textbf{{Viterbi}} runs in log space:
$\delta_t(b) = \log e_t(b) + \max_a [\delta_{{t-1}}(a) + \log T(a\to b)]$ with back-pointers.
\textbf{{Gaps, missing candidates and disconnections.}} The chain is split when two points are more than
{hy['gaps']['break_seconds']:.0f}\,s apart, when a point has no candidate, or when no finite transition exists; each piece is decoded
separately, so every point with candidates still gets an arc.

\textbf{{Tuning.}} $w$ and $\beta$ were tuned on the 80 TRAIN trips only. Classifier scores for TRAIN are
\emph{{out-of-fold}}: for each fold the classifier was refitted without that fold, because in-sample scores would be
over-confident and push the decoder to trust the classifier too much. Best: $w={hy['emission']['classifier_weight']:g}$, $\beta={hy['transition']['beta_m']:g}$\,m
(TRAIN Top-1 {f(best_grid.top1, 4)} vs {f(argmax_tr.top1, 4)} for the argmax).

{grid_tab}
{hmm_tab}
\textbf{{Did it help?}} Yes. End to end through the API code path, Top-1 rises from {pct(cls.top1)} to
{pct(hm.top1)}, continuity from {f(cls.continuity_mean)} to {f(hm.continuity_mean)}, route breaks fall from {int(cls.breaks_total)} to
{int(hm.breaks_total)} and route F1 against the reference sequence rises sharply because the predicted route stops flickering
between arcs. The HMM improves {n_better} of the 10 trips.

\textbf{{Where it did not help.}} Trips where it is worse: {worse_txt}. Distance-only emissions ($w=0$) are clearly worse
than the classifier on Top-1, so the classifier is doing most of the work and the HMM mainly removes isolated mistakes.
The cost is time: decoding takes about {hm.total_ms_mean / 1000:.1f}\,s per trip on average (max {hm.total_ms_max / 1000:.1f}\,s) against
{cls.total_ms_mean / 1000:.2f}\,s without it, dominated by shortest-path searches.

{pt_tab}
{figure(FIG / "hmm_comparison.png", "Per-trip Top-1 on STUDENT\\_TEST, classifier only vs classifier + HMM.", "fig:hmmtrip", r"0.85\linewidth")}

\textbf{{Failure analysis.}} Of the {int(eh.wrong):,} wrong points left, {int(eh.adjacent_arc):,} are on an arc adjacent to the label (sharing a
junction) and {int(eh.reverse_twin):,} on the reverse direction of the label arc. Junction errors have a median distance of about
8\,m to the shared node, inside GPS noise, where the label itself is set by where the alignment switched arcs. Trip
{int(worst):08d} alone holds {pct(worst_share, 0)} of all errors, mostly reverse-direction ones; for {int(eh.twin_pred_agrees_heading)} of the reverse errors the
prediction agrees with the GPS heading while the label does not, which points to direction noise in the reference
route rather than in the model. Counting those as label noise would give Top-1
{pct(1 - (eh.wrong - eh.twin_pred_agrees_heading) / hm.n_obs)}. We report the measured number.

\section{{Folium maps (6)}}
Maps are generated by \texttt{{scripts/make\_maps.py}} with layers for the raw GPS points, the road network near the
track, the reference route (green, STUDENT\_TEST only, offline) and the predicted route (red), wrong points in orange,
a legend and a layer control. HTML files are in \texttt{{maps/}}.
{figure(RFIG / "map_predicted_student_test_00000040.png", "STUDENT\\_TEST trip 00000040: reference route (green) vs predicted route (red), classifier-only prediction of the untuned LightGBM; orange points differ from the label.", "fig:map40", r"0.8\linewidth")}
{figure(RFIG / "map_predicted_student_test_00000077.png", "STUDENT\\_TEST trip 00000077, the hardest trip (classifier-only prediction): most disagreements are on the reverse direction of the reference arc.", "fig:map77", r"0.8\linewidth")}

\section{{API, UI and deployment (7)}}
\textbf{{Laptop 1}} runs FastAPI (\texttt{{scripts/run\_api.sh}}, port 8000 on all interfaces). It loads the classifier, the
HMM configuration and the trip files once at start-up. \texttt{{/evaluator/trips}} lists the 10 evaluator trips and
\texttt{{/evaluator/predict}} runs the full pipeline on the track and network only; the reference route is read by
\texttt{{api/evaluation.py}} \emph{{after}} prediction to compute point Top-1 (with labels from the same alignment as
training), candidate miss rate, route precision/recall/F1, edit similarity and continuity. Only these numbers leave
the server; the reference route never reaches the model input or the UI. Each request is logged with timings and
scores in the Laptop 1 terminal. \textbf{{Laptop 2}} runs the Streamlit UI (\texttt{{ui/app.py}}), which holds no data or
model: it lists the evaluator trips, sends the selected id, and shows the predicted route on a Folium map with the
metrics. \texttt{{scripts/smoke\_test\_api.py}} calls every evaluator trip and checks success.

{figure(RFIG / "ui_evaluator.png", "UI on Laptop 2 showing an evaluator trip.", "fig:ui", r"0.9\linewidth")}
{figure(RFIG / "api_terminal.png", "API log on Laptop 1 during the evaluator run.", "fig:api", r"0.9\linewidth")}

\section{{Integrity and AI use (8)}}
No evaluator data was used for training, feature selection, model choice, thresholds or decoder tuning; the loader
enforces this. There are no hard-coded predictions or trip-specific code paths. Every number in this report is read
from \texttt{{experiments/results/}} by \texttt{{report/build\_report.py}} and can be reproduced with the run order in the
README (seed 148). \textbf{{AI assistance:}} code across the pipeline and the text of this report were produced with
Claude Code (Anthropic, Claude Opus) under the team's direction, as listed in \texttt{{docs/AI\_USE.md}}; all results come
from running the repository's scripts, and every member has reviewed the work.

\section{{Limitations}}
The per-point labels are derived, not given, and are uncertain at junctions and on two-way roads; Top-1 is therefore a
conservative measure. The STUDENT\_TEST set has only 10 trips, so differences below about one point are not
meaningful. The HMM decoder is roughly 8$\times$ slower than classifier-only inference, which is acceptable here
(seconds per trip) but would need caching of shortest paths at scale.

\end{{document}}
"""
    (OUT / "report.tex").write_text(tex)
    if shutil.which("xelatex"):
        for _ in range(2):
            r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=OUT,
                               capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:])
            return 1
        for ext in ("aux", "log", "out"):
            (OUT / f"report.{ext}").unlink(missing_ok=True)
        print(f"wrote {OUT / 'report.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
