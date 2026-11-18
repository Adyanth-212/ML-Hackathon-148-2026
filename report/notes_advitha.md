# Report fact sheet: Advitha's sections (3.1, 3.2 boosting models, 3.3 figures, 4)

**Facts, numbers and figure paths only.** Write the report text in your own words (reports may be checked for AI-generated text, and anyone can be asked about any claim). Every number here comes from a file in `experiments/results/`; the commands to regenerate it are at the end.

All models: 36 features (`artifacts/feature_list.json`), `GroupKFold(5, shuffle=True, random_state=148)` by trip on the 80 train trips, student_test (10 trips) held out, unreliable labels dropped (23,165 train / 27 student_test points), seed 148. Primary metric: observation-level Top-1 (the top-scored candidate is the label arc).

## Section 3.2 Per model (Advitha's models)

Files: `experiments/results/models/advitha_{lgbm,hgb,xgb}.json`, scripts `scripts/models/advitha_*.py`.

| | LightGBM | HistGradientBoosting (sklearn) | XGBoost |
|---|---|---|---|
| A. why suitable | boosted trees: no scaling needed, learns interactions (distance x heading x context); leaf-wise growth, fast histogram binning | same family, sklearn implementation; a check that the result is not specific to one library | depth-wise trees with second-order updates: a different inductive bias from LightGBM's leaf-wise trees |
| B. key hyperparameters | 400 trees, lr 0.05, num_leaves 63, min_child_samples 50, subsample 0.8, colsample 0.8, L2 1.0 | max_iter 300, lr 0.1, max_leaf_nodes 63, min_samples_leaf 50, L2 1.0, no early stopping | 400 trees, lr 0.05, max_depth 8, min_child_weight 5, subsample 0.8, colsample 0.8, L2 1.0, hist |
| B. loss / objective | binary log loss | log loss | binary:logistic (log loss) |
| class weighting | none (Top-1 only needs the ranking within a point; unweighted log loss keeps probabilities calibrated for the HMM) | none | none |
| CV macro F1 (mean ± std) | 0.925 ± 0.018 | 0.923 ± 0.018 | 0.924 ± 0.019 |
| CV macro F1 folds | 0.943, 0.900, 0.937, 0.905, 0.938 | 0.943, 0.901, 0.935, 0.902, 0.936 | 0.943, 0.899, 0.937, 0.904, 0.938 |
| train-fold macro F1 | 0.952 (gap 0.028) | 0.957 (gap 0.034) | 0.956 (gap 0.031) |
| CV Top-1 (mean) | 0.8639 | 0.8627 | 0.8632 |
| CV Top-1 folds | 0.897, 0.817, 0.887, 0.825, 0.893 | 0.896, 0.818, 0.887, 0.823, 0.890 | 0.897, 0.814, 0.887, 0.826, 0.893 |
| student_test Top-1 | 0.8787 | 0.8773 | 0.8809 |
| student_test macro F1 | 0.933 | 0.931 | 0.934 |
| student_test log loss | 0.0473 | 0.0481 | 0.0472 |
| positive-class precision / recall | 0.898 / 0.855 | 0.898 / 0.849 | 0.899 / 0.857 |
| confusion matrix [[TN, FP], [FN, TP]] | [[223103, 1860], [2765, 16368]] | [[223127, 1836], [2895, 16238]] | [[223129, 1834], [2742, 16391]] |
| final fit time | 11.7 s | 16.1 s | 13.8 s |
| inference (predict_proba) | 37 ms per trip | 24 ms per trip | 11 ms per trip |

Talking points:
- The three boosting models are within 0.0012 CV Top-1 of each other, far less than the fold-to-fold spread (std ~0.035). They are statistically tied; the features matter more than the boosting library.
- All three beat the linear models by ~0.06 CV macro F1 and ~0.01 CV Top-1, and have a small train-vs-CV gap (~0.03): mild overfitting, not severe.
- Folds 1 and 3 are hardest for the boosting and linear models (Top-1 ~0.82 vs ~0.89). Because the same folds are hardest for the boosting and linear models alike (GNB differs only slightly), the difficulty comes from those trips, not from any one model. (Which trips: `experiments/results/cv_folds.csv`; worth a look at their maps before writing why.)
- Positive recall at threshold 0.5 is 0.85, but Top-1 does not use a threshold: it takes the best candidate per point. That is why Top-1 is the primary metric and macro F1 is secondary.

## Section 3.1 Final model choice and justification

**Chosen: LightGBM, Bayesian-tuned (`advitha_lgbm_tuned`), exported to `artifacts/model.joblib`.**

Rule, fixed before tuning (`scripts/export_artifacts.py`): the highest cross-validated Top-1 on train. Student-test numbers are reported, not used to choose.

| Evidence | Value | Source |
|---|---|---|
| CV Top-1 ranking | lgbm_tuned 0.8651 > lgbm 0.8639 > xgb 0.8632 > hgb 0.8627 > logreg 0.8534 > sgd 0.8532 > gnb 0.7881 | `artifacts/metadata.json` |
| Best CV macro F1 | lgbm_tuned 0.9253 | `model_comparison.csv` |
| Best student_test log loss | lgbm_tuned 0.0470 (xgb 0.0472, lgbm 0.0473) | same |
| student_test Top-1 | xgb 0.8809 is highest; lgbm_tuned 0.8796 is 0.0013 behind (25 of 19,133 points) | same |
| End to end (`predict_trip`, the API path) | Top-1 0.8796, continuity 0.867, 653 breaks; rule baseline 0.832 / 0.728 / 1,708 | `pipeline_summary.csv` |
| Latency end to end | mean 209 ms per trip, max 530 ms (trip 77, 3,911 points) | same |

Reasons (put them in your own words):
1. It is best on the pre-agreed selection metric (CV Top-1) and on CV macro F1.
2. It has the lowest log loss, i.e. the best-calibrated probabilities. That matters because the HMM uses them directly as emission probabilities.
3. XGBoost is 0.0013 higher on student_test, but choosing by student_test would be selecting on a test set of only 10 trips, and the gap is far inside fold-to-fold noise. We kept the rule.
4. Cost: inference is 4.6x slower than untuned LightGBM (169 vs 37 ms per trip of scoring), still well under a second per trip, so acceptable for the demo. If latency ever mattered, XGBoost (11 ms) is the near-equal fallback.

Limitations to state: the margin over the other boosting models is within noise. Any of the three would be a defensible choice; the rule is what makes the choice principled.

## Section 3.3 Comparison figures

- `experiments/figures/cv_macro_f1.png`: CV macro F1 per model, mean ± std with each fold as a dot (dot plot because the axis does not start at 0)
- `experiments/figures/confusion_matrices.png`: student_test confusion matrix for every model (threshold 0.5, row-normalised, counts printed)
- `experiments/figures/model_comparison.png`: CV macro F1, student_test Top-1, student_test log loss side by side (light bars = ablations)
- Table: `experiments/results/model_comparison.csv` (`scripts/cv_compare.py`)
- No epoch loss curves for tree models: boosting has no epochs. The MLP (Tilak) is the model with real train/val loss curves.

## Section 4 Bayesian optimisation

Code `scripts/optimize.py`, config `configs/bayes_opt.yaml`, record `experiments/results/bayes_opt/` (`study.db` = full Optuna storage, `trials.csv` = every trial, `best_params.json`, `summary.json`, `param_importance.json`), log `experiments/logs/bayes_opt.log`.

| Item | Value |
|---|---|
| Model tuned | LightGBM (the best model before tuning) |
| Method | Optuna 5.1, TPE sampler (multivariate, seed 148, 10 random start-up trials), median pruner |
| Objective | maximise mean CV Top-1 on train, same 5 trip folds and Top-1 definition as the harness. student_test and evaluator not used |
| Trial 0 | the untuned configuration (enqueued), so the record starts at the baseline: 0.8639, matching the harness |
| Search space | n_estimators 200-1000 (step 50); learning_rate 0.01-0.2 (log); num_leaves 15-255 (log); min_child_samples 10-400 (log); subsample 0.5-1; colsample_bytree 0.4-1; reg_lambda 0.001-30 (log); reg_alpha 0.001-10 (log). Reasons per parameter in `configs/bayes_opt.yaml` |
| Trials | 40, all complete, 0 pruned |
| Compute cost | 85.5 min wall on 8 CPU threads (37-197 s per trial, median 142 s); 5 model fits per trial = 200 fits |
| Trial values | CV Top-1 0.8609-0.8651, median 0.8646 |
| Best (trial 32) | n_estimators 800, learning_rate 0.0141, num_leaves 254, min_child_samples 20, subsample 0.915, colsample_bytree 0.789, reg_lambda 0.045, reg_alpha 1.52 |
| Parameter importance (fANOVA) | num_leaves 0.70, min_child_samples 0.10, learning_rate 0.07, n_estimators 0.05, rest < 0.04 |

Untuned vs tuned (`experiments/results/tuning_comparison.csv`, same partitions, folds, features, seed):

| Metric | Partition | Untuned | Tuned | Change |
|---|---|---|---|---|
| CV macro F1 | train CV | 0.9246 ± 0.0184 | 0.9253 ± 0.0188 | +0.0008 |
| CV Top-1 | train CV | 0.8639 | 0.8651 | +0.0012 (better in 4 of 5 folds) |
| CV log loss | train CV | 0.0637 | 0.0638 | +0.0001 |
| train-fold macro F1 | train CV | 0.952 | 0.963 | +0.011 (bigger trees fit train more) |
| Top-1 | student_test | 0.8787 | 0.8796 | +0.0008 |
| macro F1 | student_test | 0.9330 | 0.9334 | +0.0004 |
| log loss | student_test | 0.0473 | 0.0470 | -0.0004 |
| route continuity (argmax) | student_test | 0.8669 | 0.8666 | -0.0003 |
| final fit time | train | 11.7 s | 40.8 s | 3.5x |
| inference | student_test | 37 ms/trip | 169 ms/trip | 4.6x |

Figures: `experiments/figures/bayes_opt_history.png` (every trial, best-so-far line, untuned dashed), `bayes_opt_importance.png`.

Talking points:
- The gain is real but small: +0.12 points CV Top-1, consistent in 4 of 5 folds and on student_test. The hand-picked defaults were already close to the optimum for these features. Say this plainly; it is a result, not a failure.
- TPE learned the direction: after the 10 random trials, most trials match or beat the baseline (history plot), converging on many leaves (~200-255), a low learning rate (~0.01-0.03), more trees (800-950) and small leaves (min_child_samples 20-35).
- num_leaves explains ~70% of the variation: tree complexity is the lever that matters. The regularisers matter little at 2M rows.
- Cost of the gain: bigger trees mean 3.5x fit time and 4.6x inference time, and a larger train/CV macro F1 gap (0.038 vs 0.028), so slightly more overfitting.
- The pruner never fired: every trial's running mean stayed above the median of earlier trials at the same fold, because all configurations score within 0.004 of each other. It cost nothing to have it.
- Why the objective is Top-1 and not macro F1: Top-1 is the task metric and the selection rule. Macro F1 depends on the 0.5 threshold, which the decoder never uses.

## Regenerate

```bash
python scripts/optimize.py                     # ~85 min; resumes if interrupted; --report rewrites outputs only
python scripts/models/advitha_lgbm_tuned.py    # harness run with the best params (~4 min)
python scripts/compare_tuning.py               # tuning_comparison.csv
python scripts/cv_compare.py                   # model_comparison.csv + the three comparison figures
python scripts/export_artifacts.py             # exports the best-CV-Top-1 model (now advitha_lgbm_tuned)
python scripts/eval_pipeline.py                # end-to-end numbers through predict_trip
```

## Viva checklist (Advitha must be able to explain)
1. Every feature group and why it helps (`REFERENCE.md` section 5); why features use only GPS + network, never labels.
2. The harness: why GroupKFold by trip (neighbouring points are near-duplicates, so a row split would leak), why folds are fixed, how Top-1 is computed and why points without the true arc among the candidates count as wrong.
3. Boosting: what a tree ensemble fits (gradient of log loss), leaf-wise (LightGBM) vs depth-wise (XGBoost), what num_leaves / min_child_samples / learning rate / subsampling do.
4. Why no class weights; why the selection used CV and not student_test.
5. TPE: how it models good vs bad trials and samples where the ratio is high; why it beats grid search for 8 parameters; what the median pruner does.
6. Untuned vs tuned: the size of the gain relative to fold noise, and the latency cost.
