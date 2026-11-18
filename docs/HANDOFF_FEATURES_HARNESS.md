# Features, harness and model round: handoff

Branch `feat/advitha`. Built on Advitha's first drafts (feature list and model choices) and the shared foundation on `main` (labels, candidates). Runtimes measured on an 8-core VM.

## How to run

```bash
python scripts/build_dataset.py        # ~2 s: feature table -> data/processed_local/features/ (gitignored)
python scripts/models/advitha_lgbm.py  # ~80 s each for the boosting models (5 CV fits + final fit)
python scripts/models/advitha_hgb.py   # ~110 s
python scripts/models/advitha_xgb.py   # ~80 s
python scripts/models/adyanth_logreg.py  # ~90 s incl. two ablations
python scripts/models/adyanth_gnb.py     # ~10 s
python scripts/cv_compare.py           # model_comparison.csv + figure
```

## Feature table (`src/pathsnap/features.py`, `artifacts/feature_list.json`)

One row per (observation, candidate arc), the candidate-table contract plus 36 numeric features in 7 groups. Everything is computed from GPS and the network in per-trip UTM metres; labels are never read (`tests/test_features_harness.py::test_no_label_leakage`).

| group | features | why |
|---|---|---|
| distance | dist_m, dist_rank, dist_minus_min, dist_ratio_min, proj_frac, proj_outside, dist_from_node_m, dist_to_node_m, dist_over_len | the GPS point should be close to the true arc, and closer than its competitors |
| direction | heading_cos (central), heading_cos_prev, heading_cos_next, heading_valid, heading_cos_gap | ~92% of arcs have a geometrically identical reverse twin; only direction separates them |
| context | ctx_mean_dist_m, ctx_max_dist_m, ctx_mean_gap_m (points t-2..t+2 vs the same arc) | one noisy point can be near the wrong arc; its neighbours usually are not |
| motion | dt_prev_s, dt_next_s, speed_prev/next_mps, accel_mps2, turn_deg | fast points belong on long, major arcs; turns happen at junctions |
| network | arc_length_m, from/to in/out degree, has_reverse | road type and junction structure |
| sequence | prev/next_rule_same, prev/next_rule_adjacent | agreement with the neighbours' rule-based best arc (distance + heading, no labels) |
| density | n_candidates, n_within_10m, n_within_25m | ambiguity: parallel roads and dense junctions |

The per-feature class separation on TRAIN is in `experiments/results/feature_summary.csv`. The strongest are the sequence flags, dist_ratio_min, proj_outside and the distance/context gaps, followed by heading_cos.

## Harness (`src/pathsnap/harness.py`)

`evaluate_model(name, family, estimator, config, features=None, max_train_obs=None)`:
- `GroupKFold(n_splits=5, shuffle=True, random_state=148)` by trip on the 80 train trips; the fold of every trip is in `experiments/results/cv_folds.csv` (16 trips per fold)
- drops the 23,165 train and 27 student_test observations with unreliable derived labels (> 50 m from the route) and records the counts
- metrics: candidate-level macro F1 / precision / recall / log loss at threshold 0.5, observation-level Top-1, train-fold F1 (overfitting check), confusion matrix, argmax route continuity, candidate miss rate, fit seconds, inference ms per trip and per point
- writes `experiments/results/models/<name>.json` and the student_test probabilities to `data/processed_local/preds/<name>_student_test.parquet` (for the HMM)

## Results (`experiments/results/model_comparison.csv`)

| model | family | CV macro F1 | train F1 | CV Top-1 | st. Top-1 | st. macro F1 | st. log loss | continuity (argmax) | fit s | ms / trip |
|---|---|---|---|---|---|---|---|---|---|---|
| advitha_xgb | XGBoost | 0.924 ± 0.019 | 0.956 | 0.863 | **0.881** | 0.934 | **0.047** | 0.869 | 13.8 | 10.5 |
| advitha_lgbm | LightGBM | 0.925 ± 0.018 | 0.952 | 0.864 | 0.879 | 0.933 | 0.047 | 0.867 | 11.7 | 37.0 |
| advitha_hgb | sklearn HistGB | 0.923 ± 0.018 | 0.957 | 0.863 | 0.877 | 0.931 | 0.048 | 0.863 | 16.1 | 24.0 |
| adyanth_logreg | logistic regression | 0.861 ± 0.014 | 0.861 | 0.853 | 0.858 | 0.861 | 0.128 | 0.804 | 10.1 | 0.7 |
| adyanth_gnb | Gaussian NB | 0.813 ± 0.013 | 0.812 | 0.788 | 0.772 | 0.806 | 0.836 | 0.779 | 0.7 | 1.6 |
| ablation: LogReg, distance + density | – | 0.771 | – | 0.545 | 0.546 | 0.769 | 0.259 | 0.284 | – | – |
| ablation: LogReg, + direction | – | 0.838 | – | 0.825 | 0.843 | 0.837 | 0.157 | 0.740 | – | – |
| rule: nearest arc | not ML | – | – | – | 0.530 | – | – | – | – | – |
| rule: nearest + heading (w = 20 m) | not ML | – | – | – | 0.832 | – | – | – | – | – |

Reading:
- The three boosting models are practically tied: CV macro F1 differences (0.001-0.002) are far inside one fold standard deviation (~0.018), and student_test Top-1 differs by 0.004 (~75 of 19,133 points). They beat logistic regression by about +0.06 macro F1 and +0.02 Top-1, and have about a third of its log loss.
- Boosting train F1 is ~0.03 above CV F1: mild overfitting, a reason to regularise in the Bayesian search. The linear models show no gap (they underfit).
- Direction features are worth +0.30 Top-1 to logistic regression (0.546 -> 0.843). Context, motion, network and sequence add another +0.015.
- Gaussian NB has the worst log loss by far (0.84): its independence assumption is badly violated by our correlated features, so its probabilities are over-confident. That makes it a poor HMM emission model even though its Top-1 is reasonable.
- Top-1 counts observations whose label arc is not a candidate as wrong; the candidate miss rate on student_test reliable labels is 0.000%, so the ceiling is 1.0.

## Next steps

1. **Pick the final model:** XGBoost or LightGBM. Both are within noise; XGBoost is fastest at inference (10.5 ms per trip) and has the best student_test Top-1 and log loss. Choose on CV results, not student_test.
2. **Bayesian optimisation** (`scripts/optimize.py`, Optuna TPE or GPSampler) on the chosen model, objective = CV mean Top-1 (or log loss), with the same folds from the harness. Save every trial and compare untuned vs tuned.
3. **Export** the tuned model to `artifacts/` and plug it into `pipeline.predict_trip` as `artifacts["scorer"] = lambda geom, cands: model.predict_proba(build_features(geom, cands)[FEATURE_COLUMNS])[:, 1]`.
4. **HMM (Sudhe):** the student_test probabilities are in `data/processed_local/preds/`; the argmax continuity of 0.87 is the number the decoder has to improve.
5. **Other members' models** (Sudhe: RF/ET/DT, Tilak: KNN/MLP): call `evaluate_model` the same way; use `max_train_obs=` for KNN/SVM.
