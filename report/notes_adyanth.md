# Report fact sheet: Adyanth's sections

**Facts, numbers and figure paths only.** The report text must be written in our own words (assessment policy: reports may be checked for AI-generated text, and anyone can be asked about any claim). Every number below comes from a file in `experiments/results/`; regenerate with the commands in `README.md`.

## Section 2.2 Data & EDA (partition named for every item)

| Fact | Value | Source |
|---|---|---|
| Trips | train 80, student_test 10 (evaluator 10, never read) | `eda_partition_summary.csv` |
| GPS points | train 208,417 (361-19,313 per trip), student_test 19,160 | same |
| Network size | 299-155,003 nodes, 649-307,392 arcs per trip; own network per trip | `inventory.csv` |
| Missing / invalid GPS | 0 NaN, 0 out-of-range, 0 (0,0), 0 duplicate or backwards timestamps | `eda_partition_summary.csv` |
| Sampling | 96.5% of train steps are exactly 1 s (per trip 53-100%) | `inventory.csv` |
| Gaps (train) | 1,216 > 5 s, 135 > 30 s, 8 > 10 min; max 153,064 s (trip 90), mostly while parked | `eda_gaps.csv` |
| Speed outliers (train) | 546 steps > 60 m/s, 513 of them in trips 32/13/1, sustained not spikes | `eda_speed_outliers.csv` |
| Geography | 25 UTM zones; per-trip UTM projection, never degrees | `eda_coord_order.csv` |
| Network quality | 24 self-loops, 1,903 duplicate arcs, ~92% arcs have a reverse twin, largest WCC 69-100% | `eda_network.csv` |
| Routes | 59-2,266 arcs; 14 breaks in 11 trips; 10 trips revisit arcs | `eda_route.csv` |

Figures: `experiments/figures/eda_dt_hist.png`, `eda_speed_hist.png`, `eda_anomalous_trips.png`, `eda_trip_locations.png`, `eda_trajectories.png`, `eda_network.png`, `eda_route.png`.

**Preprocessing / labels** (`label_quality.csv`, `label_quality.png`): monotonic DP alignment; median distance to the aligned arc 4.75 m; 11.1% of train points (0.14% of student_test) are > 50 m away and flagged, mostly in trips 54/33/2, where the route covers only part of the track. These are dropped from training and evaluation.

**Candidates** (`candidate_miss_rate.csv`, `candidate_miss_rate.png`, `candidate_stats.csv`): R = 60 m, K = 30, fallback 150 m chosen on train. Miss rate on reliable labels: train 0.029%, student_test 0.000%; ~12 candidates per point.

**Features and why**: table in `REFERENCE.md` section 5; class separation per feature in `feature_summary.csv`.

## Section 3.2 Per model (Adyanth's models)

All: `GroupKFold(5, shuffle=True, random_state=148)` by trip on train; student_test held out; 36 features; unreliable labels dropped (23,165 train / 27 student_test points). Files: `experiments/results/models/<name>.json`.

| | Logistic Regression | SGD (log loss) | Gaussian NB |
|---|---|---|---|
| A. why suitable | linear, calibrated probabilities, fast, interpretable coefficients | same model as LogReg, fitted by SGD; scales to large data | generative baseline; tests the independence assumption |
| B. key hyperparameters | C = 1.0, L2, class_weight balanced, lbfgs, max_iter 2000, StandardScaler | alpha = 1e-4, L2, balanced, max_iter 100, StandardScaler | var_smoothing 1e-9, empirical priors |
| B. loss / objective | binary log loss (+ L2) | binary log loss (+ L2), SGD | maximum likelihood of class-conditional Gaussians |
| CV macro F1 (mean ± std) | 0.861 ± 0.014 | 0.862 ± 0.013 | 0.813 ± 0.013 |
| CV folds | 0.880, 0.846, 0.869, 0.845, 0.865 | 0.879, 0.846, 0.870, 0.849, 0.867 | 0.821, 0.795, 0.825, 0.822, 0.799 |
| train-fold macro F1 | 0.861 (no gap: no overfitting) | 0.863 | 0.812 |
| CV Top-1 | 0.853 | 0.853 | 0.788 |
| student_test Top-1 | 0.858 | 0.856 | 0.772 |
| student_test macro F1 | 0.861 | 0.861 | 0.806 |
| student_test log loss | 0.128 | 0.128 | 0.836 |
| positive-class precision / recall | 0.614 / 0.962 | 0.616 / 0.959 | 0.502 / 0.948 |
| confusion matrix [[TN, FP], [FN, TP]] | [[213377, 11586], [733, 18400]] | [[213536, 11427], [791, 18342]] | [[206964, 17999], [1004, 18129]] |
| fit time (final fit) | 10.1 s | 6.4 s | 0.7 s |
| inference | 0.7 ms per trip | 0.8 ms per trip | 1.6 ms per trip |

Talking points (put them in your own words):
- Linear models underfit (train F1 = CV F1); the boosting models are ~0.06 macro F1 better.
- Low positive precision at threshold 0.5 comes from `class_weight='balanced'`: many candidates score above 0.5. Top-1 only uses the ranking within each point, which is why Top-1 is the primary metric.
- GNB's log loss (0.84) is 6.5 times LogReg's. Its features are strongly correlated (several distance measures; previous/next heading), which violates the independence assumption and makes it over-confident. That makes it a poor HMM emission model.
- Ablation (LogReg): distance-only features give Top-1 0.546, about the nearest-arc rule (0.530); adding direction gives 0.843; all features give 0.858 (`ablation_*.json`).

## Section 3.1 / 3.3 comparison (shared)
`experiments/results/model_comparison.csv`, `experiments/figures/model_comparison.png` (via `scripts/cv_compare.py`).

## Section 6 Folium maps
- EDA maps: `maps/eda_*.html` (6 trips). Label check: `maps/labels_*.html` (7 trips). Reference vs predicted: `maps/predicted_student_test_*.html` (4 trips).
- Screenshots: `report/figures/map_eda_train_00000033.png`, `map_eda_train_00000072.png`, `map_predicted_student_test_00000040.png`, `map_predicted_student_test_00000077.png`.
- Layers: raw GPS, network within 200 m, reference route (green), predicted route (red), wrong points orange; LayerControl and legend. Code: `scripts/make_maps.py`, `src/pathsnap/viz.py`.

## Pipeline facts (for sections 5 and 7)
`pipeline_summary.csv`, `pipeline_error_types.csv`: exported model LightGBM (best CV Top-1). Student_test Top-1 0.879, argmax continuity 0.867, 646 breaks (the rule: 0.832 / 0.728 / 1,708), ~77 ms per trip. Errors: 43% adjacent arc, 42% reverse twin, 15% other. For 81% of the twin errors with a trusted heading, the prediction matches the driving direction and the label does not: direction noise in the reference route (mostly trip 77).

## Viva checklist (Adyanth must be able to explain)
1. Why per-trip UTM and not degrees; why haversine is used only for EDA speeds.
2. The DP alignment recurrence (stay / advance / skip with penalty, waived across gaps), its O(n·L) cost and the heading term.
3. Why the labels are derived, the 50 m outlier flag, and the partial-route trips.
4. How R and K were chosen (miss rate vs set size, train only) and why K = 5 fails (twins).
5. LogReg, SGD and GNB: assumptions, loss, why the scaler sits inside the Pipeline, why GNB's log loss is poor.
6. GroupKFold by trip: why random row splits would leak (neighbouring points are near-duplicates).
7. The pipeline: no route argument; the evaluator guard; where the HMM plugs in.
