# PathSnap — Full Requirements & Project Plan

Built from the five PDFs in this folder: Main Guidelines, Dataset Exploration, Models and Training Guidelines, Deliverables, and the Assessment and Plagiarism Policy. Section references like *(Models §7)* point back to the source document.

> **The dataset is not in this folder.** Only the five PDFs are here. Find out where the raw trips are and whether the 80/10/10 partitions are already split (see §10, Open questions).

---

## 0. What you are building, in one paragraph

You take a noisy 1 Hz GPS track (`.track`) and a road network (`.nodes` + `.arcs`), and output the road segment (arc) the vehicle was on at each GPS point, plus a connected route. The reference route (`.route`) gives the right answer. The pipeline has four stages:

1. **Candidate generation.** For each GPS point, find nearby arcs.
2. **Classifier.** Score each (GPS point, candidate arc) pair: "is this the true arc?" You need at least 5 models, then Bayesian tuning of the best one.
3. **HMM + Viterbi.** Use the classifier's probabilities as emissions and road connectivity as transitions to pick a coherent sequence.
4. **Serving.** A FastAPI server on Laptop 1 runs everything for the 10 evaluator trips and scores the result against server-side ground truth. A UI on Laptop 2 lets the evaluator pick a trip and shows the route and metrics.

---

## 1. Master requirements checklist (everything that is graded or mandatory)

### 1.1 Data & EDA *(Dataset §4, §5, §8; Deliverables §2.2)*
- [ ] Number of trips per partition (80 / 10 / 10)
- [ ] GPS observations per trip; file availability and sizes; dtypes and column meanings
- [ ] Missing or invalid GPS points (NaN, (0,0), out-of-range lat/lon, swapped lon/lat)
- [ ] Timestamp ordering, duplicates, irregular gaps (the doc says about 97% of steps are exactly 1 s, so find the other 3%)
- [ ] Geographic extent (bounding box) of each trajectory
- [ ] Road network: node and arc counts, connectivity, isolated nodes, weakly and strongly connected components
- [ ] Anomalous trips (big gaps, jumps, implausible speeds)
- [ ] Reference route availability and structure (arc-id sequence, consecutive arcs connected?)
- [ ] **Outputs to keep:** inventory table, summary stats, coordinate and trajectory plots, **interactive Folium maps** (GPS + network), network size/connectivity table, a data-quality log of issues, assumptions and decisions
- [ ] Use a proper distance measure: a per-trip projected CRS (e.g. the local UTM zone via `pyproj`) or haversine. **Never use Euclidean distance on degrees.** Justify the choice.
- [ ] State which partition every analysis used
- [ ] Write down the limitations of the reference routes; don't overstate how certain the labels are

### 1.2 Partitions & leakage *(Dataset §7; Policy §7; Models §7, §9)*
- [ ] No trip split across partitions
- [ ] Evaluator trips: no training, no feature selection, no algorithm or hyperparameter choice, no threshold choice, no decoder tuning
- [ ] Fit scalers and other learned preprocessing on training data only. Put them inside an sklearn `Pipeline` so each CV fold refits them.
- [ ] Evaluator ground truth stays out of the prediction pipeline and out of the UI
- [ ] No hardcoded predictions and no trip-specific code paths. The repo will be inspected for this.

### 1.3 Models *(Models §2–§6; Deliverables §3; Policy §3)*
- [ ] **≥ 5 distinct models**, ideally from different families
- [ ] Record each model's config, random seed, training time and inference time
- [ ] Fit and cross-validate on the 80 training trips with **group-aware CV (GroupKFold by trip)**
- [ ] Compare models on the 10 student-test trips
- [ ] Metrics where applicable: accuracy, precision, recall, F1, confusion matrix, log loss, **observation-level Top-1 accuracy**, **candidate miss rate**, **route continuity**, sequence-level performance, training time, inference time
- [ ] Name the primary selection metric and explain why it fits
- [ ] Don't select a model on training accuracy
- [ ] If you use clustering or another non-probabilistic method, explain how you turn its output into candidate scores

### 1.4 Bayesian hyperparameter optimisation *(Models §7; Deliverables §4)*
- [ ] Justified search space, stated objective, group-aware CV on the 80 training trips
- [ ] Save every trial (the search record is mandatory evidence)
- [ ] Refit with the best params and evaluate on student-test
- [ ] Compare **untuned vs tuned** on CV Macro F1, student-test Top-1, student-test Macro F1, log loss and inference time, using the same partitions
- [ ] Report computational cost (number of trials, wall time)

### 1.5 HMM / Viterbi *(Models §8–§9; Deliverables §5)*
- [ ] Define the hidden states. Keep the mapping from each candidate to its road segment.
- [ ] Emission and transition models, with justification
- [ ] Viterbi recurrence, explained
- [ ] Handle missing candidates, invalid transitions, GPS gaps and disconnected segments
- [ ] Return an arc for every relevant GPS point
- [ ] **Classifier-only vs classifier + HMM:** pointwise accuracy, sequence-level metrics, route continuity. Discuss where the HMM did *not* help.
- [ ] Tune decoder parameters on training and student-test data only

### 1.6 Folium maps *(Deliverables §6)*
- [ ] Layers: raw GPS, relevant road network, reference route, predicted route
- [ ] Distinct colours and markers, legend or labels, sensible zoom and extent, `LayerControl`
- [ ] At least one screenshot in the PDF; commit the HTML **and** the code that generates it

### 1.7 API / UI / deployment *(Main §3, §6; Deliverables §7)*
- [ ] Laptop 1: FastAPI with the model, HMM, road network and server-side evaluation
- [ ] Laptop 2: UI that lists the **10 evaluator trips**, sends the selected trip ID to the API, and shows the predicted route and results
- [ ] Works for **all 10** trips, not just one demo trip
- [ ] The API computes accuracy from its internal reference data. Ground truth never reaches the model input or the UI.
- [ ] **Show the API logs on Laptop 1** during the evaluation
- [ ] Report screenshots of the working UI and of successful API responses in Laptop 1's terminal

### 1.8 Submission *(Deliverables §1, §8)*
- [ ] **Public GitHub repo** with data processing, features, training code for every model, evaluation and plots, Bayesian optimisation code with configs and results, HMM, API, UI, deployment config, map code, CSVs and logs, saved weights and configs (within size limits), and a README covering install, run and deploy, all in clearly named folders, with no secrets
- [ ] **One PDF report** that follows the required structure (§7 below). Every number in it must be reproducible from the repo.
- [ ] Disclose AI assistance as the event policy requires. Every member must be able to explain every claim, table and graph in a viva (Policy §6 lists the topics).

---

## 2. Technical design

### 2.1 Loading and geometry
- Loaders: `.track` → (lon, lat, t); `.nodes` → (lon, lat) indexed by row; `.arcs` → (from, to) with row = arc_id; `.route` → list of arc_ids. Check column order on real plots before trusting it.
- **Find out whether each trip has its own network** (the docs say "each trip contains files describing … relevant road-network information"). If so, arc IDs are local to a trip. That decides the ML framing below.
- Projection: pick a local metric CRS per trip, e.g. the UTM zone at the trip's centroid. Trips may be in different countries. Convert everything to metres once and cache it.
- Build a **directed** graph per trip (`networkx.DiGraph`, or `scipy.sparse` for speed) with edge weight = arc length in metres. Two-way roads are probably two opposite arcs, so the GPS heading matters.

### 2.2 Deriving per-observation labels (critical; document it as an assumption)
`.route` is an ordered list of arcs, **not** a label per GPS point. Create the labels like this:
- Align each GPS point to one route arc with a **monotonic dynamic-programming alignment**: the route index can never move backwards, and the cost is the point-to-arc distance (DTW-style).
- Plot the result on maps for a few trips to check it. Log the points whose aligned arc is far away (label noise).
- Use this both for training labels and for Top-1 evaluation. Say so in the report.

### 2.3 Candidate generation
- For each GPS point, take arcs within radius **R** metres, capped at the **K** nearest. Use a spatial index such as `shapely.STRtree`.
- Choose R and K on training trips by plotting **candidate miss rate** (true arc not among the candidates) against R and K, traded off against candidate-set size.
- If a point has 0 candidates, widen R once. If it still has none, flag it for the HMM's missing-candidate handling.

### 2.4 Features (per point–candidate pair; must use only GPS + network, never labels)
| Group | Examples |
|---|---|
| Distance | perpendicular distance to the arc; rank among candidates; distance − min distance; projection fraction along the arc, plus a flag when the projection falls past an endpoint |
| Direction | angle between the GPS heading (smoothed over ±k points) and the arc bearing (directed); cos and sin of it |
| Context | mean distance from the neighbouring points (t−k…t+k) to the same arc (strong signal against noise) |
| Motion | speed, acceleration, turn angle, Δt to previous and next point |
| Network | arc length, in/out degree of the end nodes, whether the arc is reachable from the previous point's nearest arc within N hops |
| Density | number of candidates within R (parallel-road ambiguity) |

Keep a `feature_list.json`. Explain *why* you chose each group in the report. This is a required viva topic.

### 2.5 ML framing
- **Binary candidate classification.** Each row is a (point, candidate) pair and the label is 1 if the candidate is the aligned true arc.
- Top-1 = argmax probability over a point's candidates, compared with the label.
- Precision, recall, macro F1, confusion matrix and log loss are computed at candidate level (positive and negative classes).
- Class imbalance is roughly 1 positive per K candidates. Use `class_weight` or sample weights and report macro F1, not plain accuracy.
- CV: `GroupKFold(n_splits=5)` grouped by trip_id on the 80 training trips. Fixed seed.

### 2.6 Model line-up (≥ 5, from different families)
| # | Model | Family | Notes |
|---|---|---|---|
| 0 | Nearest-segment rule | baseline (not ML) | Reference point only; doesn't count toward the 5 |
| 1 | Logistic Regression | linear / probabilistic | Scaled; well-calibrated probabilities |
| 2 | Gaussian Naive Bayes | probabilistic | Shows what the independence assumption costs |
| 3 | KNN (distance-weighted) | instance-based | Needs scaling; may need a subsample of rows (document it) |
| 4 | Random Forest | ensemble | Gives feature importances |
| 5 | LightGBM / XGBoost / HistGB | boosting | Strong candidate for the final model; has per-iteration eval curves |
| 6 | MLP | neural net | The only model with true **epoch-wise train/val loss**. Plot those curves; don't fake curves for the others. |

Choose the final model **from the results**. Weigh CV mean ± std across folds, student-test Top-1, log loss (emission quality matters for the HMM), inference time and complexity.

**Primary metric (suggestion):** observation-level Top-1 accuracy. It doesn't depend on a threshold and it is exactly what the route needs. Report macro F1 and log loss alongside it everywhere, because the report tables require them.

### 2.7 Bayesian optimisation
- Optuna (TPE, a sequential model-based Bayesian method) or `GPSampler` / scikit-optimize `BayesSearchCV` (Gaussian process). If someone may question whether TPE counts as Bayesian, use the GP option.
- Objective: mean grouped-CV score (the primary metric) on the 80 training trips. Use 40–100 trials with a fixed seed.
- Persist the record: an Optuna SQLite study plus `trials.csv`, an optimisation-history plot and a parameter-importance plot.
- Example space for LightGBM: `num_leaves` 15–255 (log), `learning_rate` 0.01–0.3 (log), `n_estimators` 100–1500, `min_child_samples` 5–200, `subsample` 0.5–1, `colsample_bytree` 0.5–1, `reg_lambda` 1e-3–10 (log). Write down why each range was chosen.

### 2.8 HMM + Viterbi
- **Hidden states:** the candidate arcs of point *t*. Keep an arc_id with each state.
- **Emission:** `log e_t(j) = log max(p_clf(j), ε)`. Optionally blend in a Gaussian distance term `N(d; σ)` with weight α, tuned.
- **Transition:** Newson–Krumm style, `exp(-|d_network − d_straight| / β)`. Here `d_network` is the shortest directed path from the projection on arc i to the projection on arc j, and `d_straight` is the distance between the two GPS points. Staying on the same arc means distance along the arc. Moving backwards along a directed arc gets a penalty. Run Dijkstra with a cutoff (e.g. 2·d + 200 m) and cache the results.
- **Recurrence (log space):** `δ_t(j) = log e_t(j) + max_i [δ_{t−1}(i) + log a_ij]`, with backpointers and a final backtrack.
- **Missing candidates:** skip the point and bridge t−1 → t+1, but still output an arc for it by inheriting from a neighbour. Flag it.
- **GPS gaps** (large Δt or distance): scale β with Δt, or split the HMM into separate segments when the gap passes a threshold.
- **Disconnected segments** (every transition −∞): break the HMM there, restart from emissions alone and count the breaks.
- **Route output:** the per-point arcs, de-duplicated, with shortest-path fill between consecutive arcs, giving an arc sequence to compare with `.route`.
- Tune σ, β, α, R and the break threshold on training and student-test only.

### 2.9 Metrics to implement (`metrics.py`)
- Top-1 (classifier argmax vs HMM), candidate miss rate
- Candidate-level precision, recall, F1 (macro), log loss, confusion matrix
- **Route continuity:** the share of consecutive changes in predicted arc where `prev.to_node == next.from_node`, or the next arc is reachable within 1 hop. Also report breaks per km.
- **Sequence-level:** length-weighted route precision, recall and F1 against the `.route` arcs; edit distance between arc sequences; fraction of trips matched exactly
- Training time (fit seconds) and inference time (ms per trip and per point)

---

## 3. Repository layout

```
pathsnap/
├── README.md                 # install, data placement, run every stage, deploy on 2 laptops
├── requirements.txt          # pinned; state the Python version
├── .gitignore                # raw data, venv, .env
├── configs/                  # data.yaml, candidates.yaml, models/*.yaml, hmm.yaml, api.yaml
├── data/                     # gitignored; README explains layout: train/ student_test/ evaluator/
├── src/pathsnap/
│   ├── io.py  geo.py  network.py  labels.py  candidates.py  features.py
│   ├── models.py  metrics.py  hmm.py  pipeline.py   # pipeline.predict_trip(track, network) → arcs (NO labels arg)
│   └── viz.py                # Folium + matplotlib helpers
├── notebooks/                # 01_eda, 02_labels_candidates, 03_models, 04_bayes_opt, 05_hmm, 06_maps
├── scripts/                  # build_dataset, train_all, cv_compare, optimize, tune_hmm, eval_hmm, export_artifacts, make_maps, smoke_test_api
├── experiments/              # results/*.csv, logs/, optuna.db, figures/
├── artifacts/                # model.joblib, preprocessing.joblib, hmm.json, feature_list.json, metadata.json (versions, seeds, metrics)
├── maps/                     # *.html Folium outputs
├── api/                      # main.py, schemas.py, evaluation.py (only file that reads evaluator .route)
├── ui/                       # app.py
├── report/                   # figures + final PDF
└── tests/                    # loaders, Viterbi on a toy graph, leakage guard
```

**Leakage guard (cheap and convincing in a viva):** the loader refuses to read `data/evaluator/` unless `PATHSNAP_MODE=serve`. Only `api/evaluation.py` reads the evaluator `.route` files, and only *after* `predict_trip` has returned.

Commit early and often with meaningful messages. Repository history is one of the things compared across teams.

---

## 4. API (Laptop 1) and UI (Laptop 2)

**API (FastAPI + uvicorn)**
- `GET /health`: model loaded, version
- `GET /trips`: the 10 evaluator trip IDs and harmless metadata (point count, duration). No labels.
- `POST /predict {trip_id}`: load track and network → candidates → features → model → Viterbi → *then* `evaluation.py` scores against the server-side reference. Returns the GPS points, the predicted arcs per point, the predicted route geometry (GeoJSON), metrics (Top-1 for classifier-only and HMM, route P/R/F1, continuity, candidate miss rate) and timings.
- Load the artifacts once at startup. Enable CORS for the UI. Run with `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
- Log each request clearly to the terminal (trip, client IP, number of points, stage timings, metrics). The evaluator will look at this.

**UI (recommendation: Streamlit + `streamlit-folium`)**
- Fetches `/trips` and shows the 10 options. Has a Run button. Renders a Folium map with GPS points, the nearby network and the predicted route, plus a legend and `LayerControl`. Shows metric cards and the request latency.
- Keep **no data and no model on Laptop 2**. Everything goes through the API, which proves nothing bypasses it.
- The API base URL lives in a config or env file. Don't hardcode the IP.
- **Don't send the reference route to the UI.** The rules say ground truth must never be exposed to the UI. Reference-vs-predicted maps belong in the report and should be built offline from student-test trips.

**Two-laptop readiness**
- Campus Wi-Fi often isolates clients from each other. Bring a phone hotspot as backup. Test the real two-laptop setup **before** the event, including firewall rules.
- `scripts/smoke_test_api.py` calls `/predict` for all 10 trips and asserts success and response time. Run it right before the evaluation.

---

## 5. Report (one PDF; required structure)

| Report section | Content | Produced by |
|---|---|---|
| 2.2 Data & EDA | how you preprocessed, patterns, features and why, missing-value analysis, trajectory plots, network visualisations, ≥1 Folium map, partition named for every analysis | `01_eda`, `02_labels_candidates`, data-quality log |
| 3.1 Comparison table | Model, family, key config, CV Macro F1, Top-1, route continuity, student-test Macro F1, log loss, inference time (write N/A + reason where a metric doesn't apply) | `cv_compare` → `results/model_comparison.csv` |
| 3.2 Per model | A: why suitable, strengths and limits · B: key hyperparameters and initial values, loss or objective · C: student-test performance, Top-1/F1/log loss, evidence of over- or underfitting (train vs CV gap) | per-model results |
| 3.3 Plots | CV macro F1 bar chart with fold error bars; student-test Top-1 + F1; log loss; confusion matrices; train/val loss **only** for models with epoch history | `figures/` |
| 4 Bayesian opt | parameters, ranges, objective, number of trials, best values, untuned vs tuned table (CV F1, Top-1, test F1, log loss, inference time) | `optimize` + Optuna record |
| 5 HMM | states, emission, transition, gaps and missing candidates, disconnected segments, did it help? Classifier-only vs HMM table, plus failure cases | `eval_hmm` |
| 6 Folium | screenshot(s) with GPS, network, reference and predicted layers | `make_maps` → `maps/` |
| 7 API/UI | UI screenshots; API terminal logs on Laptop 1 | the real demo |
| 8 Integrity | AI-use disclosure | — |

Write the report in your own words, from your own results. The policy says reports may be checked for AI-generated text and every member may be questioned on any claim.

---

## 6. Execution order and time-boxing

This is a hackathon with a **first-phase evaluation before lunch**, so build a **thin working version of the whole pipeline first**, then improve it.

| Phase | Goal | Done when |
|---|---|---|
| **P0 Setup** | Repo, environment, folder skeleton, loaders, partitions confirmed | Every trip in every partition loads; inventory table exists |
| **P1 EDA** | Every item in §1.1, plus Folium maps | Plots, maps and data-quality log committed (probably what phase 1 checks; confirm) |
| **P2 Labels + candidates** | Alignment labels, R/K chosen by miss rate | Labels checked visually on maps; miss-rate curve saved |
| **P3 Baseline end-to-end** | Nearest-segment + simple distance-based HMM, API with `/trips` + `/predict`, bare UI | Evaluator can click a trip and see a route across 2 laptops. **De-risks the demo early.** |
| **P4 Features + 5+ models** | Feature table, grouped CV, student-test evaluation, comparison CSV and plots | `model_comparison.csv` complete |
| **P5 Bayesian opt** | Tune the selected model, untuned vs tuned | Optuna record + comparison saved |
| **P6 HMM with classifier emissions** | Decoder tuning, classifier-only vs HMM, failure analysis | HMM results table + example maps |
| **P7 Export + integrate** | Artifacts, API on the final model, polished UI, smoke test across all 10 trips | Smoke test passes on the real two-laptop network |
| **P8 Report + README** | PDF report, README, screenshots, public repo | Every number traceable to a CSV in the repo |

**Suggested split for 4 people** (adjust to your team): A, data + EDA + labels + maps; B, features + models + Bayesian opt; C, HMM + metrics; D, API + UI + deployment + README. Everyone writes the report sections for their own part, and everyone reads all of it before the viva.

---

## 7. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Labels misaligned (route ≠ per-point labels) | Monotonic DP alignment + visual checks; report it as a limitation |
| Shortest paths too slow | Per-trip subgraph, Dijkstra cutoff, cache per (arc, arc) |
| Millions of candidate rows | Subsample *training* rows for KNN/SVM only (document it); never subsample the test set |
| Direction confusion on two-way roads | Heading-vs-bearing features; directed graph |
| Laptops can't reach each other | Hotspot backup; `--host 0.0.0.0`; firewall; test the day before |
| Leakage accusation | Loader guard, `Pipeline`-wrapped scalers, `predict_trip` has no labels argument, evaluation module kept separate |
| Fabrication concerns | Every figure comes from a script that writes to `experiments/`; seeds logged; no hand-drawn curves |

---

## 8. Viva prep (Policy §6)
Every member should be able to explain: the dataset structure, preprocessing, feature choices, model comparison and selection, Bayesian optimisation, the HMM (states, emissions, transitions), the Viterbi recurrence, the API, the UI and the results. Hold one internal mock viva before the evaluation.

---

## 9. Definition of done (final check)
- [ ] ≥ 5 models trained, logged and compared on CV + student-test, with the full table
- [ ] Bayesian search record saved; tuned vs untuned compared
- [ ] HMM: classifier-only vs HMM comparison, continuity analysis and failure cases
- [ ] Folium maps (HTML + code + screenshot in the PDF)
- [ ] API + UI work across two laptops for all 10 evaluator trips; logs visible on Laptop 1
- [ ] No evaluator data used in development; no ground truth in the UI; no hardcoding
- [ ] Public repo with README, requirements, artifacts, results and no secrets
- [ ] One PDF in the required structure; every number reproducible; AI use disclosed

---

## 10. Open questions for the organisers (ask early)
1. Where is the raw dataset, and are the 80/10/10 partitions pre-assigned (which trip IDs)?
2. How do the 10 evaluator trips (tracks + reference routes) reach Laptop 1? Bundled now, or handed over on the day?
3. What exactly is assessed in the **first-phase evaluation before lunch**?
4. Is Macro F1 meant at candidate level (binary) as we planned, or something else?
5. May the UI show the reference route on the map, or only metrics? (We assume metrics only.)
6. Does Optuna's TPE count as "Bayesian", or do they expect a Gaussian-process method?
7. Repository size limit for model weights; required format for disclosing AI use.
