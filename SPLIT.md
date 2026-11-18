# Team split and Git workflow

Four people. The work has two halves:

1. **Shared foundation (before the first review):** preprocessing, labels, candidates and feature engineering. Built once, used by everyone.
2. **Model round:** everyone trains 2-3 models of their own on that shared foundation. Then tuning, HMM, API/UI and the report.

Each area owns its own files, so branches rarely touch the same lines. Swap people between areas if you like, but keep the file ownership.

| Person | Foundation job | Models to try (2-3) | Later job | Branch |
|---|---|---|---|---|
| **Adyanth** | Data, EDA, labels, candidates, maps | Logistic Regression, Gaussian NB (+ SGD/Ridge optional) | Wire the real pipeline, report maps | `feat/adyanth-data` |
| **Advitha** | Features, shared evaluation harness | LightGBM, HistGradientBoosting, XGBoost | Comparison table, Bayesian optimisation, export model | `feat/advitha-models` |
| **Sudhe** | Metrics (`metrics.py`) | Random Forest, Extra Trees, Decision Tree | HMM/Viterbi, decoder tuning, classifier-vs-HMM study | `feat/sudhe-hmm` |
| **Tilak** | API/UI skeleton on the toy fixture | KNN (distance-weighted), MLP (+ linear SVM on a subsample, optional) | Evaluation, deployment, two-laptop testing | `feat/tilak-api-ui` |

That is 9-11 models from different families, well above the required 5. One family each also makes the viva easy: you explain your own models in depth and the others at a high level. Everyone must be able to explain every part (the policy checks this), and everyone writes the report sections for their own work.

---

## 1. Milestones

| # | Milestone | Done when |
|---|---|---|
| **Review 1** | **Preprocessing and feature engineering** | Dataset inventory + EDA + Folium maps, data-quality log, labels, candidate table (miss-rate analysis), feature table and `artifacts/feature_list.json` are committed |
| 2 | Model round | Every person's models are evaluated through the shared harness; comparison table complete |
| 3 | Tuning + HMM | Bayesian optimisation record, tuned vs untuned table, HMM vs classifier-only table |
| 4 | Integration | Real pipeline behind the API, UI works for all 10 evaluator trips across two laptops |
| 5 | Report + viva | PDF report, README, screenshots, mock viva |

---

## 2. Who does what

### Adyanth: data foundation (critical path for Review 1)
Files: `src/pathsnap/{io,geo,network,labels,candidates,viz,pipeline}.py`, `scripts/inventory.py`, `scripts/make_maps.py`, `notebooks/01_eda.ipynb`, `notebooks/02_labels_candidates.ipynb`, `maps/`, `scripts/models/adyanth_*.py`
1. The dataset is already in the repo (`data/raw/train`, `data/raw/student_test`). Run `scripts/inventory.py` (output: `experiments/results/inventory.csv`).
2. Full EDA: missing values, time gaps, extents, network connectivity, anomalous trips, Folium maps, data-quality log.
3. `labels.py`: align each GPS point to one route arc (monotonic DP). Check visually on maps.
4. `candidates.py`: radius R and max K chosen by candidate miss rate on training trips.
5. **Push the candidate table as early as possible** (section 3), since everyone else needs it.
6. Models: Logistic Regression, Gaussian NB (and optionally SGD/Ridge).
7. Later: wire the final model and HMM into `pipeline.predict_trip`, generate the report maps.

### Advitha: features and the shared harness
Files: `src/pathsnap/{features,harness}.py`, `scripts/{build_dataset,cv_compare,optimize}.py`, `notebooks/03_models.ipynb`, `notebooks/04_bayes_opt.ipynb`, `configs/models/`, `scripts/models/advitha_*.py`
1. `features.py`: distance, direction, context, motion, network and density features. Save `artifacts/feature_list.json`. Fit any scaler on training data only (inside a `Pipeline`).
2. `harness.py`: the one function everyone calls to evaluate a model (section 4). It fixes the CV folds and the metrics, so the comparison is fair.
3. Models: LightGBM, HistGradientBoosting, XGBoost.
4. `cv_compare.py`: collect every model's result file into the comparison table, bar charts and confusion matrices.
5. Pick the final model from the results and justify it.
6. Bayesian optimisation (Optuna or a GP method) with the full trial record and an untuned vs tuned table.
7. Export the model to `artifacts/`.

### Sudhe: metrics, then HMM
Files: `src/pathsnap/{metrics,hmm}.py`, `scripts/{tune_hmm,eval_hmm}.py`, `notebooks/05_hmm.ipynb`, `experiments/results/hmm_*`, `tests/test_hmm.py`, `scripts/models/sudhe_*.py`
1. `metrics.py` first (the harness needs it): Top-1, candidate miss rate, candidate-level precision/recall/macro F1/log loss, route continuity, sequence-level route precision/recall/F1, timing. Push a first version early.
2. Models: Random Forest, Extra Trees, Decision Tree.
3. `hmm.py`: states, emissions from classifier probabilities, transitions from network vs straight-line distance, Viterbi in log space, gap handling, missing candidates, disconnected segments.
4. Start with a **distance-only emission** so you are not blocked on the final model, then swap in classifier probabilities.
5. Tune decoder parameters (`configs/hmm.yaml`) on training and student-test only.
6. Classifier-only vs classifier+HMM comparison, including cases where the HMM did not help.
7. Unit test Viterbi on a hand-computed toy graph.

### Tilak: API/UI, then deployment
Files: `api/*`, `ui/*`, `scripts/{run_api.sh,run_ui.sh,smoke_test_api.py}`, `README.md` (deployment section), `report/` screenshots, `scripts/models/tilak_*.py`
1. Make the skeleton in `api/` and `ui/` work end to end on `tests/fixtures/toy/` with a **stub** pipeline (nearest-arc guess), so the two-laptop demo path is proven early.
2. Models: KNN (distance-weighted, on a documented training subsample if needed) and MLP. The MLP is the model with real epoch-wise train/val loss curves for the report; the others get no fake curves.
3. `api/evaluation.py`: server-side scoring against the evaluator reference route. This is the only code allowed to read evaluator `.route` files, and only after prediction. Ground truth must never go to the UI.
4. Load artifacts once at startup, clear per-request logging (the evaluator will look at the API terminal), `/health`, `/trips`, `/predict`.
5. UI: trip list (10 trips), run button, Folium map with GPS + predicted route + legend + layer control, metric cards.
6. `scripts/smoke_test_api.py`: call `/predict` for all 10 trips and assert success.
7. Test across two real laptops and a phone hotspot before the event. Collect UI and API-terminal screenshots for the report.

---

## 3. Interfaces (agreed now, so nobody is blocked)

**Candidate table** (Adyanth produces, one row per observation-candidate pair), saved as parquet/CSV per trip:

| column | meaning |
|---|---|
| `trip_id`, `obs_idx`, `arc_id` | identifiers |
| `dist_m` | distance from the GPS point to the arc, in metres |
| `proj_frac` | unclipped projection fraction along the arc |
| `label` | 1 if `arc_id` is the aligned reference arc for this observation, else 0 (train and student_test only) |

**Feature table** (Advitha produces): the candidate table plus numeric feature columns listed in `artifacts/feature_list.json`. Everyone trains on this same table.

**Classifier contract:** a fitted sklearn-compatible object with `predict_proba(X)[:, 1]` = probability that the candidate is the true arc. Scalers live inside a `Pipeline`.

**Decoder contract** (Sudhe): `viterbi(...)` takes per-observation candidate arc ids and their probabilities, and returns one arc id per observation, plus a `break` flag where the chain was reset.

**Prediction contract** (Adyanth wires, Tilak serves): `pipeline.predict_trip(trip, artifacts)` returns `{"gps": [[lon, lat], ...], "pred_arcs": [...], "route_lonlat": [[lon, lat], ...], "timings_ms": {...}}`. It takes no reference route. `api/evaluation.py` adds `"metrics"` afterwards.

If you need to change an interface, tell the others in the group chat and update this file in the same PR.

---

## 4. Shared evaluation harness (how everyone's models stay comparable)

All models are evaluated the same way:

- **Data:** the shared feature table. **CV:** `GroupKFold(n_splits=5)` by `trip_id` on the 80 training trips, same fold assignment for everyone (seed `148`). **Held-out:** the 10 student-test trips. The evaluator partition is never touched.
- **Call:** `harness.evaluate_model(name, family, estimator, config)` fits, cross-validates, scores the student-test trips and writes one file per model: `experiments/results/models/<name>.json`. One file per model means no merge conflicts on a shared CSV.
- **The file contains:** model name, family, configuration, seed, CV macro F1 per fold (mean and std), student-test Top-1 accuracy, macro F1, precision, recall, log loss, route continuity, candidate miss rate, fit seconds, inference ms per trip and per point. Where a metric does not apply, write `null` and the reason.
- **Your model script** lives in `scripts/models/<yourname>_<model>.py`: build the estimator, call the harness, done. Do not re-implement CV or metrics in your own script.
- **Until the harness lands** (Advitha pushes it right after the feature table), you can prototype in a notebook with the same `GroupKFold` and seed, then move to the harness.
- Note every model you try in the report, including the ones that did badly. All must be documented.

---

## 5. Suggested order

| Step | Who | Goal |
|---|---|---|
| 1 | Everyone | Clone (the dataset comes with it), set up the environment, run `pytest`, create your branch |
| 2 | Adyanth | Inventory + EDA + labels + candidate table (push early) |
| 2 | Advitha | Draft `features.py` on the toy fixture, then on real data once candidates land |
| 2 | Sudhe | `metrics.py` + first Viterbi with distance-only emissions on the toy fixture |
| 2 | Tilak | Stub API + UI working across two laptops on the toy fixture |
| **Review 1** | Everyone | Preprocessing and feature engineering presentable and committed |
| 3 | Everyone | Train your 2-3 models through the harness; Advitha builds the comparison table |
| 4 | Advitha | Bayesian optimisation on the winning model, export it |
| 4 | Sudhe | Plug in classifier probabilities, tune the decoder, classifier-only vs HMM |
| 5 | Adyanth + Tilak | Wire the real pipeline into the API, evaluation, full smoke test on all 10 trips |
| 6 | Everyone | Report sections, screenshots, mock viva |

Detailed requirements and design notes: [`docs/PLAN.md`](docs/PLAN.md).

---

## 6. Git workflow

```bash
git clone https://github.com/Adyanth-212/ML-Hackathon-148-2026.git
cd ML-Hackathon-148-2026
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . && pytest

git checkout -b feat/<yourname>-<area>      # use the branch name from the table above
# ... work, commit small and often ...
git push -u origin feat/<yourname>-<area>
```

Rules:
1. **Never push straight to `main`.** Open a pull request from your branch and get one teammate to review it.
2. **Stay in your own files** (listed above). If you need a change in someone else's file, ask them or open a small PR for it.
3. **Do not change the shared features, labels, candidates or CV folds** without telling everyone. Changing them invalidates everybody's results.
4. **Pull often:** `git fetch origin && git merge origin/main` (or `git rebase origin/main`) at least every time you start work, so conflicts stay small.
5. **Small PRs, clear messages**, e.g. `hmm: add log-space Viterbi with gap breaks`. The repository history is compared across teams, so commit real work as you go.
6. **Never commit** raw data, passwords, API keys, `.env` files, or big binaries. `data/raw/` is gitignored. Model weights go in `artifacts/` only once final and small enough.
7. **Never use the evaluator partition during development.** The loader enforces this; do not work around it.
8. **Experiment outputs** (CSVs, logs, figures) are committed under `experiments/`. Every number in the report must come from a script in the repo, with the seed (`148`) recorded.
9. **Keep the notebooks clean:** restart and run all before committing, so the outputs are real.
