# Team split and Git workflow

Four people, four areas. Each area owns its own files, so branches rarely touch the same lines. Swap people between areas if you like, but keep the file ownership.

| Person | Area | Branch |
|---|---|---|
| **Adyanth** | Data, EDA, labels, candidates, maps, integration | `feat/adyanth-data` |
| **Advitha** | Features, 5+ models, comparison, Bayesian optimisation | `feat/advitha-models` |
| **Sudhe** | Metrics, HMM/Viterbi, decoder tuning, classifier-vs-HMM study | `feat/sudhe-hmm` |
| **Tilak** | API, UI, evaluation, deployment, two-laptop testing | `feat/tilak-api-ui` |

Everyone also writes the report sections for their own area, and everyone must be able to explain every part in the viva (the policy checks this).

---

## 1. Who does what

### Adyanth: data foundation and integration (critical path, everyone waits on this)
Files: `src/pathsnap/{io,geo,network,labels,candidates,viz,pipeline}.py`, `scripts/inventory.py`, `scripts/make_maps.py`, `notebooks/01_eda.ipynb`, `notebooks/02_labels_candidates.ipynb`, `maps/`
1. Get the dataset into `data/raw/` and run `scripts/inventory.py`. Confirm file formats against the real files (column order, whether each trip has its own network).
2. Full EDA: missing values, time gaps, extents, network connectivity, anomalous trips, Folium maps, data-quality log.
3. `labels.py`: align each GPS point to one route arc (monotonic DP). Check visually on maps.
4. `candidates.py`: radius R and max K chosen by candidate miss rate on training trips.
5. **Publish the candidate table early** (see section 2) so the others can start on real data.
6. Later: wire the final model and HMM into `pipeline.predict_trip`, generate the report maps.

### Advitha: features, models, tuning
Files: `src/pathsnap/{features,models}.py`, `scripts/{build_dataset,train_all,cv_compare,optimize}.py`, `notebooks/03_models.ipynb`, `notebooks/04_bayes_opt.ipynb`, `experiments/results/model_*`, `configs/models/`
1. `features.py`: distance, direction, context, motion, network and density features. Save `artifacts/feature_list.json`.
2. At least 5 models from different families (suggested: Logistic Regression, Gaussian NB, KNN, Random Forest, LightGBM, plus an MLP for real epoch loss curves). Group-aware 5-fold CV by trip on the 80 training trips.
3. Comparison table (CV macro F1, Top-1, route continuity, student-test macro F1, log loss, inference time), bar charts, confusion matrices.
4. Pick the final model from the results and justify it.
5. Bayesian optimisation (Optuna or a GP method) with the full trial record, untuned vs tuned table.
6. Export the model to `artifacts/`.

### Sudhe: metrics and HMM
Files: `src/pathsnap/{metrics,hmm}.py`, `scripts/{tune_hmm,eval_hmm}.py`, `notebooks/05_hmm.ipynb`, `experiments/results/hmm_*`, `tests/test_hmm.py`
1. `metrics.py`: Top-1, candidate miss rate, candidate-level precision/recall/macro F1/log loss, route continuity, sequence-level route precision/recall/F1, timing. Advitha and Tilak import these, so push a first version early.
2. `hmm.py`: states, emissions from classifier probabilities, transitions from network vs straight-line distance, Viterbi in log space, gap handling, missing candidates, disconnected segments.
3. Start with a **distance-only emission** so you are not blocked on Advitha's model, then swap in classifier probabilities.
4. Tune decoder parameters (`configs/hmm.yaml`) on training and student-test only.
5. Classifier-only vs classifier+HMM comparison, including cases where the HMM did not help.
6. Unit test Viterbi on a hand-computed toy graph.

### Tilak: API, UI, deployment
Files: `api/*`, `ui/*`, `scripts/{run_api.sh,run_ui.sh,smoke_test_api.py}`, `README.md` (deployment section), `report/` screenshots
1. Make the skeleton in `api/` and `ui/` work end to end with a **stub** pipeline (return the nearest-arc guess) so the two-laptop demo path is proven early.
2. `api/evaluation.py`: server-side scoring against the evaluator reference route. This is the only code allowed to read evaluator `.route` files, and only after prediction. Ground truth must never go to the UI.
3. Load artifacts once at startup, clear per-request logging (the evaluator will look at the API terminal), `/health`, `/trips`, `/predict`.
4. UI: trip list (10 trips), run button, Folium map with GPS + predicted route + legend + layer control, metric cards.
5. `scripts/smoke_test_api.py`: call `/predict` for all 10 trips and assert success.
6. Test across two real laptops and a phone hotspot before the event. Collect UI and API-terminal screenshots for the report.

---

## 2. Interfaces (agree on these now, so nobody is blocked)

**Candidate table** (Adyanth produces, one row per observation-candidate pair), saved as parquet/CSV per trip:

| column | meaning |
|---|---|
| `trip_id`, `obs_idx`, `arc_id` | identifiers |
| `dist_m` | distance from the GPS point to the arc, in metres |
| `proj_frac` | unclipped projection fraction along the arc |
| `label` | 1 if `arc_id` is the aligned reference arc for this observation, else 0 (train and student_test only) |

**Feature table** (Advitha produces): the candidate table plus numeric feature columns listed in `artifacts/feature_list.json`.

**Classifier contract** (Advitha exports): a fitted sklearn-compatible object with `predict_proba(X)[:, 1]` = probability that the candidate is the true arc. Scalers live inside a `Pipeline`.

**Decoder contract** (Sudhe): `viterbi(...)` takes per-observation candidate arc ids and their probabilities, and returns one arc id per observation, plus a `break` flag where the chain was reset.

**Prediction contract** (Adyanth wires, Tilak serves): `pipeline.predict_trip(trip, artifacts)` returns `{"gps": [[lon, lat], ...], "pred_arcs": [...], "route_lonlat": [[lon, lat], ...], "timings_ms": {...}}`. It takes no reference route. `api/evaluation.py` adds `"metrics"` afterwards.

If you need to change an interface, tell the others in the group chat and update this file in the same PR.

---

## 3. Suggested order

| Step | Who | Goal |
|---|---|---|
| 1 | Everyone | Clone, set up the environment, run `pytest`, create your branch |
| 2 | Adyanth | Inventory + EDA + candidate table (push as soon as it exists) |
| 2 | Tilak | Stub API + UI working across two laptops |
| 2 | Advitha, Sudhe | Develop on `tests/fixtures/toy/` and synthetic data until the candidate table lands |
| 3 | Advitha | Features + 5 models + CV + comparison |
| 3 | Sudhe | Metrics + Viterbi with distance-only emissions |
| 4 | Advitha | Bayesian optimisation, export model |
| 4 | Sudhe | Plug in classifier probabilities, tune decoder |
| 5 | Adyanth + Tilak | Wire the real pipeline into the API, full smoke test |
| 6 | Everyone | Report sections, screenshots, mock viva |

Detailed requirements and design notes: [`docs/PLAN.md`](docs/PLAN.md).

---

## 4. Git workflow

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
3. **Pull often:** `git fetch origin && git merge origin/main` (or `git rebase origin/main`) at least every time you start work, so conflicts stay small.
4. **Small PRs, clear messages**, e.g. `hmm: add log-space Viterbi with gap breaks`. The repository history is compared across teams, so commit real work as you go.
5. **Never commit** raw data, passwords, API keys, `.env` files, or big binaries. `data/raw/` is gitignored. Model weights go in `artifacts/` only once final and small enough.
6. **Never use the evaluator partition during development.** The loader enforces this; do not work around it.
7. **Experiment outputs** (CSVs, logs, figures) are committed under `experiments/`. Every number in the report must come from a script in the repo, with the seed (`148`) recorded.
8. **Keep the notebooks clean:** restart and run all before committing, so the outputs are real.
