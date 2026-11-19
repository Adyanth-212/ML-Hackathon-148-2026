# PathSnap: map-matching and route reconstruction

UE24CS352A Machine Learning hackathon, team 148 (Adyanth, Advitha, Sudhe, Tilak).

PathSnap matches a noisy GPS track to the road network. For every GPS point it generates candidate road arcs,
scores each (point, candidate) pair with a trained classifier, and decodes one connected route with an HMM/Viterbi
decoder. A FastAPI server runs the pipeline (Laptop 1); a Streamlit UI (Laptop 2) lets the evaluator pick a trip and
see the predicted route with server-side scores.

**Final result: 91.3% of GPS points on the correct road** on the 10 held-out student-test trips (Bayesian-tuned
LightGBM + HMM/Viterbi); the same 91.3% on the 10 evaluator trips, scored by the API server.
Report: [`report/team_187.pdf`](report/team_187.pdf). Every model and number: [`RESULTS.md`](RESULTS.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/pathsnap/` | Library: `io`, `geo`, `network`, `labels`, `candidates`, `features`, `models`, `metrics`, `harness`, `hmm`, `pipeline`, `viz` |
| `scripts/` | Pipeline stages, model scripts (`scripts/models/`), Bayesian optimisation, launchers, smoke test |
| `configs/` | YAML: data partitions, candidates, Bayesian optimisation, HMM, API |
| `api/` | FastAPI backend (endpoints, model loading, server-side evaluator scoring) |
| `ui/` | Streamlit evaluator interface |
| `artifacts/` | Exported model (`model.joblib`), feature list, metadata used by the API |
| `experiments/` | `results/` (CSVs, per-model JSON, Optuna study), `figures/`, `logs/` |
| `maps/` | Interactive Folium HTML maps |
| `notebooks/` | Executed notebooks: before/after preprocessing, EDA, labels and candidates |
| `report/` | Report source, generator (`build_report.py`), figures, `team_187.pdf` (final report) |
| `docs/` | Data-quality log, data/feature reference, AI-use disclosure |
| `tests/` | Unit and API tests (toy fixtures) |
| `data/` | `raw/train`, `raw/student_test` (committed), `processed/` (labels and candidates); `raw/evaluator` is never committed |

## Setup

Python 3.12 (3.10-3.13 work). On macOS, LightGBM and XGBoost need `brew install libomp`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Data and partitions

```
data/raw/train/          80 trips: fitting, cross-validation, tuning
data/raw/student_test/   10 trips: held-out comparison
data/raw/evaluator/      10 trips: API server only (not in the repo; put them here on Laptop 1)
```

`pathsnap.io` refuses to read `data/raw/evaluator/` unless `PATHSNAP_MODE=serve`, which only `scripts/run_api.sh`
sets. Reference routes are read only with `with_route=True`; `pathsnap.pipeline.predict_trip` takes no reference route,
and evaluator scoring lives only in `api/evaluation.py`, after prediction. Seed 148 everywhere.

## Reproducing the results

```bash
python scripts/inventory.py              # dataset inventory
python scripts/eda.py                    # EDA tables and figures
python scripts/make_maps.py              # Folium EDA maps (add --labels / --predicted for the others)
python scripts/build_labels.py           # per-point labels from the reference routes
python scripts/candidate_sweep.py        # choose R and K on train
python scripts/build_candidates.py       # candidate tables
python scripts/build_dataset.py          # 36-feature table (data/processed_local/, gitignored)
python scripts/eval_baseline.py          # rule baseline
for m in scripts/models/*.py; do python $m; done   # the 10 classifiers, same harness and folds
python scripts/optimize.py               # Bayesian optimisation of LightGBM (Optuna, ~85 min)
python scripts/models/advitha_lgbm_tuned.py && python scripts/compare_tuning.py
python scripts/cv_compare.py             # comparison table and figures
python scripts/loss_curves.py            # train/validation loss for boosting and SGD
python scripts/eval_hmm.py               # tune the HMM on train, classifier vs HMM
python scripts/export_artifacts.py       # selected model -> artifacts/
python scripts/eval_pipeline.py          # end to end on student_test
python scripts/make_maps.py --predicted  # reference vs predicted route maps
python scripts/results_table.py          # RESULTS.md
python report/build_report.py            # LaTeX version of the report (needs tectonic or xelatex)
```

A new model is one file in `scripts/models/` that builds an sklearn-compatible estimator and calls
`pathsnap.harness.evaluate_model(...)`; the harness fixes data, folds, metrics and output format.

## Two-laptop deployment

**Laptop 1 (API)**: the repo, the venv, `artifacts/` and `data/raw/evaluator/`.

```bash
./scripts/run_api.sh          # serves on 0.0.0.0:8000; request logs and scores print in this terminal
```

Windows: `scripts\run_api.ps1` after activating the venv; allow Python through the firewall on private networks.

**Laptop 2 (UI)**: needs only `ui/` and no data or model.

```bash
pip install -r ui/requirements.txt
export PATHSNAP_API_URL=http://<laptop-1-ip>:8000      # PowerShell: $env:PATHSNAP_API_URL = "..."
./scripts/run_ui.sh
```

Both laptops must be on the same network (a phone hotspot works if campus Wi-Fi blocks device-to-device traffic).
Check with `curl http://<laptop-1-ip>:8000/health`. To verify all evaluator trips on Laptop 1:

```bash
PATHSNAP_API_URL=http://127.0.0.1:8000 python scripts/smoke_test_api.py     # expect 10/10 passed
```

Endpoints: `GET /health`, `GET /evaluator/trips`, `POST /evaluator/predict`, plus `/student-test/*` for practice.

## Documentation

[`docs/data_quality_log.md`](docs/data_quality_log.md) (data issues and decisions with evidence),
[`docs/REFERENCE.md`](docs/REFERENCE.md) (data, preprocessing, tables, features),
[`docs/AI_USE.md`](docs/AI_USE.md) (AI-use disclosure).
