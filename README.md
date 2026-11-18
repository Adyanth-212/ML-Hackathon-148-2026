# PathSnap: Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, team 148.

PathSnap matches noisy GPS trajectories to a road network. It generates candidate road segments for every GPS point, scores them with a trained classifier, and decodes a connected route with an HMM/Viterbi decoder. A FastAPI server (Laptop 1) runs the pipeline. A Streamlit UI (Laptop 2) lets the evaluator pick a trip and see the result.

> **Team split and workflow: see [SPLIT.md](SPLIT.md). Data, preprocessing and features: see [REFERENCE.md](REFERENCE.md).**
>
> **Status (Review 1):** data foundation, EDA, maps, labels, candidates, feature table, shared evaluation harness and five classifiers are done (`experiments/results/model_comparison.csv`). `pipeline.predict_trip` works end to end with a rule-based scorer. Still to do: Bayesian optimisation, HMM/Viterbi, model export into the pipeline, the remaining models (RF/ET/DT, KNN/MLP), API/UI integration and the report.

## Repository layout

| Path | Contents |
|---|---|
| `configs/` | YAML configs: data partitions, candidate search, HMM, API |
| `data/raw/{train,student_test,evaluator}/` | Raw trip files (gitignored, not committed) |
| `src/pathsnap/` | Library code: `io`, `geo`, `network`, `labels`, `candidates`, `features`, `models`, `metrics`, `hmm`, `pipeline`, `viz` |
| `scripts/` | Runnable stages (`inventory.py`, …), launchers (`run_api.sh`, `run_ui.sh`) and per-person model scripts in `scripts/models/` |
| `notebooks/` | EDA and experiment notebooks |
| `experiments/` | Results CSVs, logs, Optuna study, figures |
| `artifacts/` | Exported model, preprocessing and HMM config used by the API |
| `maps/` | Generated Folium HTML maps |
| `api/` | FastAPI backend (Laptop 1) |
| `ui/` | Streamlit evaluator UI (Laptop 2) |
| `report/` | Technical report sources and figures |
| `tests/` | Unit tests (toy fixtures) |

## Setup

Requires Python 3.12 (3.10–3.13 work; 3.14 is too new for some ML wheels).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

On macOS, LightGBM and XGBoost need OpenMP: `brew install libomp`.

## Data

`train` (80 trips) and `student_test` (10 trips) are committed in the repo (about 90 MB). `evaluator` is never committed; put those files in `data/raw/evaluator/` on Laptop 1 only. Trips can be flat (`data/raw/train/<id>.track`, `<id>.nodes`, …) or one sub-folder per trip. The loader pairs files by name.

```
data/raw/train/          80 trips: fitting, cross-validation, tuning
data/raw/student_test/   10 trips: development comparisons
data/raw/evaluator/      10 trips: final evaluation only
```

**Leakage guard.** `pathsnap.io` refuses to read `data/raw/evaluator/` unless `PATHSNAP_MODE=serve` is set, which only `scripts/run_api.sh` does. Reference routes are read only when `with_route=True` is passed explicitly. `pathsnap.pipeline.predict_trip` takes no reference route, and evaluator scoring lives only in `api/evaluation.py`.

## Running the stages

```bash
python scripts/inventory.py              # dataset inventory           -> experiments/results/inventory.csv
python scripts/eda.py                    # EDA tables + figures        -> experiments/results/eda_*.csv, experiments/figures/eda_*.png
python scripts/make_maps.py              # Folium EDA maps             -> maps/eda_*.html
python scripts/build_labels.py           # per-observation labels      -> data/processed/labels/, experiments/results/label_quality.csv
python scripts/make_maps.py --labels     # label check maps            -> maps/labels_*.html
python scripts/candidate_sweep.py        # R/K miss-rate sweep (train) -> experiments/results/candidate_miss_rate.csv
python scripts/build_candidates.py       # candidate tables            -> data/processed/candidates/, experiments/results/candidate_stats.csv
python scripts/eval_baseline.py          # rule baseline via predict_trip -> experiments/results/baseline_*.csv
python scripts/build_dataset.py          # shared feature table        -> data/processed_local/features/ (gitignored), artifacts/feature_list.json
python scripts/models/adyanth_logreg.py  # each model script calls pathsnap.harness.evaluate_model
python scripts/models/adyanth_gnb.py     #   -> experiments/results/models/<name>.json
python scripts/models/advitha_lgbm.py
python scripts/models/advitha_hgb.py
python scripts/models/advitha_xgb.py
python scripts/cv_compare.py             # comparison table + figure  -> experiments/results/model_comparison.csv
```

**Adding a model:** write `scripts/models/<you>_<model>.py` that builds an sklearn-compatible estimator (scalers inside a `Pipeline`) and calls `pathsnap.harness.evaluate_model(name, family, estimator, config)`. The harness fixes the data, the folds (`experiments/results/cv_folds.csv`), the metrics and the output format, so every model is comparable. For slow models pass `max_train_obs=` to subsample training observations (recorded in the JSON).

Each stage runs in seconds on 8 cores (all trips in parallel, seed 148). The candidate tables (~30 MB) are committed, so you only need to rerun these if you change `configs/candidates.yaml` or `labels.py`. Decisions and their evidence: [`docs/data_quality_log.md`](docs/data_quality_log.md); handoff notes: [`docs/HANDOFF_ADYANTH.md`](docs/HANDOFF_ADYANTH.md).

Notebooks `notebooks/01_eda.ipynb` and `notebooks/02_labels_candidates.ipynb` walk through the same stages (they call these scripts) and are committed fully executed.

More stages (Bayesian optimisation, HMM tuning, artifact export) will be added under `scripts/`.

## Two-laptop deployment

**Laptop 1 (API):** has the repo, the venv, `artifacts/` and `data/raw/evaluator/`.

```bash
./scripts/run_api.sh             # prints this laptop's IP; serves on 0.0.0.0:8000
```

Request logs (trip, client IP, status, latency) print to this terminal. Show them to the evaluator.

**Laptop 2 (UI):** needs only `ui/`. It holds no data and no model.

```bash
pip install -r ui/requirements.txt
export PATHSNAP_API_URL=http://<laptop-1-ip>:8000
./scripts/run_ui.sh
```

Both laptops must be on the same network. Campus Wi-Fi often blocks traffic between devices, so a phone hotspot is the reliable fallback. Allow incoming connections for Python on Laptop 1 if the firewall asks. Check from Laptop 2 with `curl http://<laptop-1-ip>:8000/health`.
