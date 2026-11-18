# PathSnap: Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, team 148.

PathSnap matches noisy GPS trajectories to a road network. It generates candidate road segments for every GPS point, scores them with a trained classifier, and decodes a connected route with an HMM/Viterbi decoder. A FastAPI server (Laptop 1) runs the pipeline. A Streamlit UI (Laptop 2) lets the evaluator pick a trip and see the result.

> **Status:** project foundation (loaders, geometry, network analysis, dataset inventory, API/UI skeletons). Candidate generation, features, models, HMM and evaluation are still to be built; see the TODO docstrings in `src/pathsnap/`.

## Repository layout

| Path | Contents |
|---|---|
| `configs/` | YAML configs: data partitions, candidate search, HMM, API |
| `data/raw/{train,student_test,evaluator}/` | Raw trip files (gitignored, not committed) |
| `src/pathsnap/` | Library code: `io`, `geo`, `network`, `labels`, `candidates`, `features`, `models`, `metrics`, `hmm`, `pipeline`, `viz` |
| `scripts/` | Runnable stages (`inventory.py`, …) and launchers (`run_api.sh`, `run_ui.sh`) |
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

Put each partition's raw files in its folder. Trips can be flat (`data/raw/train/<id>.track`, `<id>.nodes`, …) or one sub-folder per trip. The loader pairs files by name.

```
data/raw/train/          80 trips: fitting, cross-validation, tuning
data/raw/student_test/   10 trips: development comparisons
data/raw/evaluator/      10 trips: final evaluation only
```

**Leakage guard.** `pathsnap.io` refuses to read `data/raw/evaluator/` unless `PATHSNAP_MODE=serve` is set, which only `scripts/run_api.sh` does. Reference routes are read only when `with_route=True` is passed explicitly. `pathsnap.pipeline.predict_trip` takes no reference route, and evaluator scoring lives only in `api/evaluation.py`.

## Running the stages

```bash
python scripts/inventory.py      # EDA step 1 -> experiments/results/inventory.csv
```

More stages (candidates, features, model comparison, Bayesian optimisation, HMM tuning, artifact export, map generation) will be added under `scripts/`.

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
