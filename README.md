# PathSnap: Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, team 148.

PathSnap matches noisy GPS trajectories to a road network. It generates candidate road segments for every GPS point, scores them with a trained classifier, and decodes a connected route with an HMM/Viterbi decoder. A FastAPI server (Laptop 1) runs the pipeline. A Streamlit UI (Laptop 2) lets the evaluator pick a trip and see the result.

> **Team split and workflow: see [SPLIT.md](SPLIT.md).**
>
> **Status:** project foundation (loaders, geometry, network analysis, dataset inventory, API/UI skeletons). Candidate generation, features, models, HMM and evaluation are still to be built; see the TODO docstrings in `src/pathsnap/`.

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

## Tilak's API/UI deployment milestone

The API starts with a nearest-directed-arc baseline when `artifacts/model.joblib`
is absent. This is the deliberately simple end-to-end stub for the toy fixture;
it is not the final learned map-matching model. The prediction pipeline accepts
only GPS observations and network geometry. Evaluator reference routes are read
inside `api/evaluation.py` only after prediction and are never included in the
API response.

### Windows setup (Laptop 1)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
$env:PATHSNAP_MODE = "serve"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or run `scripts/run_api.ps1` after activating the virtual environment. Allow
Python through Windows Defender Firewall on private networks.

### Windows setup (Laptop 2)

Install only the UI dependencies, then point the UI to Laptop 1's LAN address:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ui/requirements.txt
$env:PATHSNAP_API_URL = "http://<laptop-1-ip>:8000"
python -m streamlit run ui/app.py
```

### API verification

With the API running and evaluator files installed on Laptop 1:

```powershell
$env:PATHSNAP_API_URL = "http://127.0.0.1:8000"
python scripts/smoke_test_api.py
```

The smoke test calls `/predict` for every trip returned by `/trips`, verifies
that one arc prediction is returned per GPS observation, and checks that
server-side metrics are present. The toy-fixture integration tests run with
`pytest` and do not require evaluator data.
