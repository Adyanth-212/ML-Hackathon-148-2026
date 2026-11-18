# TODO: what is left

Status as of 2026-11-18 ~17:00, with PR #9 (`feat/hmm-integration`). Tick items off here in your own PRs so this file stays the single source of truth. Requirements: [`docs/PLAN.md`](docs/PLAN.md); team split: [`SPLIT.md`](SPLIT.md).

**Overall: about 95%.** All the ML work is done and reported. What is left is the **two-laptop run on the real machines** (with screenshots for report section 7), a read-through of the report, and the mock viva.

**Best result (student_test, end to end through the API code path): Top-1 0.913**, continuity 0.930, route breaks 259 (classifier alone 0.880 / 0.867 / 653; rule baseline 0.832).

## Done

- [x] Inventory, EDA, data-quality log, Folium maps + screenshots (`docs/data_quality_log.md`, `maps/`, `report/figures/`)
- [x] Derived labels (monotonic DP) with quality checks; candidates R = 60 m, K = 30 chosen on train
- [x] Feature table (36 features), shared harness, notebooks 00-02
- [x] **10 models** through the same harness (≥ 5 required): LightGBM, LightGBM tuned, XGBoost, HistGB, Random Forest, Extra Trees, Decision Tree, Logistic Regression, SGD, Gaussian NB (`experiments/results/model_comparison.csv`)
- [x] **Bayesian optimisation**: 40-trial Optuna study, every trial saved, untuned vs tuned table (PR #6)
- [x] **Tuned LightGBM exported** and served by the API
- [x] **HMM/Viterbi** (Sudhe) merged, tuned on train with out-of-fold scores (`configs/hmm.yaml`), classifier vs HMM table + per-trip + failure analysis (`scripts/eval_hmm.py`, `experiments/results/hmm_*.csv`)
- [x] Figures: CV macro F1 with folds, confusion matrices, model comparison, Bayes-opt history/importance, HMM per-trip
- [x] **API**: evaluator endpoints score point Top-1, miss rate, route P/R/F1, continuity on the server after prediction; scores printed in the Laptop 1 log
- [x] **UI** lists the 10 evaluator trips by default (student-test as practice mode)
- [x] `smoke_test_api.py` targets the evaluator endpoints
- [x] **PDF report** `report/report.pdf`, generated from the results by `python report/build_report.py` (every table read from `experiments/results/`); AI use disclosed in section 8

## Left

| # | Task | Owner | Status | Time |
|---|---|---|---|---|
| 1 | Merge PR #9 with **"Create a merge commit"** (keeps Sudhe's authorship) | Adyanth | ⏳ | 2 min |
| 2 | **Two-laptop run** on the real laptops: Laptop 1 = API (full repo + `data/raw/evaluator/` from `evaluator_test.zip`), Laptop 2 = UI. `curl http://<ip>:8000/health` from Laptop 2 first; phone hotspot if campus Wi-Fi blocks it. Run `python scripts/smoke_test_api.py` on Laptop 1 (expect 10/10) | Tilak + Adyanth | ❌ | 30 min |
| 3 | Screenshots from that run: UI showing an evaluator trip → `report/figures/ui_evaluator.png`; API terminal on Laptop 1 → `report/figures/api_terminal.png`; then `python report/build_report.py` | Tilak | ❌ | 10 min |
| 4 | Read the report end to end; fix anything you would not say in the viva | everyone | ❌ | 30 min |
| 5 | README final pass, branch protection on `main` | anyone | ⚠️ | 15 min |
| 6 | Mock viva: everyone explains every part (checklists in `report/notes_*.md`) | everyone | ❌ | 1 h |
| - | Optional: KNN / MLP (MLP would add real epoch loss curves). Not needed: 10 models exist | Tilak | optional | 1-2 h |

## Rules that still apply

- **Evaluator trips:** no training, tuning, threshold or decoder choice on them; never commit them (`data/raw/evaluator/**` is gitignored). Don't look at evaluator scores before the demo.
- Every number in the report comes from a file in `experiments/results/`; rebuild the PDF after any re-run.

## How to plug things in

- **New model:** `scripts/models/<you>_<model>.py` with `build()` + `CONFIG`, calling `pathsnap.harness.evaluate_model(...)`; then `python scripts/cv_compare.py`.
- **Re-export:** `python scripts/export_artifacts.py [name]`, then `python scripts/eval_pipeline.py`.
- **HMM:** `python scripts/eval_hmm.py` re-tunes `configs/hmm.yaml` on train (`--prep-dir DIR` reuses out-of-fold scores).
- **Report:** `python report/build_report.py` (needs `xelatex`).
