# TODO: what is left after Review 1

Status as of 2026-11-18, after the first presentation (`main` at `cda1a69`). Tick items off here in your own PRs so this file stays the single source of truth. Requirements: [`docs/PLAN.md`](docs/PLAN.md); team split: [`SPLIT.md`](SPLIT.md).

**Overall: about 60-65% done.** The ML foundation is complete. What remains is the decoder, tuning, proof that deployment works, and writing.

## Done (on `main`)

- [x] Inventory, EDA, data-quality log, Folium maps + screenshots (`docs/data_quality_log.md`, `maps/`, `report/figures/`)
- [x] Derived labels (monotonic DP) with quality checks and check maps
- [x] Candidate generation; R = 60 m, K = 30 chosen by miss rate on train
- [x] Feature table (36 features, `artifacts/feature_list.json`) and notebooks 00-02
- [x] Shared harness (`src/pathsnap/harness.py`) + comparison table (`experiments/results/model_comparison.csv`)
- [x] 6 models evaluated (≥ 5 required): XGBoost, LightGBM, HistGB, Logistic Regression, SGD, Gaussian NB
- [x] Selected model (LightGBM, best CV Top-1) exported to `artifacts/` and running inside `pipeline.predict_trip` (student_test Top-1 0.879)
- [x] FastAPI + Streamlit UI serving the trained model (toy-fixture tests pass)
- [x] Report maps (reference vs predicted), error analysis, fact sheet `report/notes_adyanth.md`

## Left, in priority order

| # | Task | Owner | Status | Why it matters |
|---|---|---|---|---|
| 1 | **HMM / Viterbi decoder** in `src/pathsnap/hmm.py`, exposing `make_decoder(config)` (contract in `docs/HANDOFF_ADYANTH.md`), tuned on train/student_test only | Sudhe | ❌ not started (`metrics.py` only, unmerged) | Required (report section 5). Biggest open item. Should fix junction errors (43% of current errors) and raise continuity above 0.87 |
| 2 | **HMM evaluation:** classifier-only vs HMM table, continuity, failure cases, where it did not help | Sudhe | ❌ | Required table |
| 3 | **Bayesian optimisation** of LightGBM or XGBoost: Optuna study saving every trial, untuned vs tuned table, compute cost | Advitha | ✅ 40-trial Optuna study, `scripts/optimize.py`; CV Top-1 0.8639 → 0.8651 (`report/notes_advitha.md`) | Required (report section 4) |
| 4 | Re-export the tuned model: `python scripts/export_artifacts.py <tuned_model_name>` | Advitha | ✅ `advitha_lgbm_tuned` exported; end-to-end Top-1 0.880, ~209 ms per trip | The API serves it automatically |
| 5 | **Remaining models** through the harness: RF / ET / DT (Sudhe), KNN / MLP (Tilak); MLP with real epoch train/val loss curves | Sudhe, Tilak | ❌ | Assigned in SPLIT.md; MLP is the only model with genuine loss curves |
| 6 | Confusion-matrix figures + CV macro F1 bar chart with fold error bars | Advitha (`scripts/cv_compare.py`) | ✅ `cv_macro_f1.png`, `confusion_matrices.png` | Report section 3.3 |
| 7 | Written justification for the final model | Advitha | ✅ facts + talking points in `report/notes_advitha.md` (prose still to write in own words) | Report section 3.1 |
| 8 | **Two-laptop test on all 10 evaluator trips** + `scripts/smoke_test_api.py`; capture API terminal logs and UI screenshots | Tilak | ⚠️ works on the toy fixture only | Mandatory demo; test on the real network, phone hotspot as backup |
| 9 | Merge Sudhe's `metrics.py` and use it in `api/evaluation.py` (Top-1 and miss rate are currently `None`) | Sudhe + Tilak | ⚠️ | Metrics shown in the UI |
| 10 | **PDF report** in the required structure (sections 2.2-8) | everyone, own sections | ❌ | The deliverable. Must be in our own words (AI-text checks); facts in `report/notes_adyanth.md` |
| 11 | README final pass, AI-use disclosure check (`docs/AI_USE.md`), branch protection on `main` | anyone | ⚠️ | Submission hygiene |
| 12 | Mock viva: everyone explains every part | everyone | ❌ | Assessment policy section 6 |

## Share of the remaining work

| Person | Share | Items |
|---|---|---|
| Sudhe | ~40% (critical path) | 1, 2, 5 (RF/ET/DT), 9 |
| Advitha | done (by Adyanth) | 3, 4, 6, 7; own report sections 3.1-3.3, 4 from `report/notes_advitha.md` |
| Tilak | ~20% | 5 (KNN/MLP), 8, 9 |
| Everyone | ~20% | 10, 11, 12 |
| Adyanth | own part done | own report sections; can help with 1, 6 and checking numbers in report drafts |

## How to plug things in (no code changes elsewhere needed)

- **New model:** `scripts/models/<you>_<model>.py` with `build()` + `CONFIG`, calling `pathsnap.harness.evaluate_model(...)`; then `python scripts/cv_compare.py`. Slow models: `max_train_obs=`.
- **HMM:** define `make_decoder(config) -> decoder(geom, cands, scores) -> np.ndarray` (one arc id per observation) in `src/pathsnap/hmm.py`; `pipeline.load_artifacts()` and the API pick it up. Student_test probabilities for offline tuning: `data/processed_local/preds/<model>_student_test.parquet` (rebuild with the model scripts).
- **Tuned model:** `python scripts/export_artifacts.py <name>`, then `python scripts/eval_pipeline.py` to confirm end-to-end numbers.
