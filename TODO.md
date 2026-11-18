# TODO: what is left

Status as of 2026-11-18, after PRs #6 and #7 were merged (`main` at `ec042b5`). Tick items off here in your own PRs so this file stays the single source of truth. Requirements: [`docs/PLAN.md`](docs/PLAN.md); team split: [`SPLIT.md`](SPLIT.md).

**Overall: about 85% of the engineering and about 70% of the whole project.** All required ML work exists in code (models, tuning, HMM). What remains is merging Sudhe's branch, tuning and reporting the HMM, the two-laptop demo, and above all the **report**, which is the long pole: start it now, in parallel. Nothing of the PDF exists yet beyond fact sheets.

## Done

- [x] Inventory, EDA, data-quality log, Folium maps + screenshots (`docs/data_quality_log.md`, `maps/`, `report/figures/`)
- [x] Derived labels (monotonic DP) with quality checks and check maps
- [x] Candidate generation; R = 60 m, K = 30 chosen by miss rate on train
- [x] Feature table (36 features, `artifacts/feature_list.json`) and notebooks 00-02
- [x] Shared harness (`src/pathsnap/harness.py`) + comparison table (`experiments/results/model_comparison.csv`)
- [x] 7 models evaluated (≥ 5 required): XGBoost, LightGBM, LightGBM tuned, HistGB, Logistic Regression, SGD, Gaussian NB
- [x] **Bayesian optimisation** (PR #6): 40-trial Optuna study, every trial saved, untuned vs tuned table; CV Top-1 0.8639 → 0.8651 (`report/notes_advitha.md`)
- [x] **Tuned LightGBM exported** (PR #6) and served by the API; end to end student_test Top-1 0.880, ~0.2 s per trip
- [x] Figures: `cv_macro_f1.png`, `confusion_matrices.png`, `bayes_opt_history.png`, `bayes_opt_importance.png`
- [x] Final-model justification: facts + talking points in `report/notes_advitha.md`
- [x] FastAPI + Streamlit UI serving the trained model; separate student-test and evaluator API workflows (PR #7)
- [x] **Evaluator trips received** and placed in `data/raw/evaluator/` (gitignored; read only by the API with `PATHSNAP_MODE=serve`). `smoke_test_api.py`: **10/10 trips pass** on one laptop, each < 1 s
- [x] Report maps (reference vs predicted), error analysis, fact sheet `report/notes_adyanth.md`

## Pushed, not merged

- [ ] **`feat/sudhe-hmm`**: HMM/Viterbi decoder + tests, pipeline wiring, `metrics.py` + tests, Random Forest, and (latest push) `scripts/eval_hmm.py` plus Extra Trees and Decision Tree scripts on the shared trainer. Checked in an earlier trial merge: 40 tests pass, and on student_test the HMM lifts Top-1 0.880 → **0.902**, continuity 0.867 → **0.931**, breaks 653 → **253**; better on 9/10 trips (trip 9 slightly worse: 0.864 → 0.856). One conflict in the `return` block of `src/pathsnap/pipeline.py`: keep both sides (`decoder_breaks` from the branch, `scorer_name` from main). The Extra Trees / Decision Tree / HMM comparison *results* are not committed yet, only the scripts. Sudhe to finish pushing, then open a PR.

## Left, in priority order

| # | Task | Owner | Status | Rough time |
|---|---|---|---|---|
| 1 | Merge Sudhe's branch (conflict fix above); re-run `eval_pipeline.py` and the smoke test with the HMM on | Adyanth | ⏳ waiting for Sudhe's PR (PR #6 is merged) | 15 min |
| 2 | **Tune `configs/hmm.yaml`** (still placeholders) on train/student_test only, never evaluator | Sudhe | ❌ | 1 h |
| 3 | **Classifier-only vs HMM table** + failure cases (trip 9) + latency cost (HMM ~1.6 s per trip, max 3.5 s vs 0.2 s). `eval_pipeline.py` needs a `model_hmm` variant (`load_artifacts()` with the decoder) | Sudhe | ⚠️ numbers above from the trial merge | 1 h |
| 4 | Remaining models: run **Extra Trees, Decision Tree** and commit their results (Sudhe; scripts pushed, RF done). **KNN, MLP** (Tilak) are optional now that 7 models exist (≥ 5 required); do them only if time allows, and only the MLP gives real epoch loss curves. | Sudhe, Tilak | ⚠️ scripts only | 1-2 h |
| 5 | Use `metrics.py` in `api/evaluation.py` (UI shows Top-1 / miss rate as `None`) | Tilak (+ Adyanth) | ❌ needs Sudhe's merge | 30 min |
| 6 | **Two-laptop test** on all 10 evaluator trips over the real network (hotspot as backup); API terminal log + UI screenshots for the report. Unzip `evaluator_test.zip` into `data/raw/evaluator/` on the API laptop. Re-run after the HMM is merged. | Tilak | ⚠️ one-laptop smoke test passes (10/10) | 30-60 min |
| 7 | **PDF report** (sections 2.2-8), each person writes their own sections in their own words (AI-text checks). Facts: `report/notes_adyanth.md`, `report/notes_advitha.md`; Sudhe and Tilak need their own fact sheets | everyone | ❌ | **several hours** |
| 8 | README final pass, AI-use disclosure check (`docs/AI_USE.md`), branch protection on `main` | anyone | ⚠️ | 20 min |
| 9 | Mock viva: everyone explains every part (checklists at the end of each `report/notes_*.md`) | everyone | ❌ | 1 h |

## Who does what now

| Person | Items |
|---|---|
| Sudhe | open the PR; 2, 3, 4 (commit ET/DT results); fact sheet for the HMM section |
| Tilak | 5, 6; 4 (KNN/MLP) only if time allows; fact sheet for the deployment section |
| Advitha | report sections 3.1-3.3 and 4 from `report/notes_advitha.md`; viva checklist |
| Adyanth | 1 (merges), own report sections, help with 5, check numbers in report drafts |
| Everyone | 7, 8, 9 |

## Rules that still apply

- **Evaluator trips:** no training, tuning, threshold or decoder choice on them; never commit them (`data/raw/evaluator/**` is gitignored). Don't look at evaluator scores before the demo.
- Every number in the report must come from a file in `experiments/results/` and be reproducible from the README run order.

## How to plug things in (no code changes elsewhere needed)

- **New model:** `scripts/models/<you>_<model>.py` with `build()` + `CONFIG`, calling `pathsnap.harness.evaluate_model(...)`; then `python scripts/cv_compare.py`. Slow models: `max_train_obs=`.
- **HMM:** `make_decoder(config)` in `src/pathsnap/hmm.py`; `pipeline.load_artifacts()` and the API pick it up. Student_test probabilities for offline tuning: `data/processed_local/preds/<model>_student_test.parquet`.
- **Re-export:** `python scripts/export_artifacts.py [name]`, then `python scripts/eval_pipeline.py`.

## Remaining effort at a glance

| Block | Left |
|---|---|
| Merge HMM, re-run end to end | ~15 min (after Sudhe's PR) |
| Tune and report the HMM | ~2 h (Sudhe) |
| API metrics + two-laptop demo | ~1-1.5 h (Tilak) |
| README, disclosure, branch protection | ~20 min |
| **PDF report** | **several hours, split across four people** |
| Mock viva | ~1 h |

The ML is done. The remaining risk is the report and the live demo, not the models.
