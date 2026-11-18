# Advitha's model work: where it lives now

The first drafts in this folder (three model notebooks) and `pathsnap_features/` were replaced by
code on the shared foundation. They remain in the git history (commits `38a7229`, `4bf8739`).

| Draft | Replaced by |
|---|---|
| `pathsnap_features/features.py` | `src/pathsnap/features.py` (same feature ideas, vectorised, on the shared candidate table) + `scripts/build_dataset.py` |
| `01_lightgbm.ipynb` | `scripts/models/advitha_lgbm.py` |
| `02_histgradientboosting.ipynb` | `scripts/models/advitha_hgb.py` |
| `03_xgboost.ipynb` | `scripts/models/advitha_xgb.py` |
| (missing) shared harness | `src/pathsnap/harness.py`, `scripts/cv_compare.py` |

See `docs/HANDOFF_FEATURES_HARNESS.md` for how to run them and the current results.
