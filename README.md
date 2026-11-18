# PathSnap model notebooks

These notebooks are deliberately written against a **feature-engineered tabular dataset**, because the supplied PathSnap documents require the team to investigate the actual raw files and design its own candidate generation and feature extraction.

Files:
- `01_lightgbm.ipynb`
- `02_histgradientboosting.ipynb`
- `03_xgboost.ipynb`

## Expected feature table

One row should represent one candidate associated with one GPS observation.

Required columns:
- `trip_id`: trip identifier used only for group-aware splitting
- `target_arc_id`: ground-truth road-segment/candidate label
- numeric feature columns

Optional:
- `partition`: `train` / `student_test`

The notebooks automatically exclude `trip_id`, `target_arc_id`, and `partition` from model features.

## Important

The notebooks do not fabricate metrics. Run them on the team's actual feature table and use the generated CSV/JSON artifacts for the technical report.

The PathSnap requirements specify:
- 80 training trips for fitting/cross-validation
- 10 student-testing trips for development-stage comparison
- 10 evaluator-testing trips for final evaluation only
- group-aware validation by trip
- actual experimental results for model comparison
- at least five models overall before final model selection

LightGBM and XGBoost require their respective Python packages. HistGradientBoosting is included in scikit-learn.
