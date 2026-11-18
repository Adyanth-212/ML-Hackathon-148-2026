# Handoff: Adyanth's data foundation

Branch `feat/adyanth-data`. Everything below runs from a clean clone after `pip install -r requirements.txt pyarrow && pip install -e .`. Runtimes measured on an 8-core VM. Decisions and evidence: [`docs/data_quality_log.md`](data_quality_log.md).

## What exists

| Stage | Code | Output | Runtime |
|---|---|---|---|
| Inventory | `scripts/inventory.py` | `experiments/results/inventory.csv` | ~10 s |
| EDA | `scripts/eda.py` | `experiments/results/eda_*.csv`, `experiments/figures/eda_*.png` | ~2 s |
| EDA maps | `scripts/make_maps.py`, `src/pathsnap/viz.py` | `maps/eda_*.html` (6 trips), `report/figures/map_eda_*.png` | ~7 s |
| Labels | `src/pathsnap/labels.py`, `scripts/build_labels.py` | `data/processed/labels/<part>/<trip>.parquet`, `experiments/results/label_quality.csv` | ~2 s |
| Label maps | `scripts/make_maps.py --labels` | `maps/labels_*.html` (7 trips) | ~10 s |
| R/K sweep | `scripts/candidate_sweep.py` | `experiments/results/candidate_miss_rate.csv`, figure | ~2 s |
| Candidates | `src/pathsnap/candidates.py`, `scripts/build_candidates.py` | `data/processed/candidates/<part>/<trip>.parquet`, `experiments/results/candidate_stats.csv` | ~1.5 s |
| LogReg / GNB / SGD | `scripts/models/adyanth_{logreg,gnb,sgd}.py` via `pathsnap.harness` | `experiments/results/models/adyanth_*.json`, `ablation_*.json` | ~90 s / ~10 s / ~40 s |
| Export selected model | `scripts/export_artifacts.py [model_name]` | `artifacts/model.joblib`, `artifacts/metadata.json` | ~15 s |
| Pipeline evaluation | `scripts/eval_pipeline.py` | `experiments/results/pipeline_*.csv` | ~2 s |
| Report maps | `scripts/make_maps.py --predicted` | `maps/predicted_student_test_*.html`, `report/figures/map_predicted_*.png` | ~5 s |

Geometry helper for everyone: `pathsnap.geo.TripGeometry(track, nodes, arcs)` projects one trip to its UTM zone once and exposes `obs_x/obs_y`, `node_x/node_y`, arc endpoints `ax, ay, bx, by`, `arc_lengths()`, `obs_arc_distance(obs, arcs)` and `distance_matrix(arc_ids)`.

## Interfaces

**Candidate table** (exactly the SPLIT.md contract), one parquet per trip, committed (29 MB):
`trip_id (string), obs_idx (int32), arc_id (int32), dist_m (float32), proj_frac (float32, unclipped), label (int8)`.
Rows are sorted by `obs_idx`, then `(dist_m, arc_id)`. R = 60 m, K = 30, fallback 150 m (`configs/candidates.yaml`).
At inference time call `generate_candidates(TripGeometry.from_trip(trip), trip_id)`: it needs no route and returns `(table_without_label, obs_ids_without_candidates)`.

**Label table** (per observation), `data/processed/labels/<part>/<trip>.parquet`:
`obs_idx, route_pos, arc_id, dist_m, is_outlier, after_gap`. Join on `(trip_id from the file name, obs_idx)`.

## Headline numbers

- Labels: median distance to the aligned arc 4.75 m (median over trips). 11.1% of train points (5 trips account for most) and 0.14% of student_test points are > 50 m from the route, mostly because the **reference route covers only part of the GPS track** in trips 00000054, 00000033 and 00000002.
- Candidates: mean 11.75 per point (train), 12.74 (student_test). Miss rate on reliable labels: train 0.029%, student_test 0.000%; all points: 6.46% / 0.13%.
- Models: the prototype in `scripts/models/_adyanth_proto.py` has been replaced by the shared harness (`src/pathsnap/harness.py`). Current results for LogReg, GNB, the boosting models and the feature-group ablations are in `experiments/results/model_comparison.csv` and [`docs/HANDOFF_FEATURES_HARNESS.md`](HANDOFF_FEATURES_HARNESS.md). Key finding, unchanged: distance-only features give no more than the nearest-arc rule (Top-1 ~0.54); direction features add ~0.30.

## Pipeline (works now, with the trained model)

`pathsnap.pipeline.predict_trip(trip, artifacts)` returns `{"gps", "pred_arcs", "route_lonlat", "timings_ms", "no_candidate_obs", "method"}`. It never takes a route.
- `artifacts = load_artifacts()` loads `artifacts/model.joblib` (currently LightGBM, chosen by best CV Top-1) and scores candidates with `features.build_features`. Student_test Top-1 is 0.879 with continuity 0.867, at ~77 ms per trip (`experiments/results/pipeline_summary.csv`).
- `artifacts = None` (or `{}`) uses the rule-based baseline: distance + 20 m × heading mismatch, Top-1 0.832.
- **HMM hook (Sudhe):** if `pathsnap.hmm` defines `make_decoder(config: dict)` returning `decoder(geom, cands, scores) -> np.ndarray` (one arc id per observation), `load_artifacts()` picks it up automatically with `configs/hmm.yaml`. `scores` are the classifier probabilities in candidate-row order.
- **Tilak (API):** nothing to change. `api/main.py`'s own loader returns `{"mode": "classifier", "model": est}`, and `predict_trip` wraps that `model` in the same feature-based scorer (checked: identical predictions to `pipeline.load_artifacts()`). The response keeps every field the UI uses (`route_arc_ids`, `network_lonlat`, `nearest_dist_m`, `model`, `timings_ms.total`). Re-export after Bayesian tuning with `python scripts/export_artifacts.py <tuned_model_name>`.

## For the others

- **Advitha (features):** the single most important feature is **direction** (measured above: +0.3 Top-1): the angle between the GPS heading (central difference, ignore points that moved < 5 m) and the arc bearing. Without it, Top-1 is capped near 0.53 because about 92% of arcs have an identical reverse twin. `labels.heading_mismatch(geom, arc_ids, min_move_m)` already computes `(1 - cos)/2` vectorised; feel free to reuse it. Clip or robustify speed features: trips 00000032/00000013/00000001 have sustained steps > 60 m/s.
- **Harness:** done (`src/pathsnap/harness.py`); it drops `is_outlier` observations and records the counts, with `GroupKFold(5, shuffle=True, random_state=148)` by trip.
- **Sudhe (HMM):** 198 train observations have no candidate even at 150 m (student_test: 0). Gaps > 30 s: 135 in train; the label aligner treats them as free skips, which suggests breaking or relaxing transitions there too. Networks have up to 421 weak components, so expect unreachable transitions.
- **Tilak (API):** maps only draw arcs within 200 m of the track (`viz.arcs_near_track`), which keeps the HTML small for 300k-arc networks. The basemap uses Esri Light Gray tiles, because OSM tiles return 403 from `file://` and CARTO now needs a key.

## Open issues

- The reference route covers only part of the track in a few trips; those labels are flagged, not fixed.
- Top-1 counts an observation without a positive candidate (a miss) as wrong; at R = 60 m / K = 30 this is negligible on reliable labels.
- The pipeline runs the exported classifier; the HMM plugs in through `hmm.make_decoder` (above). About 81% of the reverse-twin "errors" whose heading can be trusted look like label errors in the reference route (data_quality_log section 10).
