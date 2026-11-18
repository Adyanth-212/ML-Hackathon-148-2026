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
| LogReg / GNB | `scripts/models/adyanth_{logreg,gnb}.py` (+ `_adyanth_proto.py`) | `experiments/results/models/adyanth_*.json` | ~13 s / ~9 s |

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
- Prototype models (simple distance-only features, outlier observations dropped, GroupKFold(5, shuffle, seed 148)):

| model | CV macro F1 (mean ± std) | student_test Top-1 | Top-1 counting reverse twin | macro F1 | log loss | fit s | ms / trip |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.767 ± 0.008 | 0.539 | 0.875 | 0.766 | 0.257 | 1.1 | 2.9 |
| Gaussian NB | 0.769 ± 0.007 | 0.529 | 0.875 | 0.770 | 0.748 | 0.2 | 3.5 |
| nearest-arc rule (reference, not a model) | – | 0.530 | 0.875 | – | – | – | – |

  Honest reading: with distance-only features both models do no better than "pick the nearest arc". Almost all remaining errors are the reverse twin of the right arc (same geometry, opposite direction). GNB's probabilities are badly calibrated (log loss 0.75 vs 0.26) because its features are strongly correlated. Train and CV F1 are equal, so there is no overfitting; the models are underfitting for lack of information.

## Baseline pipeline (works now)

`pathsnap.pipeline.predict_trip(trip, artifacts=None)` returns `{"gps", "pred_arcs", "route_lonlat", "timings_ms", "no_candidate_obs", "method"}`. It never takes a route.
- `artifacts=None` gives the baseline rule: `score = -(dist_m + 20 m * heading mismatch)`, with the 20 m weight chosen on train (`scripts/eval_baseline.py`). Student_test Top-1 is 0.832 (nearest arc alone: 0.530), continuity 0.73, ~21 ms per trip.
- `artifacts["scorer"](geom, cands) -> np.ndarray` (higher = better, e.g. `predict_proba[:, 1]` on your features) replaces the rule.
- `artifacts["decoder"](geom, cands, scores) -> arc per observation` replaces the argmax (Sudhe's Viterbi).
- **Tilak:** `api/main.py` already calls `predict_trip(trip, ARTIFACTS)` with `ARTIFACTS = None`, so `/predict` now returns a real route instead of a 501.

## For the others

- **Advitha (features):** the single most important feature is **direction**: the angle between the GPS heading (central difference, ignore points that moved < 5 m) and the arc bearing. Without it, Top-1 is capped near 0.53 because about 92% of arcs have an identical reverse twin. `labels.heading_mismatch(geom, arc_ids, min_move_m)` already computes `(1 - cos)/2` vectorised; feel free to reuse it. Clip or robustify speed features: trips 00000032/00000013/00000001 have sustained steps > 60 m/s.
- **Advitha (harness):** drop `is_outlier` observations from training and evaluation (or report both) and record the count, as `_adyanth_proto.py` does. My CV used `GroupKFold(n_splits=5, shuffle=True, random_state=148)`; if the harness uses a different fold assignment, my JSONs will be regenerated through it. Once `harness.py` is on main, I'll switch `scripts/models/adyanth_*.py` to it.
- **Sudhe (HMM):** 198 train observations have no candidate even at 150 m (student_test: 0). Gaps > 30 s: 135 in train; the label aligner treats them as free skips, which suggests breaking or relaxing transitions there too. Networks have up to 421 weak components, so expect unreachable transitions.
- **Tilak (API):** maps only draw arcs within 200 m of the track (`viz.arcs_near_track`), which keeps the HTML small for 300k-arc networks. The basemap uses Esri Light Gray tiles, because OSM tiles return 403 from `file://` and CARTO now needs a key.

## Open issues

- The reference route covers only part of the track in a few trips; those labels are flagged, not fixed.
- Top-1 counts an observation without a positive candidate (a miss) as wrong; at R = 60 m / K = 30 this is negligible on reliable labels.
- `pipeline.predict_trip` now runs a rule-based baseline (nearest arc + heading penalty, student_test Top-1 0.832). The trained model and the HMM still need to be plugged in as `artifacts["scorer"]` / `artifacts["decoder"]`.
