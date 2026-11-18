# Reference: data, preprocessing and features

For everyone on the team. It describes what the data is, what preprocessing has already been done, what the tables mean, and which features exist. Evidence and numbers: [`docs/data_quality_log.md`](docs/data_quality_log.md). Walk through one trip step by step: [`notebooks/00_before_after.ipynb`](notebooks/00_before_after.ipynb) (outputs saved, you can read it without running anything).

## 1. The raw data

- `data/raw/train/` has 80 trips and `data/raw/student_test/` has 10, one folder per trip (`<8-digit id>/`). The 10 evaluator trips are never in the repo.
- Each trip has four header-less, whitespace-separated files:

| File | Columns | Notes |
|---|---|---|
| `<id>.track` | `lon lat t` | the noisy GPS points; t in seconds; about 1 Hz |
| `<id>.nodes` | `lon lat` | row number is the node id |
| `<id>.arcs` | `from_node to_node` | row number is the arc id; a directed road segment |
| `<id>.route` | `arc_id` | the reference route: arc ids in driving order |

- **Every trip has its own road network.** Arc ids only mean something inside one trip. Networks go up to 155k nodes and 307k arcs.
- **A row in `.track` does not correspond to a row in any other file.** The route is an ordered list of arcs, not one label per GPS point.
- A two-way street is two arcs with identical geometry (about 92% of arcs have a reverse twin).
- Load trips with `pathsnap.io` (`discover_trips`, `load_trip`). The loader refuses the evaluator partition unless `PATHSNAP_MODE=serve`, and reads reference routes only with `with_route=True`.

## 2. Preprocessing that is already done

Everything below is implemented in `src/pathsnap/` and run by `scripts/`. **Do not redo it in your own code.** Use the saved tables or the library functions.

| # | Step | What happens | Code |
|---|---|---|---|
| 1 | Quality check | Missing values, bad coordinates, time gaps, speeds, networks and routes are measured. **No rows are deleted.** Problems are flagged. | `scripts/inventory.py`, `scripts/eda.py` |
| 2 | Projection | Each trip is projected into the UTM zone of its median GPS point, so x and y are in metres. Trips are in 25 different zones. Distances on raw degrees are never used. | `pathsnap.geo.TripGeometry` |
| 3 | Labels | Each GPS point is aligned to one arc of the reference route by a monotonic dynamic programme (route position never goes backwards; cost is distance in metres, capped at 100 m, plus a heading penalty; skipping route arcs is free across gaps over 30 s). | `pathsnap.labels`, `scripts/build_labels.py` |
| 4 | Candidates | For each GPS point, the arcs within **R = 60 m**, at most **K = 30** nearest, found with a spatial index. If none is found, one retry at 150 m. | `pathsnap.candidates`, `scripts/build_candidates.py`, `configs/candidates.yaml` |

**R and K are frozen.** Changing them invalidates everyone's feature tables. They were chosen from the miss-rate sweep on training trips only (`experiments/results/candidate_miss_rate.csv`).

**The labels are derived, not ground truth.** The reference route is not independently verified. In trips 54, 33 and 2 it covers only part of the GPS track, so 11.1% of train points (0.14% of student-test) are more than 50 m from their aligned arc and flagged `is_outlier`.

## 3. The tables

**Candidate table** `data/processed/candidates/<partition>/<trip_id>.parquet`: one row per (GPS point, candidate arc), sorted by `obs_idx`, then `(dist_m, arc_id)`.

| Column | Meaning |
|---|---|
| `trip_id` | trip id (string) |
| `obs_idx` | row number of the GPS point in the `.track` file |
| `arc_id` | the candidate road (local to the trip) |
| `dist_m` | distance from the GPS point to the arc, in metres |
| `proj_frac` | where the point projects onto the arc; **unclipped**: below 0 or above 1 means the nearest point is an endpoint |
| `label` | 1 if this is the arc the point was aligned to, else 0 |

Each GPS point has at most one row with `label = 1`. A point with none is a candidate miss (rare).

**Label table** `data/processed/labels/<partition>/<trip_id>.parquet`: one row per GPS point.

| Column | Meaning |
|---|---|
| `obs_idx` | GPS point index |
| `route_pos` | aligned position along the route |
| `arc_id` | aligned route arc |
| `dist_m` | distance from the point to that arc |
| `is_outlier` | true if `dist_m` is over 50 m (the label is unreliable) |
| `after_gap` | true if the time since the previous point is over 30 s |

Join the two tables on `(trip_id, obs_idx)`; the trip id is the parquet file name for labels.

At inference time use `generate_candidates(TripGeometry.from_trip(trip), trip_id)`. It needs no reference route.

## 4. Rules for training and evaluation

1. **Drop observations with `is_outlier == True`** from training and evaluation, and report how many were dropped (join the label table on `obs_idx`).
2. **Group the cross-validation by trip**: `GroupKFold(5, shuffle=True, random_state=148)` on the 80 train trips. Seed everything with 148.
3. Use the 10 student-test trips for model comparison. **Never** touch the evaluator partition.
4. Fit scalers and other learned preprocessing on training data only, inside a `Pipeline`.
5. About 1 candidate row in 12 is positive, so do not rely on plain accuracy. Report macro F1, log loss and **observation-level Top-1** (the highest-scored candidate per GPS point equals the label arc; a point with no positive counts as wrong).
6. Features must use only GPS and road-network information. Never use `label`, the route, or anything derived from them.

## 5. Features

Grain: **one row per (GPS point, candidate road)**: the candidate table plus 36 numeric features from `pathsnap.features.build_features(geom, cands)`. The ordered list is in [`artifacts/feature_list.json`](artifacts/feature_list.json); build the table with `python scripts/build_dataset.py` (about 2 s; written to `data/processed_local/features/`, gitignored). Everything uses only GPS and road-network geometry in per-trip UTM metres; labels are never read (there is a test for this).

| Group | Features | Why |
|---|---|---|
| distance | `dist_m`, `dist_rank`, `dist_minus_min`, `dist_ratio_min`, `proj_frac`, `proj_outside`, `dist_from_node_m`, `dist_to_node_m`, `dist_over_len` | the point should be close to the true arc, and closer than its competitors |
| direction | `heading_cos` (central difference), `heading_cos_prev`, `heading_cos_next`, `heading_valid`, `heading_cos_gap` | about half the candidate rows are twins (an arc and its reverse) with identical distance; only direction separates them |
| context | `ctx_mean_dist_m`, `ctx_max_dist_m`, `ctx_mean_gap_m` (points t-2..t+2 against the same arc) | one noisy point can sit near the wrong road; its neighbours usually do not |
| motion | `dt_prev_s`, `dt_next_s`, `speed_prev_mps`, `speed_next_mps`, `accel_mps2`, `turn_deg` | speed and turning say what kind of road the vehicle is on and when it is at a junction |
| network | `arc_length_m`, `from_in_deg`, `from_out_deg`, `to_in_deg`, `to_out_deg`, `has_reverse` | road and junction structure |
| sequence | `prev_rule_same`, `prev_rule_adjacent`, `next_rule_same`, `next_rule_adjacent` | agreement with the neighbouring points' best arc under a label-free rule (distance + heading) |
| density | `n_candidates`, `n_within_10m`, `n_within_25m` | ambiguity from parallel roads and dense junctions |

The motion, endpoint-distance, arc-length and previous/next heading-error features come from Advitha's first draft (`pathsnap_features/`, now replaced). They are computed on the shared candidate table, so every model sees the same rows. Class separation of every feature on TRAIN: `experiments/results/feature_summary.csv`.

**Evaluating a model:** call `pathsnap.harness.evaluate_model(name, family, estimator, config)` from `scripts/models/<you>_<model>.py`. It applies the rules in section 4 (same folds for everyone, saved in `experiments/results/cv_folds.csv`) and writes `experiments/results/models/<name>.json`. `python scripts/cv_compare.py` builds the comparison table. Details and current results: [`docs/HANDOFF_FEATURES_HARNESS.md`](docs/HANDOFF_FEATURES_HARNESS.md).

## 6. Where things are

| What | Where |
|---|---|
| Team split and Git rules | [`SPLIT.md`](SPLIT.md) |
| Full requirements and design | [`docs/PLAN.md`](docs/PLAN.md) |
| Data issues, assumptions, decisions with evidence | [`docs/data_quality_log.md`](docs/data_quality_log.md) |
| What the data foundation provides, runtimes | [`docs/HANDOFF_ADYANTH.md`](docs/HANDOFF_ADYANTH.md) |
| Features, harness, model results | [`docs/HANDOFF_FEATURES_HARNESS.md`](docs/HANDOFF_FEATURES_HARNESS.md), `experiments/results/model_comparison.csv` |
| Notebooks: raw to candidate table (00), EDA (01), labels and candidates (02) | `notebooks/` |
| Figures and CSV results | `experiments/figures/`, `experiments/results/` |
| Folium maps | `maps/` |
