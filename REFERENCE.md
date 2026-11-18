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

Grain: **one row per (GPS point, candidate road)**. The columns `dist_m` and `proj_frac` come straight from the candidate table.

### 5.1 Features currently in the shared code (`scripts/models/_adyanth_proto.py`, used for the prototype models)

| Feature | Meaning |
|---|---|
| `dist_m` | distance from the point to the arc (m) |
| `dist_rank` | rank of this candidate among the point's candidates by distance (0 = nearest) |
| `dist_minus_min` | `dist_m` minus the nearest candidate's distance for the same point |
| `proj_frac` | projection fraction along the arc (unclipped) |
| `proj_outside` | 1 if `proj_frac` is below 0 or above 1 |
| `n_candidates` | how many candidates the point has |
| `heading_cos` | cosine of the angle between the GPS heading and the arc direction (0 if the heading is unknown) |
| `heading_valid` | 1 if the point moved at least 5 m between its neighbours, so the heading is trusted |
| `heading_cos_gap` | the best `heading_cos` among this point's candidates minus this candidate's |

**Why direction matters:** about half the candidate rows are twins (an arc and its reverse) with identical `dist_m` and mirrored `proj_frac`. Only direction features can tell them apart. In the prototype results, adding the three heading features raised Logistic Regression student-test Top-1 from 0.54 to 0.85 (`experiments/results/models/`, `docs/HANDOFF_ADYANTH.md`).

### 5.2 Advitha's feature module (branch `feat/advitha`, folder `pathsnap_features/`, not merged yet)

Per GPS point (motion): `dt_prev_s`, `dt_next_s`, `gps_distance_prev_m`, `gps_distance_next_m`, `speed_prev_mps`, `speed_next_mps`, `acceleration_mps2`, `gps_bearing_prev_deg`, `gps_bearing_next_deg`, `gps_bearing_change_deg`.
Per candidate: `distance_to_arc_m`, `projection_fraction`, `distance_to_from_node_m`, `distance_to_to_node_m`, `arc_length_m`, `arc_bearing_deg`, `candidate_rank`.
Derived: `heading_error_prev_deg`, `heading_error_next_deg`, `distance_ratio_to_arc_length`, `speed_change_mps`.

### 5.3 To reconcile before the shared model harness

The module in 5.2 was written against a different data layout and differs from the shared preprocessing. Before everyone uses one feature table, align these points:

- **Files:** it opens `track`, `nodes`, `arcs` without the id prefix and extension. In this repo the files are `<id>.track` and so on, inside one folder per trip.
- **Candidates:** it builds its own (nearest 8, within 250 m, from a brute-force search) instead of the frozen table (R = 60 m, K = 30). Features should be computed on the shared candidate table so every model sees the same rows.
- **Projection:** it uses a local approximation around each point instead of the per-trip UTM projection.
- **Rows:** it sorts the track by time and drops rows that fail its checks. Shared preprocessing deletes nothing, because `obs_idx` must stay equal to the row number for the label join.
- **Labels:** it creates none. Join the label table (and drop `is_outlier` points) so the model has a target.

The motion and heading features in 5.2 are good additions; the suggestion is to compute them on top of the shared candidate table and keep the single feature list in `artifacts/feature_list.json`.

## 6. Where things are

| What | Where |
|---|---|
| Team split and Git rules | [`SPLIT.md`](SPLIT.md) |
| Full requirements and design | [`docs/PLAN.md`](docs/PLAN.md) |
| Data issues, assumptions, decisions with evidence | [`docs/data_quality_log.md`](docs/data_quality_log.md) |
| What the data foundation provides, runtimes | [`docs/HANDOFF_ADYANTH.md`](docs/HANDOFF_ADYANTH.md) |
| Notebooks: raw to candidate table (00), EDA (01), labels and candidates (02) | `notebooks/` |
| Figures and CSV results | `experiments/figures/`, `experiments/results/` |
| Folium maps | `maps/` |
