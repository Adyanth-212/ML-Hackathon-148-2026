# Data-quality log

Issues found, assumptions made and decisions taken while preparing the data.
Every number comes from a script in this repo (named next to it). Partitions
used: **TRAIN** (80 trips) and **STUDENT_TEST** (10 trips). The evaluator
partition was never read. Seed everywhere: 148.

## 1. Inventory and file format
Source: `scripts/inventory.py` -> `experiments/results/inventory.csv`, `scripts/eda.py` -> `eda_partition_summary.csv`.

| | train | student_test |
|---|---|---|
| trips | 80 | 10 |
| GPS observations (total / min / median / max) | 208,417 / 361 / 1,529 / 19,313 | 19,160 / 262 / 1,420 / 3,958 |
| nodes per trip (median) | 9,608 | 6,597 |
| arcs per trip (median / max) | 19,527 / 307,392 | 13,145 / 56,027 |
| bytes (track / nodes / arcs) | 6.9 / 38.4 / 44.3 MB | 0.6 / 1.8 / 1.9 MB |
| missing sibling files | 0 | 0 |

- Every trip folder has all four files. Formats confirmed: `.track` = `lon lat t`, `.nodes` = `lon lat` (row = node id), `.arcs` = `from to` (row = arc id), `.route` = one arc id per line.
- **Each trip has its own road network**; arc and node ids are local to the trip. Decision: everything (geometry, candidates, labels) is computed per trip; there is no global arc vocabulary, so the ML task is per-(observation, candidate) binary classification.

## 2. GPS quality
Source: `inventory.csv`, `eda_gaps.csv`, `eda_speed_outliers.csv`, figures `eda_dt_hist.png`, `eda_speed_hist.png`, `eda_anomalous_trips.png`.

- **No NaN rows, no out-of-range coordinates, no (0,0) points, no duplicate or backwards timestamps** in any of the 90 trips. Decision: no row deletion is needed; tracks are used in file order.
- **Sampling:** 96.5% of train steps (98.6% student_test) are exactly 1 s; per-trip share of 1 s steps ranges from 53% to 100% (median 99.1%).
- **Gaps:** 1,216 train steps (44 student_test) are longer than 5 s. 135 are longer than 30 s, 70 longer than 60 s and 8 longer than 10 min. The largest are trip 00000090 (153,064 s, about 42 h, between two points 3.3 m apart) and trip 00000078 (52,810 s between obs 1 and 2, plus 123 more gaps). The task prompt quoted "216,512 s" as the largest gap; that figure is the *sum* of per-trip maxima, the true maximum is 153,064 s. Long gaps almost always occur while the vehicle is parked (jump distance a few metres), so they are breaks in time, not in space.
  Decision: keep all points; labelling and candidate generation do not depend on dt; the alignment relaxes its skip penalty across gaps > 30 s (section 5) and the HMM should break / relax transitions there (Sudhe, `configs/hmm.yaml: gaps.break_seconds`).
- **Speed outliers:** 546 train steps (2 student_test) exceed 60 m/s (216 km/h) by haversine distance / dt. They are concentrated: trip 00000032 (358), 00000013 (103), 00000001 (52); every other trip has at most 6. None of them is an isolated spike (the point two steps later is never closer than half the jump), and in 00000032 97% of them have neighbours that are also > 40 m/s. Trip 00000079 has two steps at 157-180 m/s.
  Interpretation: these are mostly **sustained very fast segments** (motorway driving, possibly a timestamp-scaling issue in those trips), not single-point glitches. Decision: no points are removed; the label aligner flags per-observation outliers by distance to the aligned arc instead (section 5), and motion features should be clipped or robust (note for Advitha).
- **Stationary jitter:** some tracks have GPS path length far above the route length (route/GPS ratio 0.13 for 00000033, 0.14 for 00000002, 0.19 for 00000054; median over trips 0.85). These are dense 1 Hz tracks where noise accumulates path length; GPS path length is therefore not a reliable trip-length measure.

## 3. Coordinates and projection
Source: `eda_coord_order.csv`, `eda_trip_locations.png`.

- Coordinate order is `lon lat`: all latitudes are within +/-90, every track lies inside its network's bounding box (100% of points in all 90 trips), and the median distance from a GPS point to its nearest network node is 7.7-92.9 m (median 16.6 m) after projection. Swapping the columns would put points thousands of km away.
- Trips are spread worldwide: **25 different UTM zones** (EPSG 32614-32653 north, 32721/32722/32735 south). Decision: each trip is projected into the UTM zone of its median GPS point (`pathsnap.geo.LocalProjector`, wrapped by `TripGeometry`); all distances are planar metres in that CRS. Euclidean distance on degrees is never used. Haversine is used only for per-step speeds in EDA.

## 4. Road networks
Source: `eda_network.csv`, `inventory.csv`, figure `eda_network.png`.

- Size: 299 to 155,003 nodes and 649 to 307,392 arcs per trip. Decision: candidate search uses a spatial index (shapely `STRtree` on projected arc segments), never brute force.
- 24 self-loops and 1,903 duplicate arcs (same from/to pair) in total; 106 arcs have zero length (< 1 cm). Kept as they are (removing them would renumber arc ids). Duplicate arcs are geometrically identical, so the classifier cannot separate them; they are a source of irreducible label ambiguity (section 5).
- About 92% of arcs have a reverse arc (median per trip 95%, min 35%): two-way streets are two opposite arcs with **identical geometry**. Decision: the candidate table keeps both directions; heading-vs-bearing features are needed to separate them (note for Advitha).
- No isolated nodes; 1,907 dead-end nodes in total. Largest weakly connected component covers 69% to 100% of nodes (up to 421 weak components), largest strongly connected component 64% to 100%. The HMM must handle disconnected candidates.
- Arcs are straight segments between two nodes (there are no shape points); median arc length about 25 m, longest 9.4 km.

## 5. Reference routes
Source: `eda_route.csv`, figure `eda_route.png`.

- All route arc ids are valid. Route length 59 to 2,266 arcs.
- 14 breaks in total (consecutive route arcs not sharing a node) in 11 trips; the gap between the end of one arc and the start of the next is 10.6 to 597.6 m. Decision: kept; the aligner allows skipping and the breaks are reported, not repaired.
- 10 trips revisit an arc (up to 255 repeated arc entries in one trip); the route can loop, so the alignment must follow the route **order**, not just arc membership.
- The `.route` is an ordered arc list, **not** one label per GPS point; per-observation labels must be derived (milestone 2).

## 6. Maps
`scripts/make_maps.py` -> `maps/eda_<partition>_<trip>.html` for six trips: normal (00000072), gappy (00000078), fast/outliers (00000032), largest network (00000013), low route/GPS ratio (00000033) and student_test (00000056). Layers: raw GPS (blue), road network within 200 m of the track (grey), reference route (green), layer control and legend.
- Decision: only arcs with an end node within 200 m of the track are drawn. A bounding-box filter produced 8-15 MB files for long diagonal trips; the corridor keeps every map under 3 MB.
- Basemap: Esri Light Gray Canvas (no API key, works from `file://`). The OSM tile server returns 403 for pages opened from disk and CARTO now requires a key. Tiles need internet; the data layers do not.
- Screenshots: `report/figures/map_eda_train_00000033.png`, `map_eda_train_00000072.png` (Firefox headless `--screenshot`; basemap tiles did not load in time for the second one).

## 7. Derived per-observation labels (milestone 2)
Source: `src/pathsnap/labels.py`, `scripts/build_labels.py` -> `data/processed/labels/<partition>/<trip>.parquet`,
`experiments/results/label_quality.csv`, `experiments/figures/label_quality.png`, check maps `maps/labels_*.html` (`scripts/make_maps.py --labels`).

**The labels are derived, not observed.** The reference route is an ordered arc list supplied with the data; it is not independently verified ground truth, and assigning a GPS point to one of its arcs is our own modelling step. Every metric that uses these labels (training, Top-1, miss rate) inherits that uncertainty, and the report must say so.

Method: monotonic DP alignment (DTW-like). State = position in the route; cost per point = distance to that route arc in metres, capped at 100 m, plus up to 10 m when the GPS heading (central difference, only if the point moved >= 5 m) is opposite to the arc direction; moving forward by more than one route arc costs 10 m per skipped arc, free across time gaps > 30 s; skipping route arcs before the first / after the last point costs 0.5 m per arc. The route index never decreases. O(n_obs x route_len) per trip, all 90 trips in ~2 s on 8 cores. Verified against brute force on random small cases (`tests/test_labels.py`).
- Why heading: about 92% of arcs have a geometrically identical reverse twin, and routes that drive out and back along the same street contain both. Distance alone cannot tell which leg a point belongs to.
- Why the cap: a single far-off point must still get a label but must not pull its neighbours to the wrong part of the route.

Quality (all 90 trips): median distance to the aligned arc 4.75 m (median of trip medians; range 3.5-140 m), p95 14.6 m (median over trips). Route coverage (share of route arcs that received at least one point) median 0.99, minimum 0.81: very short arcs passed at speed get no 1 Hz sample.
- Points > 50 m from their aligned arc are flagged `is_outlier` but keep their label. Train: 23,165 of 208,417 points (11.1%), student_test: 27 of 19,160 (0.14%). **They are concentrated in five trips**: 00000054 (89%), 00000033 (59%), 00000002 (12%), 00000066 (9%), 00000043 (7%); 85 of 90 trips have < 5%.
- Cause, checked on the maps and numerically: in 00000054, 00000033 and 00000002 those points are > 50 m from *every* route arc (87%, 59%, 12% of points), i.e. the **reference route covers only part of the GPS track**. This is a limitation of the reference data, not an alignment failure; the labels for those points are meaningless.
- Decision: labels stay as they are (contract), and the per-observation flag lives in `data/processed/labels/<partition>/<trip>.parquet` (`obs_idx, route_pos, arc_id, dist_m, is_outlier, after_gap`). **Recommendation for training and evaluation:** drop observations with `is_outlier == True` (join on `trip_id, obs_idx`), and report how many were dropped. Student-test is barely affected (0.14%).
- Per trip, a median of 94% of non-outlier points (mean 91%) are aligned to the geometrically nearest route arc; where they are not, the aligned arc is usually within 0.5 m of the nearest one (median 98% of points per trip), i.e. ties between twin arcs or at corners resolved by route order and heading.

## 8. Candidate generation and the choice of R and K (milestone 3)
Source: `src/pathsnap/candidates.py`, `scripts/candidate_sweep.py` -> `experiments/results/candidate_miss_rate.csv`, `experiments/figures/candidate_miss_rate.png`; `scripts/build_candidates.py` -> `data/processed/candidates/<partition>/<trip>.parquet`, `experiments/results/candidate_stats.csv`.

Method: shapely `STRtree` over each trip's projected arc segments; `dwithin` query, exact point-to-segment distance, rows ordered by `(dist_m, arc_id)` per observation, keep the first K. No candidate within R: one retry at the fallback radius; still nothing: zero rows, counted in `n_no_candidates`.

**R and K were chosen on TRAIN only.** The miss rate is the share of observations whose aligned arc is not in the candidate set. It was computed on all train observations and on those with reliable labels (`is_outlier == False`, section 7); the choice uses the reliable subset because the outlier labels are not meaningful (they are > 50 m from the route by definition).

Reliable train observations, miss rate (%) / mean candidates:

| R \ K | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|
| 25 m | 3.96 / 3.4 | 2.60 / 4.4 | 2.45 / 4.7 | 2.39 / 4.8 | 2.35 / 4.9 |
| 40 m | 2.54 / 4.0 | 0.98 / 5.8 | 0.77 / 6.7 | 0.64 / 7.2 | 0.59 / 7.5 |
| 60 m | 2.33 / 4.3 | 0.73 / 7.0 | 0.51 / 8.7 | 0.29 / 9.8 | **0.03 / 11.1** |
| 80 m | 2.33 / 4.5 | 0.73 / 7.7 | 0.51 / 10.0 | 0.29 / 11.7 | 0.03 / 14.1 |

- Radius: the miss rate stops falling at R = 60 m for every K (no reliable label is beyond 50 m by construction, and GPS error is ~5 m median, ~15 m p95), while every extra 20 m adds 1.5-3 candidates per point. R = 60 m is the knee.
- K: K = 5 misses 2.3% even at large R, because two-way streets and junctions put several arcs at almost the same distance (each road appears twice, with identical geometry). Going from K = 20 to K = 30 cuts the miss rate tenfold (0.29% -> 0.03%) for +1.3 candidates per point; beyond that there is nothing left to gain. K = 30 chosen.
- Fallback radius 150 m: on all train points, 8.4% have no arc within 60 m, but only 0.12% have none within 120 m. These are almost all in the trips where the GPS leaves the area the network was cut for (section 7).
- Final: **R = 60 m, K = 30, fallback 150 m** (`configs/candidates.yaml`).

Result (`candidate_stats.csv`):

| | train | student_test |
|---|---|---|
| rows | 2,449,078 | 244,176 |
| mean candidates / point | 11.75 | 12.74 |
| miss rate, all points | 6.46% | 0.13% |
| miss rate, reliable labels | 0.029% | 0.000% |
| points rescued by the fallback | 17,294 | 0 |
| points with no candidate | 198 | 0 |
| parquet size (zstd) | 26.6 MB | 2.7 MB |

The student_test row is a check only; it played no part in the choice. The candidate tables are small enough to commit, so teammates can use them straight away. Build time: about 1-2 s for all 90 trips on 8 cores (labels + candidates).

Notes for the feature/model owners:
- About half the rows are twins (an arc and its reverse) with identical `dist_m`, and `proj_frac` mirrored (t vs 1 - t). Only direction features (GPS heading vs arc bearing) can separate them.
- Each observation has at most one `label == 1`. Observations without a positive row are the misses above; for training, keep or drop them consistently, and drop `is_outlier` observations (join `data/processed/labels/...` on `trip_id, obs_idx`).
- `proj_frac` is unclipped: < 0 or > 1 means the nearest point is an arc endpoint.

## 9. Baseline pipeline (rule, not ML)
Source: `src/pathsnap/pipeline.py`, `scripts/eval_baseline.py` -> `experiments/results/baseline_heading_sweep.csv`, `baseline_student_test.csv`.

`predict_trip(trip, artifacts)` works end to end without a model: it projects the trip, builds candidates (R = 60 m, K = 30), scores each candidate as `-(dist_m + w * (1 - cos(heading, arc direction)) / 2)`, takes the argmax per observation, gives observations without candidates the arc of the nearest-in-time observation, and returns the SPLIT.md prediction contract. A trained classifier (`artifacts["scorer"]`) and the HMM (`artifacts["decoder"]`) plug in without code changes.

Heading weight w chosen on TRAIN (reliable-label Top-1):

| w (m) | 0 (nearest arc) | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| train Top-1 | 0.548 | 0.797 | 0.808 | 0.819 | **0.822** | 0.813 |
| student_test Top-1 | 0.530 | 0.813 | 0.821 | 0.832 | 0.832 | 0.827 |

w = 20 m. Student_test via the real `predict_trip` path: Top-1 0.832, route continuity (arc changes that share a node) 0.73 on average, about 21 ms per trip (stage timings in the CSV).
- Caveat: the label aligner also uses a heading term (section 7). It only decides between route arcs, mostly a street's two directions when the route uses both, so it does not make the baseline right by construction. The jump from 0.53 to 0.83 comes mostly from points on one-pass two-way streets, where the label is fixed by route order alone.
- Continuity is low (0.73) because a per-point argmax flips between parallel and adjacent arcs. Fixing that is exactly the HMM's job.

## 10. Trained classifier in the pipeline, and what its errors say about the labels
Source: `scripts/export_artifacts.py` -> `artifacts/model.joblib`, `artifacts/metadata.json`; `scripts/eval_pipeline.py` -> `experiments/results/pipeline_student_test.csv`, `pipeline_summary.csv`, `pipeline_error_types.csv`; maps `maps/predicted_student_test_*.html` (`scripts/make_maps.py --predicted`).

- **Selection rule:** highest cross-validated observation-level Top-1 on TRAIN. LightGBM 0.8639, XGBoost 0.8632, HistGB 0.8627, LogReg 0.8534, SGD 0.8532, GNB 0.7881. The top three are within noise; LightGBM is exported. Student-test numbers were not used for the choice. Bayesian tuning may replace it (re-run the export with the tuned model's name).
- **End-to-end on student_test** (`predict_trip`, the API code path, trip loaded without its route): Top-1 **0.879** (16,813 of 19,133 reliable points), the same as the harness result. Route continuity of the per-point argmax 0.867 with 646 breaks, against 0.832 / 0.728 / 1,708 breaks for the rule. Mean 77 ms per trip, worst 189 ms, including the UI network overlay (feature building + LightGBM ~54 ms).
- **Error types** (2,320 wrong points): adjacent arc sharing a node 997 (43%, the boundary between consecutive arcs at junctions), reverse twin 971 (42%), other 351 (15%), duplicate arc 1.
- **Direction noise in the reference routes.** For reverse-twin errors where the GPS heading is trustworthy (712 points), the **predicted** arc points the way the vehicle is moving in 577 cases (81%) and the **label** arc in only 135. Trip 00000077 holds most of them (548). The reference route there uses the opposite-direction arc of a two-way street for long stretches. Many of these "errors" are therefore label errors, not model errors. This is another reason not to overstate Top-1 as ground truth, and it gives the HMM a ceiling.
- Adjacent-arc errors at junctions are the kind the HMM's transition model is designed to fix: the per-point argmax switches arcs a few points early or late.
