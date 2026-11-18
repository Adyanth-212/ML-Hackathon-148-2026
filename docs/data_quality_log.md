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
