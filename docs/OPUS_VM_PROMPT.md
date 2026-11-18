# Task prompt for Claude Code (Opus) on the VM

You are working in the PathSnap hackathon repo (PES University UE24CS352A, team 148). You are doing **Adyanth's part**: the data foundation. Work autonomously and fast. Do not stop to ask questions; make a sensible decision, write it down in `docs/data_quality_log.md`, and keep going. Commit and push often.

## 0. Read first (10 minutes, no more)
- `SPLIT.md` (team split, interfaces, Git rules) and `docs/PLAN.md` (full requirements and design).
- `src/pathsnap/{io,geo,network}.py`, `scripts/inventory.py`, `configs/*.yaml`, `tests/test_foundation.py`.
- `experiments/results/inventory.csv` (already generated; re-run `scripts/inventory.py` if missing).

## 1. Setup
```bash
git checkout feat/adyanth-data        # create from main if it does not exist: git checkout -b feat/adyanth-data
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pyarrow && pip install -e . && pytest
```
Use all CPU cores (`joblib.Parallel(n_jobs=-1)` per trip). Seed everywhere: **148**.

## 2. Ground rules (non-negotiable)
- **Never read, create or commit anything in `data/raw/evaluator/`.** The loader blocks it unless `PATHSNAP_MODE=serve`; do not set that variable and do not work around it.
- **Only real results.** Every number, table and figure must come from a script in this repo that you actually ran. Never invent, estimate or hand-type metrics, curves or plots. If something cannot be computed, say so in the log.
- Never use Euclidean distance on degrees. Use the per-trip UTM projection in `pathsnap.geo.LocalProjector` (trips are in many different cities: UTM zones 14 to 53 appear, so a per-trip CRS is required).
- Don't edit other people's files (`features.py`, `models.py`, `harness.py`, `metrics.py`, `hmm.py`, `api/`, `ui/`). If you need a change there, note it in `docs/HANDOFF_ADYANTH.md`.
- Commit small and often with meaningful messages; push to `origin feat/adyanth-data` after each milestone. Never push to `main`; open a PR at the end.
- No secrets, no big binaries. If a derived file exceeds 90 MB, do not commit it; put it under `data/processed_local/` (gitignored) and commit the script that builds it.
- Record all assumptions and decisions in `docs/data_quality_log.md` as you go (what, why, evidence).

## 3. What we already know about the data (from `inventory.csv`; verify, don't trust blindly)
- `data/raw/train/<8-digit id>/` has 80 trips, `data/raw/student_test/` has 10 trips. Each trip folder holds `<id>.track` (lon lat t, whitespace separated, no header), `.nodes` (lon lat, row = node_id), `.arcs` (from_node to_node, row = arc_id) and `.route` (one arc_id per line).
- **Each trip has its own road network**, and the networks are big: 299 to 155,003 nodes and 649 to 307,392 arcs per trip. Arc ids are local to a trip. Brute-force point-to-all-arcs distance is far too slow for big trips (19k points x 300k arcs); use a spatial index.
- Track lengths: 262 to 19,313 points. About 94% of time steps are exactly 1 s on average, but some trips are as low as 53%. Train has 1,216 gaps over 5 s; the largest are 216,512 s (so gaps must be handled), and trips 90 and 78 have giant gaps.
- No NaN rows, no out-of-range or (0,0) coordinates, timestamps strictly increasing in all 90 trips.
- 546 steps in train exceed 60 m/s (outlier jumps; trip 24 for example). Find and characterise them.
- Networks: self-loops (24), duplicate arcs (1,903 pairs), about 92% of arcs have a reverse arc, largest weakly connected component covers 69% to 100% of nodes.
- Reference route: all ids valid, consecutive arcs are connected in 98.1% to 100% of cases (a few breaks), route length about 100 to 1,440 arcs. The `.route` is an ordered arc list, **not** one label per GPS point.

## 4. Deliverables, in order

### 4.1 EDA (script-driven, reproducible): push as milestone 1
Create `scripts/eda.py` (and optionally a clean `notebooks/01_eda.ipynb` that only calls library code) producing into `experiments/figures/` and `experiments/results/`:
1. Dataset inventory table per partition (trip counts, files, sizes, observation counts). Clearly state the partition for every analysis.
2. Missing/invalid GPS analysis, timestamp ordering and gap analysis (histogram of dt, trips with large gaps), speed distribution, jump outliers.
3. Geographic extent per trip and a world/regional scatter of trip locations; coordinate-order sanity check.
4. Network tables and plots: nodes/arcs per trip, degree distribution, connected components, isolated nodes, duplicate arcs and self-loops.
5. Reference-route structure: route length vs track length, connectivity breaks.
6. `docs/data_quality_log.md`: issues found, assumptions made, decisions.
7. Folium maps via `scripts/make_maps.py` (code in `pathsnap/viz.py`): for at least 5 trips (mix of normal, gappy, noisy, large network, and a student_test trip) with layers: raw GPS points (blue), nearby road network (grey), reference route (green), layer control and legend, sensible extent. Only draw network arcs inside the trip's bounding box plus a margin, otherwise the HTML is huge. Save to `maps/`. Also export one PNG screenshot of the best map into `report/figures/` (Selenium or just document how to screenshot if no browser is available).

### 4.2 Labels: `src/pathsnap/labels.py` (milestone 2)
Implement `align_route_to_observations(...)`: a **monotonic dynamic-programming alignment** from every GPS observation to one arc of that trip's `.route` (route index never moves backwards; cost = point-to-arc distance in metres, plus a small penalty for jumping over route arcs). Handle large time gaps and outlier points gracefully (outliers still receive a label, but flag them). Vectorise the distance matrix per trip (n_obs x route_len is at most about 19k x 1.4k, fine).
- Validate: per-trip stats (median/p95/max distance to the aligned arc, fraction of points farther than 50 m and 100 m, fraction of route arcs that received at least one point). Save to `experiments/results/label_quality.csv`.
- Produce Folium check maps for at least 5 trips showing GPS points coloured by aligned arc plus the route. Fix the algorithm if alignment looks wrong.
- Write down in the log that labels are **derived** (the reference route is not independently verified ground truth).
- Add unit tests on the toy fixture and a hand-built case.

### 4.3 Candidates: `src/pathsnap/candidates.py` (milestone 3)
For each observation, find arcs within radius R metres (at most K nearest) using a spatial index (shapely `STRtree` on projected arc LineStrings, or a KD-tree over densified arc points). Output the **candidate table contract** from `SPLIT.md` section 3 exactly: `trip_id, obs_idx, arc_id, dist_m, proj_frac, label` (label is 1 for the aligned arc, `label` omitted or NaN for unlabeled data). No-candidate fallback: retry once with `fallback_radius_m`; if still none, keep the observation with zero rows and flag it.
- **Choose R and K from data, on training trips only:** sweep R in {15, 25, 40, 60, 80, 120, 200} m and K in {5, 10, 15, 20, 30}, compute the candidate miss rate (reference arc absent from the set) and the mean candidate-set size; save `experiments/results/candidate_miss_rate.csv` and a plot; pick the knee, put it in `configs/candidates.yaml` and justify it in the log.
- `scripts/build_candidates.py` builds tables for train and student_test (parallel over trips) into `data/processed/candidates/<partition>/<trip_id>.parquet` (zstd). If the total is at most about 90 MB commit it, so teammates can start immediately; otherwise write to `data/processed_local/` and document the command and runtime in `README.md`. Also write `experiments/results/candidate_stats.csv` (per trip: observations, mean candidates, miss rate, observations with no candidates).
- **Push this as soon as it exists**: Advitha, Sudhe and Tilak are waiting for it.

### 4.4 Your two models, only after 4.1 to 4.3 are pushed
Logistic Regression and Gaussian Naive Bayes on the candidate table. If `origin/main` already has `src/pathsnap/features.py` and `harness.py` (check `git fetch && git log origin/main`), use them: `scripts/models/adyanth_logreg.py`, `scripts/models/adyanth_gnb.py`, which call the harness. Otherwise prototype with simple features (`dist_m`, distance rank, `dist_m - min dist_m`, `proj_frac`, candidate count), `GroupKFold(5)` by trip on the 80 train trips, seed 148, scaler inside a `Pipeline`, held-out evaluation on the 10 student_test trips, writing `experiments/results/models/<name>.json` with: name, family, config, seed, CV macro F1 per fold, student-test Top-1, macro F1, precision, recall, log loss, candidate miss rate, fit seconds, inference ms per trip. Report honestly, including bad results.

### 4.5 If time remains
`docs/HANDOFF_ADYANTH.md`: what you built, how to run each script, runtimes, open issues, interface notes for the others. Open a PR `feat/adyanth-data` to `main` with a summary of milestones and the key numbers.

## 5. Finish
- `pytest` passes; scripts run from a clean clone with the commands in `README.md`.
- Final message: list what was produced (files and headline numbers from the CSVs), what failed or was skipped, and the chosen R/K.
- Add a line to `docs/AI_USE.md` stating that this part was implemented with Claude Code (Opus) under the team's direction, since AI use must be disclosed. The team still has to read and be able to explain all of it.
