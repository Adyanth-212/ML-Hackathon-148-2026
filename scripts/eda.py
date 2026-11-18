"""EDA (milestone 1): reproducible tables and figures for TRAIN and STUDENT_TEST.

Never touches the evaluator partition. Every output is written by this script:

    python scripts/eda.py            # ~1-2 min on 8 cores

Outputs
-------
experiments/results/eda_partition_summary.csv   per-partition inventory
experiments/results/eda_gaps.csv                every time step > 5 s (trip, index, dt, jump distance)
experiments/results/eda_speed_outliers.csv      every step > 60 m/s with context
experiments/results/eda_network.csv             per-trip network table (incl. degree stats)
experiments/results/eda_route.csv               per-trip route structure incl. break positions
experiments/results/eda_coord_order.csv         lon/lat order sanity check per trip
experiments/figures/eda_*.png                   plots
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap.config import repo_path
from pathsnap.geo import TripGeometry, haversine_m
from pathsnap.io import discover_trips, load_trip

DEV_PARTITIONS = ("train", "student_test")
RES = repo_path("experiments/results")
FIG = repo_path("experiments/figures")
SPEED_JUMP_MPS = 60.0
GAP_S = 5.0


def analyse_trip(files) -> dict:
    trip = load_trip(files, with_route=True)
    tr, nodes, arcs, route = trip.track, trip.nodes, trip.arcs, trip.route
    lon, lat, t = tr["lon"].to_numpy(), tr["lat"].to_numpy(), tr["t"].to_numpy()
    dt = np.diff(t)
    step = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    speed = step / np.where(dt > 0, dt, np.nan)
    tid, part = trip.trip_id, trip.partition

    gaps = [dict(partition=part, trip_id=tid, obs_idx=int(i), dt_s=float(dt[i]), jump_m=float(step[i]),
                 implied_speed_mps=float(speed[i])) for i in np.flatnonzero(dt > GAP_S)]

    out_rows = []
    for i in np.flatnonzero(speed > SPEED_JUMP_MPS):
        # is it an isolated spike (A->B->C with B off-line, A-C short) or a sustained high speed?
        prev_v = speed[i - 1] if i > 0 else np.nan
        next_v = speed[i + 1] if i + 1 < len(speed) else np.nan
        skip = haversine_m(lon[i], lat[i], lon[i + 2], lat[i + 2]) if i + 2 < len(lon) else np.nan
        out_rows.append(dict(partition=part, trip_id=tid, obs_idx=int(i), dt_s=float(dt[i]), jump_m=float(step[i]),
                             speed_mps=float(speed[i]), prev_speed_mps=float(prev_v), next_speed_mps=float(next_v),
                             dist_i_to_i2_m=float(skip)))

    # network: degree distribution and geometry
    n = len(nodes)
    f, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    indeg = np.bincount(to, minlength=n)
    outdeg = np.bincount(f, minlength=n)
    geom = TripGeometry(tr, nodes, arcs)
    alen = geom.arc_lengths()
    deg_hist = np.bincount(np.minimum(indeg + outdeg, 12), minlength=13)
    net = dict(partition=part, trip_id=tid, n_nodes=n, n_arcs=len(arcs),
               isolated_nodes=int(((indeg + outdeg) == 0).sum()),
               deg_mean=float((indeg + outdeg).mean()),
               dead_end_nodes=int(((indeg + outdeg) > 0).sum() - ((indeg > 0) & (outdeg > 0)).sum()),
               arc_len_median_m=float(np.median(alen)), arc_len_p95_m=float(np.percentile(alen, 95)),
               arc_len_max_m=float(alen.max()), zero_len_arcs=int((alen < 0.01).sum()),
               net_lon_span_deg=float(nodes.lon.max() - nodes.lon.min()),
               net_lat_span_deg=float(nodes.lat.max() - nodes.lat.min()),
               track_inside_net_bbox_frac=float(np.mean((lon >= nodes.lon.min()) & (lon <= nodes.lon.max())
                                                        & (lat >= nodes.lat.min()) & (lat <= nodes.lat.max()))))

    # route: breaks between consecutive arcs, length vs track
    br = np.flatnonzero(arcs["to_node"].to_numpy()[route[:-1]] != arcs["from_node"].to_numpy()[route[1:]])
    rlen = alen[route]
    gap_m = [float(np.hypot(geom.node_x[f[route[i + 1]]] - geom.node_x[to[route[i]]],
                            geom.node_y[f[route[i + 1]]] - geom.node_y[to[route[i]]])) for i in br]
    rt = dict(partition=part, trip_id=tid, n_obs=len(tr), route_len=len(route),
              route_length_m=float(rlen.sum()), track_path_m=float(np.nansum(step)),
              n_breaks=len(br), break_positions=";".join(map(str, br.tolist())),
              break_gap_m=";".join(f"{g:.1f}" for g in gap_m),
              route_revisits=int(len(route) - len(np.unique(route))))

    # coordinate order: the projected track must sit on the network and lat must be within +/-90
    co = dict(partition=part, trip_id=tid, lon_med=float(np.median(lon)), lat_med=float(np.median(lat)),
              lat_abs_le_90=bool(np.all(np.abs(lat) <= 90)), utm_epsg=geom.proj.epsg,
              median_nearest_node_m=float(_median_nearest_node(geom)))

    return dict(gaps=gaps, jumps=out_rows, net=net, route=rt, coord=co, deg_hist=deg_hist,
                dt=dt, speed=speed[np.isfinite(speed)], lon=lon[:: max(1, len(lon) // 200)],
                lat=lat[:: max(1, len(lat) // 200)])


def _median_nearest_node(geom: TripGeometry) -> float:
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.c_[geom.node_x, geom.node_y]).query(np.c_[geom.obs_x, geom.obs_y][::10])
    return float(np.median(d))


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    jobs = [f for part in DEV_PARTITIONS for f in discover_trips(part).values()]
    print(f"Analysing {len(jobs)} trips on all cores ...")
    res = Parallel(n_jobs=-1)(delayed(analyse_trip)(f) for f in jobs)
    parts = [f.partition for f in jobs]

    inv = pd.read_csv(RES / "inventory.csv", dtype={"trip_id": str})
    summ = inv.groupby("partition").agg(
        trips=("trip_id", "count"), obs_total=("n_obs", "sum"), obs_min=("n_obs", "min"),
        obs_median=("n_obs", "median"), obs_max=("n_obs", "max"),
        track_mb=("track_bytes", lambda s: s.sum() / 1e6), nodes_mb=("nodes_bytes", lambda s: s.sum() / 1e6),
        arcs_mb=("arcs_bytes", lambda s: s.sum() / 1e6), route_kb=("route_bytes", lambda s: s.sum() / 1e3),
        missing_files=("has_route", lambda s: int((~s).sum())),
        nan_rows=("n_nan_rows", "sum"), out_of_range=("n_out_of_range", "sum"), zero_coords=("n_zero_coords", "sum"),
        dup_timestamps=("n_dup_timestamps", "sum"), backwards_time=("n_backwards_time", "sum"),
        gaps_gt_5s=("n_gaps_gt_5s", "sum"), max_gap_s=("dt_max_s", "max"), jumps_gt_60mps=("n_jumps_gt_60mps", "sum"),
        nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"), arcs_max=("n_arcs", "max"),
    ).reset_index()
    summ.to_csv(RES / "eda_partition_summary.csv", index=False)
    print(summ.T.to_string())

    gaps = pd.DataFrame([g for r in res for g in r["gaps"]])
    jumps = pd.DataFrame([j for r in res for j in r["jumps"]])
    net = pd.DataFrame([r["net"] for r in res]).merge(
        inv[["trip_id", "self_loops", "duplicate_arcs", "frac_arcs_with_reverse", "n_weak_components",
             "largest_wcc_frac", "n_strong_components", "largest_scc_frac"]], on="trip_id")
    route = pd.DataFrame([r["route"] for r in res])
    coord = pd.DataFrame([r["coord"] for r in res])
    gaps.to_csv(RES / "eda_gaps.csv", index=False)
    jumps.to_csv(RES / "eda_speed_outliers.csv", index=False)
    net.to_csv(RES / "eda_network.csv", index=False)
    route.to_csv(RES / "eda_route.csv", index=False)
    coord.to_csv(RES / "eda_coord_order.csv", index=False)

    # ---- figures (partition stated in every title) ----
    tr = [r for r, p in zip(res, parts) if p == "train"]
    dt = np.concatenate([r["dt"] for r in tr])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    vals, cnt = np.unique(np.minimum(dt, 11), return_counts=True)
    ax[0].bar([str(int(v)) if v < 11 else ">10" for v in vals], cnt / cnt.sum())
    ax[0].set(title="TRAIN: time step dt (s)", xlabel="dt (s)", ylabel="fraction of steps")
    big = dt[dt > GAP_S]
    ax[1].hist(np.log10(big), bins=40)
    ax[1].set(title=f"TRAIN: gaps > {GAP_S:.0f} s (n={len(big)})", xlabel="log10(dt seconds)", ylabel="count")
    fig.tight_layout(); fig.savefig(FIG / "eda_dt_hist.png", dpi=130); plt.close(fig)

    spd = np.concatenate([r["speed"] for r in tr])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(spd, 0, 100), bins=100, log=True)
    ax.axvline(SPEED_JUMP_MPS, color="r", ls="--", label=f"{SPEED_JUMP_MPS:.0f} m/s outlier threshold")
    ax.set(title="TRAIN: step speed (haversine / dt), clipped at 100 m/s", xlabel="m/s", ylabel="steps (log)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "eda_speed_hist.png", dpi=130); plt.close(fig)

    jt = jumps[jumps.partition == "train"].groupby("trip_id").size().sort_values(ascending=False)
    gt = gaps[gaps.partition == "train"].groupby("trip_id").dt_s.max().sort_values(ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    jt.head(15).plot.bar(ax=ax[0]); ax[0].set(title="TRAIN: steps > 60 m/s per trip (top 15)", ylabel="count")
    np.log10(gt.head(15)).plot.bar(ax=ax[1]); ax[1].set(title="TRAIN: largest gap per trip (top 15)", ylabel="log10(s)")
    fig.tight_layout(); fig.savefig(FIG / "eda_anomalous_trips.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for p, c in (("train", "#1f77b4"), ("student_test", "#d62728")):
        sub = coord[coord.partition == p]
        ax.scatter(sub.lon_med, sub.lat_med, s=25, c=c, label=f"{p} ({len(sub)})", alpha=.8)
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="longitude", ylabel="latitude",
           title="Trip locations (median GPS point), TRAIN + STUDENT_TEST")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "eda_trip_locations.png", dpi=130); plt.close(fig)

    # per-trip trajectories (small multiples, first 16 train trips), in metres is not needed here
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for a, r, f in zip(axs.flat, tr, [f for f in jobs if f.partition == "train"]):
        a.plot(r["lon"], r["lat"], lw=.8); a.set_title(f.trip_id, fontsize=8); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("TRAIN: trajectory shapes (lon/lat, first 16 trips)"); fig.tight_layout()
    fig.savefig(FIG / "eda_trajectories.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for p, c in (("train", "#1f77b4"), ("student_test", "#d62728")):
        sub = net[net.partition == p]
        ax[0].scatter(sub.n_nodes, sub.n_arcs, c=c, s=18, label=p)
        ax[1].scatter(sub.n_nodes, sub.largest_wcc_frac, c=c, s=18, label=p)
    ax[0].set(xscale="log", yscale="log", xlabel="nodes", ylabel="arcs", title="Network size per trip")
    ax[1].set(xscale="log", xlabel="nodes", ylabel="largest WCC / nodes", title="Weak connectivity per trip")
    dh = np.sum([r["deg_hist"] for r in tr], axis=0)
    ax[2].bar([str(i) if i < 12 else "12+" for i in range(13)], dh / dh.sum())
    ax[2].set(title="TRAIN: node degree (in+out) distribution", xlabel="degree", ylabel="fraction of nodes")
    ax[0].legend(); fig.tight_layout(); fig.savefig(FIG / "eda_network.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for p, c in (("train", "#1f77b4"), ("student_test", "#d62728")):
        sub = route[route.partition == p]
        ax[0].scatter(sub.n_obs, sub.route_len, c=c, s=18, label=p)
        ax[1].scatter(sub.track_path_m / 1000, sub.route_length_m / 1000, c=c, s=18, label=p)
    ax[0].set(xlabel="GPS observations", ylabel="route arcs", title="Route length vs track length", xscale="log", yscale="log")
    m = route[["track_path_m", "route_length_m"]].max().max() / 1000
    ax[1].plot([0, m], [0, m], "k--", lw=.8)
    ax[1].set(xlabel="GPS path length (km)", ylabel="route length (km, straight arcs)", title="Route vs GPS distance")
    ax[0].legend(); fig.tight_layout(); fig.savefig(FIG / "eda_route.png", dpi=130); plt.close(fig)

    print(f"\ngaps>5s: {len(gaps)} rows; jumps>60m/s: {len(jumps)} rows; route breaks: {route.n_breaks.sum()}")
    print("Jumps per trip (train):", jt.head(8).to_dict())
    print("Coord check: median distance GPS->nearest node (m):", coord.median_nearest_node_m.describe().round(1).to_dict())
    print("Track inside network bbox:", net.track_inside_net_bbox_frac.describe().round(3).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
