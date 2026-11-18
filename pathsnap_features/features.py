"""PathSnap candidate-level feature engineering.

Raw trip files expected:
  track: longitude latitude timestamp
  nodes: longitude latitude (0-based row index = node_id)
  arcs: from_node_id to_node_id (0-based row index = arc_id)
  route: chronological reference arc_id sequence

The route-to-GPS-observation association is intentionally kept separate because
PathSnap's dataset description warns that GPS rows do not necessarily correspond
one-to-one with road-network rows.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import math
import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_008.8

@dataclass
class FeatureConfig:
    n_candidates: int = 8
    max_candidate_distance_m: float = 250.0


def read_track(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["longitude", "latitude", "timestamp"], engine="python")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df[df.longitude.between(-180, 180) & df.latitude.between(-90, 90)]
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def read_nodes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["longitude", "latitude"], engine="python")
    df.index.name = "node_id"
    return df.reset_index()


def read_arcs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["from_node_id", "to_node_id"], engine="python")
    df.index.name = "arc_id"
    return df.reset_index()


def read_route(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0]
    return pd.DataFrame({"route_position": np.arange(len(x)), "arc_id": x.astype(int)})


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.asarray, (lon1, lat1, lon2, lat2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.asarray, (lon1, lat1, lon2, lat2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def angle_diff_deg(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def add_motion_features(track: pd.DataFrame) -> pd.DataFrame:
    out = track.copy()
    plon, plat, pt = out.longitude.shift(1), out.latitude.shift(1), out.timestamp.shift(1)
    nlon, nlat, nt = out.longitude.shift(-1), out.latitude.shift(-1), out.timestamp.shift(-1)
    dtp = (out.timestamp - pt).replace(0, np.nan)
    dtn = (nt - out.timestamp).replace(0, np.nan)
    dp = haversine_m(plon, plat, out.longitude, out.latitude)
    dn = haversine_m(out.longitude, out.latitude, nlon, nlat)
    out["dt_prev_s"] = dtp
    out["dt_next_s"] = dtn
    out["gps_distance_prev_m"] = dp
    out["gps_distance_next_m"] = dn
    out["speed_prev_mps"] = dp / dtp
    out["speed_next_mps"] = dn / dtn
    out["acceleration_mps2"] = out["speed_prev_mps"].diff() / dtp
    out["gps_bearing_prev_deg"] = bearing_deg(plon, plat, out.longitude, out.latitude)
    out["gps_bearing_next_deg"] = bearing_deg(out.longitude, out.latitude, nlon, nlat)
    out["gps_bearing_change_deg"] = angle_diff_deg(
        out["gps_bearing_prev_deg"], out["gps_bearing_next_deg"])
    return out


def build_arc_geometry(nodes: pd.DataFrame, arcs: pd.DataFrame) -> pd.DataFrame:
    n = nodes.set_index("node_id")
    out = arcs.copy()
    out["from_lon"] = out.from_node_id.map(n.longitude)
    out["from_lat"] = out.from_node_id.map(n.latitude)
    out["to_lon"] = out.to_node_id.map(n.longitude)
    out["to_lat"] = out.to_node_id.map(n.latitude)
    out["arc_length_m"] = haversine_m(out.from_lon, out.from_lat, out.to_lon, out.to_lat)
    out["arc_bearing_deg"] = bearing_deg(out.from_lon, out.from_lat, out.to_lon, out.to_lat)
    return out


def _local_xy(lon, lat, lon0, lat0):
    x = np.radians(np.asarray(lon) - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_RADIUS_M
    return x, y


def point_segment_features(lon: float, lat: float, arc: pd.Series) -> dict:
    ax, ay = _local_xy(arc.from_lon, arc.from_lat, lon, lat)
    bx, by = _local_xy(arc.to_lon, arc.to_lat, lon, lat)
    vx, vy = float(bx - ax), float(by - ay)
    denom = vx * vx + vy * vy
    t = 0.0 if denom <= 1e-12 else float(np.clip(-(float(ax) * vx + float(ay) * vy) / denom, 0, 1))
    cx, cy = float(ax) + t * vx, float(ay) + t * vy
    return {
        "distance_to_arc_m": math.hypot(cx, cy),
        "projection_fraction": t,
        "distance_to_from_node_m": math.hypot(float(ax), float(ay)),
        "distance_to_to_node_m": math.hypot(float(bx), float(by)),
    }


def generate_candidates(track: pd.DataFrame, arc_geometry: pd.DataFrame,
                         config: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """Baseline nearest-arc candidate generation; replace/extend with a spatial index if needed."""
    arcs = arc_geometry.dropna(subset=["from_lon", "from_lat", "to_lon", "to_lat"]).copy()
    rows = []
    for obs_idx, obs in track.iterrows():
        endpoint_dist = haversine_m(obs.longitude, obs.latitude,
                                    arcs.from_lon.to_numpy(), arcs.from_lat.to_numpy())
        k = min(max(config.n_candidates * 4, config.n_candidates), len(arcs))
        idx = np.argpartition(endpoint_dist, k - 1)[:k]
        cand = arcs.iloc[idx].copy()
        geom = [point_segment_features(obs.longitude, obs.latitude, r) for _, r in cand.iterrows()]
        cand = pd.concat([cand.reset_index(drop=True), pd.DataFrame(geom)], axis=1)
        cand = cand.sort_values("distance_to_arc_m").head(config.n_candidates).reset_index(drop=True)
        for rank, (_, c) in enumerate(cand.iterrows(), 1):
            if c.distance_to_arc_m <= config.max_candidate_distance_m:
                rows.append({
                    "obs_index": int(obs_idx), "candidate_rank": rank, "arc_id": int(c.arc_id),
                    "distance_to_arc_m": c.distance_to_arc_m,
                    "projection_fraction": c.projection_fraction,
                    "distance_to_from_node_m": c.distance_to_from_node_m,
                    "distance_to_to_node_m": c.distance_to_to_node_m,
                    "arc_length_m": c.arc_length_m,
                    "arc_bearing_deg": c.arc_bearing_deg,
                })
    return pd.DataFrame(rows)


def merge_features(track: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    obs_cols = ["longitude", "latitude", "timestamp", "dt_prev_s", "dt_next_s",
                "gps_distance_prev_m", "gps_distance_next_m", "speed_prev_mps",
                "speed_next_mps", "acceleration_mps2", "gps_bearing_prev_deg",
                "gps_bearing_next_deg", "gps_bearing_change_deg"]
    out = candidates.merge(track[obs_cols], left_on="obs_index", right_index=True, how="left")
    out["heading_error_prev_deg"] = angle_diff_deg(out.gps_bearing_prev_deg, out.arc_bearing_deg)
    out["heading_error_next_deg"] = angle_diff_deg(out.gps_bearing_next_deg, out.arc_bearing_deg)
    out["distance_ratio_to_arc_length"] = out.distance_to_arc_m / out.arc_length_m.replace(0, np.nan)
    out["speed_change_mps"] = out.speed_next_mps - out.speed_prev_mps
    return out.replace([np.inf, -np.inf], np.nan)


def build_features_for_trip(trip_dir: Path, config: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    track = add_motion_features(read_track(trip_dir / "track"))
    nodes = read_nodes(trip_dir / "nodes")
    arcs = read_arcs(trip_dir / "arcs")
    geometry = build_arc_geometry(nodes, arcs)
    candidates = generate_candidates(track, geometry, config)
    out = merge_features(track, candidates)
    out.insert(0, "trip_id", trip_dir.name)
    return out


def build_dataset(dataset_root: Path, trip_dirs: Optional[Iterable[Path]] = None,
                  config: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    if trip_dirs is None:
        trip_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    frames = []
    for trip_dir in trip_dirs:
        required = [trip_dir / "track", trip_dir / "nodes", trip_dir / "arcs"]
        if not all(p.exists() for p in required):
            continue
        try:
            frames.append(build_features_for_trip(trip_dir, config))
        except Exception as exc:
            print(f"[WARN] Skipping {trip_dir.name}: {exc}")
    if not frames:
        raise ValueError("No usable trips found.")
    return pd.concat(frames, ignore_index=True)


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric predictors while excluding identifiers and target-like fields."""
    excluded = {"trip_id", "obs_index", "arc_id", "candidate_rank", "target", "reference_arc_id"}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def save_features(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    DATASET_ROOT = Path("../data/raw")
    OUTPUT = Path("../data/processed/candidate_features.csv")
    cfg = FeatureConfig(n_candidates=8, max_candidate_distance_m=250.0)
    data = build_dataset(DATASET_ROOT, config=cfg)
    save_features(data, OUTPUT)
    print(f"Saved {len(data):,} candidate rows to {OUTPUT}")
