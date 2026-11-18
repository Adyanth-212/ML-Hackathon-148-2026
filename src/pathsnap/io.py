"""Raw-data loaders for PathSnap trips.

Raw file formats (per the dataset description; verify against the real files):

* ``.track``  - whitespace separated, no header: longitude, latitude, timestamp (s)
* ``.nodes``  - whitespace separated, no header: longitude, latitude; row index = node_id
* ``.arcs``   - two integers, no header: from_node_id, to_node_id; row index = arc_id
* ``.route``  - one integer per line: ground-truth arc_id sequence

Leakage guard
-------------
The evaluator partition can only be read when the environment variable
``PATHSNAP_MODE=serve`` is set (the API launcher sets it). Any notebook or
training script that touches evaluator data fails loudly instead of leaking.
Reference routes are never loaded unless ``with_route=True`` is passed.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, repo_path

PARTITIONS = ("train", "student_test", "evaluator")
PROTECTED_PARTITION = "evaluator"
SERVE_ENV = "PATHSNAP_MODE"


class LeakageError(RuntimeError):
    """Raised when development code tries to read evaluator data."""


def _check_access(partition: str) -> None:
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition {partition!r}; expected one of {PARTITIONS}")
    if partition == PROTECTED_PARTITION and os.environ.get(SERVE_ENV) != "serve":
        raise LeakageError(
            "The evaluator partition is reserved for final evaluation. "
            f"It can only be read by the API with {SERVE_ENV}=serve."
        )


def partition_dir(partition: str) -> Path:
    _check_access(partition)
    return repo_path(load_config("data")["partitions"][partition])


@dataclass(frozen=True)
class TripFiles:
    trip_id: str
    partition: str
    track: Path
    nodes: Path | None
    arcs: Path | None
    route: Path | None


def discover_trips(partition: str, root: Path | None = None) -> dict[str, TripFiles]:
    """Find every trip in a partition by locating ``*.track`` files.

    Sibling files with the same stem (``<id>.nodes``, ``<id>.arcs``, ``<id>.route``)
    are attached to the trip. Works whether trips sit in one flat folder or in one
    sub-folder per trip. Missing siblings are reported as ``None`` so the
    inventory can flag them.
    """
    _check_access(partition)
    root = Path(root) if root is not None else partition_dir(partition)
    trips: dict[str, TripFiles] = {}
    for track in sorted(root.rglob("*.track")):
        stem, folder = track.stem, track.parent

        def sibling(ext: str) -> Path | None:
            p = folder / f"{stem}.{ext}"
            return p if p.exists() else None

        if stem in trips:
            raise ValueError(f"Duplicate trip id {stem!r} under {root}")
        trips[stem] = TripFiles(stem, partition, track, sibling("nodes"), sibling("arcs"), sibling("route"))
    return trips


def _read_ws(path: Path, names: list[str], dtype) -> pd.DataFrame:
    """Read a header-less whitespace file, keeping the first ``len(names)`` columns."""
    if path.stat().st_size == 0:
        return pd.DataFrame({n: pd.Series(dtype=dtype) for n in names})
    df = pd.read_csv(path, sep=r"\s+", header=None, comment=None)
    if df.shape[1] < len(names):
        raise ValueError(f"{path.name}: expected >= {len(names)} columns, found {df.shape[1]}")
    if df.shape[1] > len(names):
        warnings.warn(f"{path.name}: {df.shape[1]} columns found, using the first {len(names)}")
    df = df.iloc[:, : len(names)]
    df.columns = names
    return df.astype(dtype)


def read_track(path: Path) -> pd.DataFrame:
    """GPS observations, in file order, unmodified (cleaning is a separate step)."""
    df = _read_ws(path, ["lon", "lat", "t"], float)
    df.index.name = "obs_idx"
    return df


def read_nodes(path: Path) -> pd.DataFrame:
    df = _read_ws(path, ["lon", "lat"], float)
    df.index.name = "node_id"
    return df


def read_arcs(path: Path) -> pd.DataFrame:
    df = _read_ws(path, ["from_node", "to_node"], np.int64)
    df.index.name = "arc_id"
    return df


def read_route(path: Path) -> np.ndarray:
    df = _read_ws(path, ["arc_id"], np.int64)
    return df["arc_id"].to_numpy()


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame
    nodes: pd.DataFrame | None
    arcs: pd.DataFrame | None
    route: np.ndarray | None = None  # only populated when explicitly requested


def load_trip(files: TripFiles, with_route: bool = False) -> Trip:
    """Load one trip. The reference route is only read if ``with_route`` is True."""
    _check_access(files.partition)
    return Trip(
        trip_id=files.trip_id,
        partition=files.partition,
        track=read_track(files.track),
        nodes=read_nodes(files.nodes) if files.nodes else None,
        arcs=read_arcs(files.arcs) if files.arcs else None,
        route=read_route(files.route) if (with_route and files.route) else None,
    )
