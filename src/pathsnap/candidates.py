"""Candidate generation (TODO, phase P2).

For each GPS observation return the arcs within ``radius_m`` (at most
``max_candidates`` nearest), using a spatial index such as ``shapely.STRtree``.
Choose radius/K from the candidate-miss-rate curve on TRAIN trips.
"""
from __future__ import annotations

import pandas as pd


def generate_candidates(trip, radius_m: float, max_candidates: int) -> pd.DataFrame:
    """One row per (obs_idx, arc_id) pair with the basic geometry (distance, t)."""
    raise NotImplementedError
