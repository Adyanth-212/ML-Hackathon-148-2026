"""Feature engineering for (observation, candidate arc) pairs (TODO, phase P4).

Use only GPS + road-network information, never labels. Planned groups:
distance, direction (GPS heading vs arc bearing), context (neighbouring points'
distance to the same arc), motion, network topology and candidate density.
Keep the final column list in artifacts/feature_list.json.
"""
from __future__ import annotations

import pandas as pd


def build_features(trip, candidates: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError
