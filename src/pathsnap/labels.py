"""Per-observation training labels (TODO, phase P2).

The reference ``.route`` is an ordered list of arcs, not one label per GPS
point. Plan: align each observation to one route arc with a monotonic dynamic
programme (route index never decreases; cost = point-to-arc distance), then
check the alignment visually on Folium maps and log far-away matches as label
noise. Used for TRAIN and STUDENT_TEST trips only.
"""
from __future__ import annotations

import numpy as np


def align_route_to_observations(obs_xy: np.ndarray, route: np.ndarray, arc_geom) -> np.ndarray:
    """Return the aligned arc_id for every observation (length = n_obs)."""
    raise NotImplementedError
