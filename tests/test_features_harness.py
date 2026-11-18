"""Features and harness helpers on the toy fixture."""
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.candidates import attach_labels, generate_candidates
from pathsnap.features import FEATURE_COLUMNS, FEATURE_GROUPS, build_features
from pathsnap.geo import TripGeometry
from pathsnap.harness import cv_folds, top1
from pathsnap.io import discover_trips, load_trip

TOY = Path(__file__).parent / "fixtures" / "toy"


def toy_features(with_label=True):
    trip = load_trip(discover_trips("train", root=TOY)["toy01"])
    g = TripGeometry.from_trip(trip)
    c, _ = generate_candidates(g, "toy01", 60, 5, 150)
    if with_label:
        c = attach_labels(c, np.array([0, 0, 0, 2, 2]))
    return build_features(g, c)


def test_columns_finite_and_grouped():
    f = toy_features()
    assert set(FEATURE_COLUMNS) <= set(f.columns)
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == sum(map(len, FEATURE_GROUPS.values()))
    assert np.isfinite(f[FEATURE_COLUMNS].to_numpy()).all()


def test_no_label_leakage():
    a = toy_features(with_label=True)
    b = toy_features(with_label=False)
    assert np.array_equal(a[FEATURE_COLUMNS].to_numpy(), b[FEATURE_COLUMNS].to_numpy())


def test_direction_separates_twins():
    f = toy_features().set_index(["obs_idx", "arc_id"])
    # obs 1 drives east along arc 0 (node 0 -> 1); arc 1 is its westbound twin
    assert f.loc[(1, 0), "heading_cos"] > 0.9 and f.loc[(1, 1), "heading_cos"] < -0.9
    assert f.loc[(1, 0), "dist_m"] == f.loc[(1, 1), "dist_m"]
    assert f.loc[(1, 0), "has_reverse"] == 1


def test_cv_folds_fixed_and_balanced():
    trips = [f"{i:08d}" for i in range(80)]
    a, b = cv_folds(trips, 5), cv_folds(list(reversed(trips)), 5)
    assert a == b and sorted(pd.Series(a).value_counts().tolist()) == [16] * 5


def test_top1_counts_missing_positives_as_wrong():
    df = pd.DataFrame({"trip_id": ["t"] * 4, "obs_idx": [0, 0, 1, 1], "label": [1, 0, 0, 0]})
    assert top1(df, np.array([0.9, 0.1, 0.6, 0.4]), n_obs=2) == 0.5
