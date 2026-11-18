"""predict_trip on the toy fixture: contract keys, one arc per observation, no route used."""
import inspect
from pathlib import Path

import numpy as np

from pathsnap.io import discover_trips, load_trip
from pathsnap.pipeline import fill_missing, predict_trip

TOY = Path(__file__).parent / "fixtures" / "toy"


def test_predict_trip_contract():
    trip = load_trip(discover_trips("train", root=TOY)["toy01"], with_route=False)
    res = predict_trip(trip, None)
    assert {"gps", "pred_arcs", "route_lonlat", "timings_ms"} <= set(res)
    assert len(res["pred_arcs"]) == len(res["gps"]) == 5
    # heading picks the eastbound arc 0 (not its reverse twin 1) and then the northbound arc 2
    assert res["pred_arcs"][:2] == [0, 0] and res["pred_arcs"][-2:] == [2, 2]
    assert res["route_lonlat"][0] == [77.59, 12.93] and res["route_lonlat"][-1] == [77.591, 12.931]
    assert "route" not in inspect.signature(predict_trip).parameters


def test_scorer_artifact_is_used():
    trip = load_trip(discover_trips("train", root=TOY)["toy01"])
    # a scorer that prefers the highest arc id must change the output
    res = predict_trip(trip, {"scorer": lambda geom, c: c["arc_id"].to_numpy().astype(float)})
    assert res["method"]["scorer"] == "artifact" and max(res["pred_arcs"]) == 4


def test_fill_missing_nearest_in_time():
    pred = np.array([-1, 3, -1, -1, 7, -1])
    t = np.array([0, 1, 2, 5, 6, 9.0])
    assert fill_missing(pred, t).tolist() == [3, 3, 3, 7, 7, 7]
