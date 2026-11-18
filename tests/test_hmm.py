import numpy as np
import pandas as pd
from types import SimpleNamespace
from pathsnap.hmm import viterbi, viterbi_ragged, decode_candidates


def test_viterbi_hand_computed_path():
    e = np.log([[0.9, 0.1], [0.1, 0.9]])
    tr = np.log([[0.1, 0.9], [0.9, 0.1]])
    assert viterbi(e, tr).tolist() == [0, 1]


def test_viterbi_rejects_bad_transition_shape():
    try:
        viterbi(np.zeros((2, 2)), np.zeros((3, 3)))
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_ragged_states_and_chain_reset():
    states = [np.array([10, 11]), np.array([11]), np.array([20, 21])]
    emissions = [np.log([0.9, 0.1]), np.log([1.0]), np.log([0.1, 0.9])]
    transitions = [np.log([[0.8], [0.2]]), np.full((1, 2), -np.inf)]
    path, breaks = viterbi_ragged(states, emissions, transitions, [False, False, True])
    assert path.tolist() == [10, 11, 21]
    assert breaks.tolist() == [False, False, True]


def test_decode_candidates_shape_and_gap():
    geom = SimpleNamespace(
        n_obs=2, t=np.array([0.0, 1.0]),
        obs_x=np.array([0.0, 10.0]), obs_y=np.array([0.0, 0.0]),
        node_x=np.array([0.0, 10.0, 20.0]), node_y=np.array([0.0, 0.0, 0.0]),
        from_node=np.array([0, 1]), to_node=np.array([1, 2]),
        arc_lengths=lambda: np.array([10.0, 10.0]),
    )
    c = pd.DataFrame({"obs_idx":[0,0,1,1], "arc_id":[0,1,0,1],
                      "dist_m":[1.,5.,5.,1.], "proj_frac":[.5,0.,1.,.5]})
    pred, breaks = decode_candidates(geom, c, np.array([.9,.1,.1,.9]))
    assert pred.shape == (2,) and breaks.shape == (2,)
    assert np.all(pred >= 0)
