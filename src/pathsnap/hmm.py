"""Candidate-state HMM/Viterbi decoder for map matching.

Supports ragged candidate sets, directed road-network transitions, distance
and classifier emissions, long GPS gaps, and disconnected segments.
"""
from __future__ import annotations
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from .config import CONFIG_DIR


def _log_softmax(x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Emission scores must be finite.")
    m = float(x.max())
    return x - (m + np.log(np.exp(x - m).sum()))


def viterbi(emission_logp, transition_logp):
    """Fixed-state Viterbi. Transition rows are previous states, columns next."""
    e, tr = np.asarray(emission_logp, float), np.asarray(transition_logp, float)
    if e.ndim != 2 or not e.shape[0] or not e.shape[1]:
        raise ValueError("emission_logp must be a non-empty (T, N) array.")
    n = e.shape[1]
    if tr.shape != (n, n):
        raise ValueError("transition_logp must have shape (N, N).")
    if np.isnan(e).any() or np.isnan(tr).any() or np.isposinf(e).any() or np.isposinf(tr).any():
        raise ValueError("Log probabilities cannot contain NaN or +inf.")
    score = np.full_like(e, -np.inf)
    back = np.zeros(e.shape, dtype=np.int32)
    score[0] = e[0]
    for t in range(1, len(e)):
        z = score[t - 1][:, None] + tr
        back[t] = np.argmax(z, axis=0)
        score[t] = np.max(z, axis=0) + e[t]
        if not np.isfinite(score[t]).any():
            raise ValueError(f"No valid state path reaches time step {t}.")
    path = np.empty(len(e), dtype=np.int64)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(len(e) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_ragged(state_ids, emission_logp, transition_logp, break_before=None):
    """Viterbi for variable state counts; returns chosen IDs and reset flags."""
    n = len(state_ids)
    if len(emission_logp) != n or len(transition_logp) != max(0, n - 1):
        raise ValueError("State, emission and transition sequence lengths do not match.")
    forced = np.zeros(n, bool) if break_before is None else np.asarray(break_before, bool)
    if forced.shape != (n,):
        raise ValueError("break_before must have one flag per step.")
    out, breaks = np.full(n, -1, np.int64), forced.copy()
    start = 0
    while start < n:
        if len(state_ids[start]) == 0:
            breaks[start] = True
            start += 1
            continue
        end = start + 1
        while end < n and len(state_ids[end]) and not forced[end]:
            tr = np.asarray(transition_logp[end - 1], float)
            if tr.shape != (len(state_ids[end - 1]), len(state_ids[end])):
                raise ValueError("A transition matrix has an incompatible shape.")
            if not np.isfinite(tr).any():
                breaks[end] = True
                break
            end += 1
        score = np.asarray(emission_logp[start], float).copy()
        backs = []
        for t in range(start + 1, end):
            tr = np.asarray(transition_logp[t - 1], float)
            z = score[:, None] + tr
            nxt = np.max(z, axis=0) + np.asarray(emission_logp[t], float)
            if not np.isfinite(nxt).any():
                breaks[t] = True
                end = t
                break
            backs.append(np.argmax(z, axis=0))
            score = nxt
        j = int(np.argmax(score))
        out[end - 1] = int(np.asarray(state_ids[end - 1])[j])
        for k in range(len(backs) - 1, -1, -1):
            j = int(backs[k][j])
            out[start + k] = int(np.asarray(state_ids[start + k])[j])
        start = end
    return out, breaks


def load_hmm_config(config=None):
    if config is not None:
        return config
    with (CONFIG_DIR / "hmm.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _emissions(cands, scores, cfg):
    floor = float(cfg["emission"].get("prob_floor", 1e-6))
    sigma = max(float(cfg["emission"].get("distance_sigma_m", 10.0)), 1e-6)
    weight = float(cfg["emission"].get("classifier_weight", 1.0))
    if not 0 <= weight <= 1:
        raise ValueError("classifier_weight must be between 0 and 1.")
    states, emissions, obs_ids = [], [], []
    for obs, frame in cands.groupby("obs_idx", sort=True):
        rows = frame.index.to_numpy()
        raw = np.asarray(scores)[rows]
        if np.all((raw >= 0) & (raw <= 1)):
            p = np.maximum(raw, floor)
            cls = np.log(p / p.sum())
        else:
            cls = _log_softmax(raw)
        dlog = _log_softmax(-(frame.dist_m.to_numpy(float) ** 2) / (2 * sigma ** 2))
        emissions.append(_log_softmax(weight * cls + (1 - weight) * dlog))
        states.append(frame.arc_id.to_numpy(np.int64))
        obs_ids.append(int(obs))
    return obs_ids, states, emissions


def _transition(geom, oi, oj, prev_arcs, next_arcs, prev_frac, next_frac, cfg, graph):
    beta = max(float(cfg["transition"].get("beta_m", 20)), 1e-6)
    factor = float(cfg["transition"].get("max_route_factor", 2))
    slack = float(cfg["transition"].get("route_slack_m", 200))
    straight = float(np.hypot(geom.obs_x[oj] - geom.obs_x[oi], geom.obs_y[oj] - geom.obs_y[oi]))
    lengths = geom.arc_lengths()
    cutoff = factor * straight + slack
    result = np.full((len(prev_arcs), len(next_arcs)), -np.inf)
    cache = {}
    for i, a in enumerate(prev_arcs):
        node = int(geom.to_node[a])
        if node not in cache:
            cache[node] = nx.single_source_dijkstra_path_length(graph, node, cutoff=cutoff, weight="length")
        distmap = cache[node]
        for j, b in enumerate(next_arcs):
            if a == b:
                route = max(0.0, (next_frac[j] - prev_frac[i]) * lengths[a])
            else:
                middle = distmap.get(int(geom.from_node[b]))
                if middle is None:
                    continue
                route = (1 - prev_frac[i]) * lengths[a] + middle + next_frac[j] * lengths[b]
            if route <= cutoff:
                result[i, j] = -abs(route - straight) / beta
    for i in range(len(result)):
        finite = np.isfinite(result[i])
        if finite.any():
            result[i, finite] = _log_softmax(result[i, finite])
    return result


def decode_candidates(geom, candidates, scores, config=None):
    """Return (one predicted arc per GPS observation, break flags)."""
    cfg = load_hmm_config(config)
    scores = np.asarray(scores, float)
    if len(scores) != len(candidates):
        raise ValueError("One score is required per candidate.")
    pred = np.full(geom.n_obs, -1, np.int64)
    breaks = np.zeros(geom.n_obs, bool)
    if candidates.empty:
        return pred, breaks
    c = candidates.reset_index(drop=True)
    obs_ids, states, emissions = _emissions(c, scores, cfg)
    lengths = geom.arc_lengths()
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(geom.node_x)))
    for a, (u, v) in enumerate(zip(geom.from_node, geom.to_node)):
        graph.add_edge(int(u), int(v), length=float(lengths[a]))
    transitions, forced = [], np.zeros(len(obs_ids), bool)
    gap_s = float(cfg["gaps"].get("break_seconds", 30))
    for k in range(1, len(obs_ids)):
        oi, oj = obs_ids[k - 1], obs_ids[k]
        if oj != oi + 1 or geom.t[oj] - geom.t[oi] > gap_s:
            forced[k] = True
            transitions.append(np.full((len(states[k - 1]), len(states[k])), -np.inf))
            continue
        p = c[c.obs_idx == oi].proj_frac.to_numpy(float).clip(0, 1)
        q = c[c.obs_idx == oj].proj_frac.to_numpy(float).clip(0, 1)
        transitions.append(_transition(geom, oi, oj, states[k - 1], states[k], p, q, cfg, graph))
    decoded, small_breaks = viterbi_ragged(states, emissions, transitions, forced)
    for obs, arc, br in zip(obs_ids, decoded, small_breaks):
        pred[obs], breaks[obs] = arc, br
    return pred, breaks


def make_decoder(config=None):
    """Callable adapter for artifacts['decoder'] in pathsnap.pipeline."""
    return lambda geom, candidates, scores: decode_candidates(geom, candidates, scores, config)
