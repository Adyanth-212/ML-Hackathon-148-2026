"""Road-network construction and connectivity analysis.

Arcs are *directed* (from_node -> to_node). A two-way street usually appears as
two opposite arcs, and two nodes may be joined by more than one arc, so the
graph is a ``MultiDiGraph`` keyed by arc_id.
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd


def arc_lengths_m(node_x: np.ndarray, node_y: np.ndarray, arcs: pd.DataFrame) -> np.ndarray:
    """Straight-line length of every arc in metres (nodes already projected)."""
    f, t = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    return np.hypot(node_x[t] - node_x[f], node_y[t] - node_y[f])


def build_graph(n_nodes: int, arcs: pd.DataFrame, lengths: np.ndarray | None = None) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_nodes_from(range(n_nodes))
    for arc_id, (u, v) in enumerate(arcs[["from_node", "to_node"]].to_numpy()):
        g.add_edge(int(u), int(v), key=arc_id, arc_id=arc_id,
                   length=float(lengths[arc_id]) if lengths is not None else 1.0)
    return g


def successor_arcs(arcs: pd.DataFrame) -> dict[int, list[int]]:
    """arc_id -> arcs that can directly follow it (their from_node == its to_node)."""
    by_from: dict[int, list[int]] = defaultdict(list)
    for arc_id, u in enumerate(arcs["from_node"].to_numpy()):
        by_from[int(u)].append(arc_id)
    return {a: by_from.get(int(v), []) for a, v in enumerate(arcs["to_node"].to_numpy())}


def network_summary(nodes: pd.DataFrame, arcs: pd.DataFrame) -> dict[str, float]:
    """Size and connectivity statistics for one road network."""
    n_nodes, n_arcs = len(nodes), len(arcs)
    f, t = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    bad_ref = int(((f < 0) | (f >= n_nodes) | (t < 0) | (t >= n_nodes)).sum())
    out: dict[str, float] = {
        "n_nodes": n_nodes,
        "n_arcs": n_arcs,
        "arcs_bad_node_ref": bad_ref,
        "self_loops": int((f == t).sum()),
        "duplicate_arcs": int(arcs.duplicated(["from_node", "to_node"]).sum()),
    }
    if bad_ref or n_nodes == 0:
        return out
    pairs = set(zip(f.tolist(), t.tolist()))
    out["frac_arcs_with_reverse"] = float(np.mean([(v, u) in pairs for u, v in zip(f, t)])) if n_arcs else 0.0

    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(zip(f.tolist(), t.tolist()))
    deg = np.array([d for _, d in g.degree()])
    out["isolated_nodes"] = int((deg == 0).sum())
    wcc = [len(c) for c in nx.weakly_connected_components(g)]
    scc = [len(c) for c in nx.strongly_connected_components(g)]
    out["n_weak_components"] = len(wcc)
    out["largest_wcc_frac"] = max(wcc) / n_nodes
    out["n_strong_components"] = len(scc)
    out["largest_scc_frac"] = max(scc) / n_nodes
    return out


def route_summary(route: np.ndarray, arcs: pd.DataFrame) -> dict[str, float]:
    """Structure of a reference route: size, validity and arc-to-arc continuity."""
    n_arcs = len(arcs)
    valid = (route >= 0) & (route < n_arcs)
    out: dict[str, float] = {
        "route_len": int(len(route)),
        "route_unique_arcs": int(len(np.unique(route))),
        "route_invalid_ids": int((~valid).sum()),
    }
    r = route[valid]
    if len(r) >= 2:
        to_n = arcs["to_node"].to_numpy()[r[:-1]]
        from_n = arcs["from_node"].to_numpy()[r[1:]]
        out["route_consecutive_connected_frac"] = float(np.mean(to_n == from_n))
        out["route_repeated_consecutive"] = int((r[:-1] == r[1:]).sum())
    return out
