"""HMM + Viterbi route reconstruction (TODO, phase P6).

Hidden states = candidate arcs of each observation. Emission = classifier
probability (floored, optionally blended with a distance term). Transition =
exp(-|network distance - straight-line distance| / beta) on the directed graph.
Decode in log space; break the chain at large GPS gaps or when no transition is
possible. Parameters live in configs/hmm.yaml.
"""
from __future__ import annotations


def viterbi(emission_logp, transition_logp):
    raise NotImplementedError
