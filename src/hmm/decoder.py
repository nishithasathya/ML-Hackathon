"""Hidden Markov Model + Viterbi route reconstruction.

Hidden states
    At GPS observation t the hidden state is one of that observation's candidate
    road positions s = (arc_id, offset_frac). The candidate table row keeps the
    mapping observation -> candidate -> road arc -> features -> classifier score.

Emission model
    log P(o_t | s) = w * log(max(p_clf(s), p_min))
    p_clf is the selected classifier's probability that the candidate is the
    reference arc; w (emission_weight) balances it against the transitions.

Transition model (Newson & Krumm 2009 style, with the directed network)
    d_gc    = straight-line distance between the two GPS fixes (metres)
    d_route = network distance from position s_{t} to s_{t+1}, following arc
              directions (RoadGraph.route_distance), searched only up to a
              physically plausible budget  v_max * dt + 2 * tolerance
    log P(s_{t+1} | s_t) = -|d_route - d_gc| / beta      if reachable
                         = unreachable_log_prob           otherwise
    If either fix is a flagged GPS spike (preprocessing: speed in AND out > 70 m/s)
    its displacement is meaningless, so the transition uses -d_route / beta
    (prefer the shortest legal move) instead of comparing with d_gc.
    A move that the network cannot explain is not forbidden (log prob -inf would
    make whole sequences impossible when a correct candidate is missing); it is
    heavily penalised instead, and the continuity metric reports it as a break.

Viterbi (log space, so long sequences cannot underflow)
    delta_0(j)  = e_0(j)
    delta_t(j)  = max_i [delta_{t-1}(i) + log A_t(i, j)] + e_t(j)
    psi_t(j)    = argmax_i [...]
    The best final state is traced back through psi.

Chains are cut (decoded independently) at
    * GPS gaps: a new preprocessing segment (dt > 30 s) - no transition is
      invented across missing intervals;
    * observations without any candidate - they receive no prediction.
A chain of length 1 reduces to picking the best emission.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.common.graph import RoadGraph


@dataclass
class HMMConfig:
    emission_weight: float = 1.0
    transition_beta_m: float = 20.0
    unreachable_log_prob: float = -30.0
    max_speed_mps: float = 60.0
    position_tolerance_m: float = 25.0
    min_prob: float = 1e-4

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def emission_log_probs(prob: np.ndarray, config: HMMConfig) -> np.ndarray:
    return config.emission_weight * np.log(np.clip(np.asarray(prob, dtype=float), config.min_prob, 1.0))


def transition_log_matrix(graph: RoadGraph, prev: Tuple[np.ndarray, np.ndarray], cur: Tuple[np.ndarray, np.ndarray],
                          gc_dist: float, dt: float, config: HMMConfig, spike: bool = False) -> np.ndarray:
    """(k_prev x k_cur) matrix of log transition probabilities (``spike``: ignore the GPS displacement)."""
    arcs_a, frac_a = prev
    arcs_b, frac_b = cur
    budget = config.max_speed_mps * max(dt, 1.0) + 2.0 * config.position_tolerance_m
    out = np.full((len(arcs_a), len(arcs_b)), config.unreachable_log_prob)
    for i, (a, fa) in enumerate(zip(arcs_a.tolist(), frac_a.tolist())):
        for j, (b, fb) in enumerate(zip(arcs_b.tolist(), frac_b.tolist())):
            d = graph.route_distance(a, fa, b, fb, budget)
            if np.isfinite(d):
                out[i, j] = -(d if spike else abs(d - gc_dist)) / config.transition_beta_m
    return out


def viterbi(emissions: List[np.ndarray], transitions: List[np.ndarray]) -> List[int]:
    """Most likely state index per step. ``transitions[t]`` maps step t -> t+1."""
    delta = emissions[0].copy()
    back: List[np.ndarray] = []
    for t in range(1, len(emissions)):
        scores = delta[:, None] + transitions[t - 1]
        back.append(np.argmax(scores, axis=0))
        delta = scores[back[-1], np.arange(scores.shape[1])] + emissions[t]
    path = [int(np.argmax(delta))]
    for ptr in reversed(back):
        path.append(int(ptr[path[-1]]))
    return path[::-1]


def decode_trip(candidates: pd.DataFrame, track: pd.DataFrame, graph: RoadGraph,
                config: HMMConfig = HMMConfig(), prob_col: str = "prob") -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Decode one trip.

    ``candidates``: obs_idx, arc_id, offset_frac, <prob_col> (any order; one trip).
    ``track``: PreprocessedTrip.track (obs_idx, t, x, y, segment_id).
    Returns (obs_idx, arc_id, offset_frac, prob) per decoded observation, and stats.
    """
    start = time.perf_counter()
    groups = {obs: g for obs, g in candidates.sort_values(["obs_idx", "distance_rank"] if "distance_rank" in candidates
                                                          else ["obs_idx"]).groupby("obs_idx", sort=False)}
    obs_ids = track["obs_idx"].to_numpy()
    t_arr, x_arr, y_arr = track["t"].to_numpy(), track["x"].to_numpy(), track["y"].to_numpy()
    seg = track["segment_id"].to_numpy()
    spike = track["is_speed_outlier"].to_numpy(dtype=bool) if "is_speed_outlier" in track else np.zeros(len(track), bool)

    # Split into chains: consecutive observations in the same segment that all have candidates.
    chains: List[List[int]] = []
    current: List[int] = []
    for pos, obs in enumerate(obs_ids):
        has = obs in groups
        if current and (not has or seg[pos] != seg[current[-1]]):
            chains.append(current)
            current = []
        if has:
            current.append(pos)
    if current:
        chains.append(current)

    out = []
    n_unreachable = 0
    for chain in chains:
        states = [(groups[obs_ids[p]]["arc_id"].to_numpy(), groups[obs_ids[p]]["offset_frac"].to_numpy(),
                   groups[obs_ids[p]][prob_col].to_numpy()) for p in chain]
        emissions = [emission_log_probs(s[2], config) for s in states]
        transitions = []
        for k in range(1, len(chain)):
            p0, p1 = chain[k - 1], chain[k]
            gc = float(np.hypot(x_arr[p1] - x_arr[p0], y_arr[p1] - y_arr[p0]))
            mat = transition_log_matrix(graph, states[k - 1][:2], states[k][:2], gc, float(t_arr[p1] - t_arr[p0]),
                                        config, spike=bool(spike[p0] or spike[p1]))
            transitions.append(mat)
        path = viterbi(emissions, transitions)
        for k, (p, j) in enumerate(zip(chain, path)):
            arcs, fracs, probs = states[k]
            out.append((int(obs_ids[p]), int(arcs[j]), float(fracs[j]), float(probs[j])))
            if k > 0 and transitions[k - 1][path[k - 1], j] == config.unreachable_log_prob:
                n_unreachable += 1
    result = pd.DataFrame(out, columns=["obs_idx", "arc_id", "offset_frac", "prob"])
    stats = {"n_chains": len(chains), "n_decoded": len(result), "n_unreachable_transitions": n_unreachable,
             "decode_time_s": time.perf_counter() - start}
    return result, stats
