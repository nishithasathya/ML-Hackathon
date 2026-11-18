"""HMM / Viterbi: emissions, transitions, recurrence, gaps, missing candidates, disconnected roads."""
import itertools

import numpy as np
import pandas as pd
import pytest

from src.common.graph import RoadGraph
from src.common.io import load_trip
from src.hmm.decoder import HMMConfig, decode_trip, emission_log_probs, transition_log_matrix, viterbi
from src.preprocessing.pipeline import preprocess_trip


@pytest.fixture
def setup(synthetic_root, grid):
    pre = preprocess_trip(load_trip("train", "00000001", synthetic_root))
    return pre, RoadGraph(pre.nodes_xy, pre.arcs), grid


def test_emissions_are_monotone_and_finite():
    e = emission_log_probs(np.array([0.0, 0.1, 0.9]), HMMConfig(emission_weight=2.0))
    assert np.all(np.isfinite(e)) and e[0] < e[1] < e[2]


def test_transition_prefers_connected_arcs(setup):
    pre, g, grid = setup
    a = grid["arc_id"](grid["node_id"](0, 0), grid["node_id"](1, 0))
    nxt = grid["arc_id"](grid["node_id"](1, 0), grid["node_id"](2, 0))
    far = grid["arc_id"](grid["node_id"](0, 2), grid["node_id"](1, 2))
    cfg = HMMConfig()
    mat = transition_log_matrix(g, (np.array([a]), np.array([0.9])), (np.array([nxt, far]), np.array([0.1, 0.1])),
                                gc_dist=20.0, dt=1.0, config=cfg)
    assert mat[0, 0] > mat[0, 1]
    assert mat[0, 1] == cfg.unreachable_log_prob  # disconnected road: penalised, not impossible


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(0)
    em = [np.log(rng.dirichlet(np.ones(k))) for k in (3, 2, 4, 3)]
    tr = [np.log(rng.dirichlet(np.ones(em[t + 1].size), size=em[t].size)) for t in range(3)]
    best = max(itertools.product(*[range(e.size) for e in em]),
               key=lambda p: sum(em[t][p[t]] for t in range(4)) + sum(tr[t][p[t], p[t + 1]] for t in range(3)))
    assert viterbi(em, tr) == list(best)


def test_single_step_chain_takes_best_emission():
    assert viterbi([np.log(np.array([0.2, 0.7, 0.1]))], []) == [1]


def _candidates(pre, g, grid, noisy_obs, keep_spike=False):
    """Two candidates per fix on the first street: the correct eastbound arc and its westbound twin.
    The classifier is made to prefer the wrong twin at ``noisy_obs``."""
    east = grid["arc_id"](grid["node_id"](0, 0), grid["node_id"](1, 0))
    west = grid["arc_id"](grid["node_id"](1, 0), grid["node_id"](0, 0))
    rows = []
    track = pre.track[pre.track["x"] - pre.track["x"].min() < 95]
    if not keep_spike:
        track = track[~track["is_speed_outlier"]]
    track = track.iloc[:8]
    for r in track.itertuples():
        f = float(np.clip((r.x - pre.nodes_xy[grid["node_id"](0, 0), 0]) / 100.0, 0, 1))
        p_east = 0.3 if r.obs_idx in noisy_obs else 0.8
        rows += [(r.obs_idx, east, f, 1, p_east), (r.obs_idx, west, 1 - f, 2, 1 - p_east)]
    return pd.DataFrame(rows, columns=["obs_idx", "arc_id", "offset_frac", "distance_rank", "prob"]), track, east


def test_hmm_corrects_an_isolated_classifier_error(setup):
    pre, g, grid = setup
    cands, track, east = _candidates(pre, g, grid, noisy_obs={3})
    decoded, stats = decode_trip(cands, track, g, HMMConfig())
    assert (decoded["arc_id"] == east).all()
    assert stats["n_chains"] == 1


def test_gps_spike_does_not_force_a_u_turn(setup):
    pre, g, grid = setup
    cands, track, east = _candidates(pre, g, grid, noisy_obs=set(), keep_spike=True)
    assert track["is_speed_outlier"].any()
    decoded, _ = decode_trip(cands, track, g, HMMConfig())
    assert (decoded["arc_id"] == east).all()


def test_missing_candidates_split_the_chain(setup):
    pre, g, grid = setup
    cands, track, _ = _candidates(pre, g, grid, noisy_obs=set())
    cands = cands[cands["obs_idx"] != track["obs_idx"].iloc[4]]
    decoded, stats = decode_trip(cands, track, g, HMMConfig())
    assert stats["n_chains"] == 2
    assert track["obs_idx"].iloc[4] not in set(decoded["obs_idx"])


def test_gps_gap_starts_a_new_chain(synthetic_root):
    pre = preprocess_trip(load_trip("train", "00000001", synthetic_root))
    g = RoadGraph(pre.nodes_xy, pre.arcs)
    from src.features.features import build_trip_features
    table, _ = build_trip_features(pre, with_labels=False)
    decoded, stats = decode_trip(table.assign(prob=0.5), pre.track, g, HMMConfig())
    assert stats["n_chains"] >= pre.track["segment_id"].nunique()  # never a transition across a gap
    assert len(decoded) == table["obs_idx"].nunique()
