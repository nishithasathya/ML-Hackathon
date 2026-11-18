"""Shared metrics, road graph and label-free loading (integration lead)."""
import numpy as np
import pandas as pd
import pytest

from src.common.config import get_eval_data_root
from src.common.graph import RoadGraph
from src.common.io import load_reference_route, load_trip, load_trip_for_inference
from src.evaluation.metrics import (classification_metrics, observation_accuracy, route_continuity, route_overlap,
                                    select_top1)
from src.preprocessing.pipeline import preprocess_trip


def test_classification_metrics_known_values():
    m = classification_metrics([1, 0, 1, 0], [0.9, 0.2, 0.4, 0.6])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["macro_f1"] == pytest.approx(0.5)
    assert m["log_loss"] > 0


def test_select_top1_breaks_ties_by_distance_rank():
    c = pd.DataFrame({"trip_id": ["a"] * 3, "obs_idx": [0, 0, 0], "arc_id": [5, 6, 7], "distance_rank": [2, 1, 3],
                      "score": [0.7, 0.7, 0.1], "is_reference": [False, True, False]})
    assert select_top1(c)["arc_id"].tolist() == [6]


def test_observation_accuracy_counts_unpredicted_observations_as_wrong():
    chosen = pd.DataFrame({"trip_id": ["a", "a"], "obs_idx": [0, 1], "is_reference": [True, False]})
    obs = pd.DataFrame({"trip_id": ["a"] * 4, "obs_idx": [0, 1, 2, 3], "has_reference_candidate": [True, True, True, False]})
    m = observation_accuracy(chosen, obs)
    assert m["top1_accuracy"] == pytest.approx(0.25)
    assert m["candidate_miss_rate"] == pytest.approx(0.25)
    assert m["n_predicted"] == 2


def test_road_graph_distances(synthetic_root, grid):
    pre = preprocess_trip(load_trip("train", "00000001", synthetic_root))
    g = RoadGraph(pre.nodes_xy, pre.arcs)
    a = grid["arc_id"](grid["node_id"](0, 0), grid["node_id"](1, 0))
    b = grid["arc_id"](grid["node_id"](1, 0), grid["node_id"](2, 0))
    twin = grid["arc_id"](grid["node_id"](1, 0), grid["node_id"](0, 0))
    assert g.route_distance(a, 0.5, b, 0.5, 500) == pytest.approx(100, abs=0.5)
    assert g.route_distance(a, 0.5, twin, 0.5, 500) == pytest.approx(100, abs=0.5)  # drive to the end and back
    assert g.route_distance(b, 0.5, a, 0.5, 150) == np.inf  # going backwards needs a detour > budget
    assert g.route_distance(a, 0.6, a, 0.4, 10) == pytest.approx(20, abs=0.5)  # GPS jitter on one arc is allowed


def test_route_continuity_detects_break(synthetic_root, grid):
    pre = preprocess_trip(load_trip("train", "00000001", synthetic_root))
    g = RoadGraph(pre.nodes_xy, pre.arcs)
    a = grid["arc_id"](grid["node_id"](0, 0), grid["node_id"](1, 0))
    far = grid["arc_id"](grid["node_id"](0, 2), grid["node_id"](1, 2))
    track = pre.track.iloc[:3]
    good = pd.DataFrame({"obs_idx": track["obs_idx"], "arc_id": [a] * 3, "offset_frac": [0.0, 0.1, 0.2]})
    bad = good.assign(arc_id=[a, far, a])
    assert route_continuity(good, track, g)["route_continuity"] == 1.0
    assert route_continuity(bad, track, g)["n_breaks"] == 2


def test_route_overlap(synthetic_root, grid):
    pre = preprocess_trip(load_trip("train", "00000001", synthetic_root))
    g = RoadGraph(pre.nodes_xy, pre.arcs)
    perfect = route_overlap(grid["route"], grid["route"], g)
    assert perfect["route_f1"] == pytest.approx(1.0)
    half = route_overlap(grid["route"][:1], grid["route"][:2], g)
    assert half["route_precision"] == pytest.approx(1.0) and half["route_recall"] == pytest.approx(0.5, abs=1e-3)


def test_inference_loader_never_reads_the_route(synthetic_root):
    raw = load_trip_for_inference(synthetic_root / "student_test", "00000002")
    assert raw.route.size == 0 and raw.n_observations > 0
    assert load_reference_route(synthetic_root / "student_test", "00000002").size == 3
    assert load_reference_route(synthetic_root / "student_test", "missing") is None


def test_eval_data_root_comes_from_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("PATHSNAP_EVAL_DATA_ROOT", raising=False)
    assert get_eval_data_root() is None
    monkeypatch.setenv("PATHSNAP_EVAL_DATA_ROOT", str(tmp_path))
    assert get_eval_data_root() == tmp_path.resolve()
