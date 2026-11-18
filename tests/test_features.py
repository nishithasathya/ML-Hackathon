import numpy as np
import pandas as pd
import pytest

from src.common.geo import point_segment_distance
from src.common.io import load_trip
from src.features import run_features
from src.features.candidates import CandidateConfig, generate_candidates
from src.features.features import build_trip_features, observation_features
from src.features.labels import align_track_to_route
from src.features.spatial_index import ArcIndex
from src.preprocessing.pipeline import preprocess_trip
from tests.conftest import SPIKE_ROW


@pytest.fixture
def pre(synthetic_root):
    return preprocess_trip(load_trip("train", "00000001", synthetic_root))


def test_arc_index_matches_brute_force(pre):
    # small spacing relative to 100 m arcs exercises the "sample along arc" logic
    index = ArcIndex(pre.nodes_xy, pre.arcs, sample_spacing_m=30)
    x, y = pre.track["x"].to_numpy(), pre.track["y"].to_numpy()
    found = index.arcs_within(x, y, radius_m=40)
    for px, py, ids in zip(x, y, found):
        d, _ = point_segment_distance(px, py, index.ax, index.ay, index.bx, index.by)
        assert set(ids.tolist()) == set(np.flatnonzero(d <= 40).tolist())


def test_candidates_sorted_and_limited(pre):
    cands = generate_candidates(pre.track, ArcIndex(pre.nodes_xy, pre.arcs), CandidateConfig(radius_m=50, max_candidates=3))
    assert cands.groupby("obs_idx").size().max() <= 3
    first = cands[cands["obs_idx"] == 0]
    assert first["distance_m"].is_monotonic_increasing and first["distance_rank"].tolist() == [1, 2, 3]
    assert SPIKE_ROW not in set(cands["obs_idx"])  # 300 m away from every road: no candidates


def test_route_alignment_is_monotonic_and_correct(pre, grid):
    labels = align_track_to_route(pre.track, pre.nodes_xy, pre.arcs, np.array(grid["route"]))
    assert labels["ref_route_pos"].is_monotonic_increasing
    by_obs = labels.set_index("obs_idx")["ref_arc_id"]
    node = grid["node_id"]
    assert by_obs[2] == grid["arc_id"](node(0, 0), node(1, 0))  # x = 20 m, first street
    assert by_obs[15] == grid["arc_id"](node(1, 0), node(2, 0))  # x = 150 m
    assert by_obs.iloc[-1] == grid["arc_id"](node(2, 0), node(2, 1))  # last fix, going north


def test_observation_features(pre):
    feats = observation_features(pre.track).set_index("obs_idx")
    assert feats.loc[2, "speed_mps"] == pytest.approx(10, abs=0.5)
    assert feats.loc[2, "heading_deg"] == pytest.approx(90, abs=2)  # driving east
    assert np.isnan(feats.loc[feats["is_segment_start"]].iloc[1]["dt_prev"])  # nothing measured across the gap


def test_build_trip_features(pre):
    table, summary = build_trip_features(pre, CandidateConfig(radius_m=50))
    assert {"heading_diff_deg", "distance_m", "is_reference", "arc_has_reverse"} <= set(table.columns)
    assert summary["candidate_miss_rate"] < 0.1  # only the spike lacks candidates
    ref = table[table["is_reference"]]
    assert ref["heading_diff_deg"].median() < 20  # vehicle heading agrees with the reference arc
    _, no_label_summary = build_trip_features(pre, with_labels=False)
    assert "candidate_miss_rate" not in no_label_summary


def test_run_features_cli(synthetic_root, tmp_path):
    out = tmp_path / "feat"
    run_features.main(["--partition", "train", "--data-root", str(synthetic_root), "--output-dir", str(out)])
    summary = pd.read_csv(out / "feature_summary.csv", dtype={"trip_id": str})
    assert summary["trip_id"].tolist() == ["00000001"]
    run_features.main(["--partition", "train", "--data-root", str(synthetic_root), "--output-dir", str(out),
                       "--sweep", "25", "50"])
    assert set(pd.read_csv(out / "radius_sweep.csv")["radius_m"]) == {25, 50}


# ---------------------------------------------------------------- phase 2
def test_feature_list_excludes_labels_and_trip_progress():
    from src.features.features import FEATURE_COLUMNS, LABEL_COLUMNS
    assert not set(FEATURE_COLUMNS) & set(LABEL_COLUMNS)
    assert not {"t_rel", "cum_dist_m", "segment_id", "trip_id", "obs_idx", "arc_id"} & set(FEATURE_COLUMNS)


def test_inference_features_have_no_labels_and_full_feature_set(pre):
    from src.features.features import FEATURE_COLUMNS, LABEL_COLUMNS, model_matrix
    table, _ = build_trip_features(pre, with_labels=False)
    assert not set(LABEL_COLUMNS) & set(table.columns)
    X = model_matrix(table)
    assert list(X.columns) == FEATURE_COLUMNS and len(X) == len(table)


def test_context_features(pre):
    table, _ = build_trip_features(pre, with_labels=True)
    ref = table[table["is_reference"]]
    # the vehicle drives along the reference arc: positive along-arc motion, neighbours close to it
    assert (ref["along_arc_motion_m"].dropna() > 0).mean() > 0.8
    assert ref["prev_obs_dist_m"].median() < 10
    starts = table[table["is_segment_start"].astype(bool)]
    assert starts["prev_obs_dist_m"].isna().all()  # never measured across a gap


def test_duplicate_arcs_share_the_reference_label(synthetic_root):
    from src.common.io import RawTrip
    raw = load_trip("train", "00000001", synthetic_root)
    dup = RawTrip(raw.trip_id, raw.partition, raw.track, raw.nodes, np.vstack([raw.arcs, raw.arcs[raw.route[0]]]),
                  raw.route, raw.directory)
    table, _ = build_trip_features(preprocess_trip(dup), with_labels=True)
    first = table[table["obs_idx"] == table["obs_idx"].min()]
    labels = first.set_index("arc_id")["is_reference"]
    assert labels[raw.route[0]] and labels[len(raw.arcs)]  # the copy is labelled like the original
