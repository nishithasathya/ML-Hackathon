import pandas as pd

from src.common.io import load_trip
from src.preprocessing.pipeline import preprocess_trip
from src.visualization import plots, run_visualizations
from src.visualization.folium_maps import build_trip_map, nearby_arc_ids, save_trip_map


def test_folium_map_has_layers(synthetic_root, tmp_path):
    raw = load_trip("train", "00000001", synthetic_root)
    pre = preprocess_trip(raw)
    cands = pd.DataFrame({"obs_idx": [0], "arc_id": [0], "distance_m": [3.0], "distance_rank": [1]})
    path = save_trip_map(build_trip_map(raw, track=pre.track, candidates=cands, predicted_arc_ids=raw.route),
                         tmp_path / "m.html")
    html = path.read_text()
    for name in ("Reference route", "Raw GPS observations", "Speed outliers (flagged)",
                 "Segment starts after time gap", "Candidate arcs", "Predicted route"):
        assert name in html


def test_nearby_arcs_cover_track(synthetic_root):
    raw = load_trip("train", "00000001", synthetic_root)
    assert len(nearby_arc_ids(raw, buffer_m=50)) > 0


def test_dataset_plots_from_small_frames(tmp_path):
    stats = pd.DataFrame({"partition": ["train", "train", "student_test"], "trip_id": ["1", "2", "3"],
                          "center_lon": [10, 37, -80], "center_lat": [50, 55, 30],
                          "n_observations": [100, 2000, 500], "n_nodes": [10, 100, 50], "n_arcs": [20, 200, 90],
                          "gps_route_dist_median_m": [4, 5, 130], "max_gap_s": [1, 30, 900]})
    assert plots.plot_trip_locations(stats, tmp_path / "a.png").exists()
    assert plots.plot_distribution(stats, "n_observations", tmp_path / "b.png", "t", "x", log_x=True).exists()
    assert plots.plot_network_size(stats, tmp_path / "c.png").exists()
    issues = pd.DataFrame({"trip_id": ["1", "2"], "code": ["TIME_GAP", "TIME_GAP"]})
    assert plots.plot_validation_issues(issues, tmp_path / "d.png").exists()
    sweep = pd.DataFrame({"radius_m": [25, 50], "candidate_miss_rate": [0.1, 0.05],
                          "mean_candidates_per_obs": [3, 5], "subset": ["all", "all"]})
    assert plots.plot_radius_sweep(sweep, tmp_path / "e.png").exists()


def test_interval_counts(synthetic_root):
    counts = plots.interval_category_counts(data_root=synthetic_root)
    assert counts.loc["<=0 s", "train"] == 1  # the duplicate timestamp
    assert counts.loc["31-300 s", "train"] == 1  # the 60 s gap


def test_cli_maps(synthetic_root, tmp_path):
    run_visualizations.main(["maps", "--partition", "train", "--data-root", str(synthetic_root),
                             "--output-root", str(tmp_path)])
    assert (tmp_path / "maps" / "train_00000001.html").exists()
    assert (tmp_path / "figures" / "trip_train_00000001.png").exists()


def test_map_draws_classifier_and_hmm_layers(synthetic_root, tmp_path):
    raw = load_trip("train", "00000001", synthetic_root)
    fmap = build_trip_map(raw, predicted_arc_ids=list(raw.route), classifier_arc_ids=list(raw.route[:1]))
    html = save_trip_map(fmap, tmp_path / "pred.html").read_text()
    assert "Predicted route (HMM/Viterbi)" in html and "Classifier-only prediction (no HMM)" in html
