import numpy as np
import pandas as pd

from src.common.io import load_trip
from src.preprocessing import run_preprocessing
from src.preprocessing.cleaning import drop_invalid_coordinates, enforce_time_order, track_to_frame
from src.preprocessing.pipeline import PreprocessConfig, preprocess_trip
from tests.conftest import DUPLICATE_ROW, SPIKE_ROW


def test_drop_invalid_coordinates():
    df = track_to_frame(np.array([[10, 50, 1], [np.nan, 50, 2], [0, 0, 3], [200, 50, 4], [10, 50, 5]], float))
    clean, n = drop_invalid_coordinates(df)
    assert n == 3 and clean["obs_idx"].tolist() == [0, 4]


def test_enforce_time_order_sorts_and_dedupes():
    df = track_to_frame(np.array([[1, 1, 3], [1, 1, 1], [1, 1, 1], [1, 1, 2]], float))
    clean, n_missing, n_unordered, n_dup = enforce_time_order(df)
    assert clean["t"].tolist() == [1, 2, 3]
    assert (n_missing, n_dup) == (0, 1) and n_unordered >= 1
    assert clean["obs_idx"].tolist() == [1, 3, 0]  # original row ids are preserved


def test_preprocess_trip(synthetic_root):
    raw = load_trip("train", "00000001", synthetic_root)
    pre = preprocess_trip(raw, PreprocessConfig(gap_threshold_s=30, max_speed_mps=70))
    track = pre.track
    assert pre.report["n_dropped_duplicate_time"] == 1
    assert DUPLICATE_ROW not in set(track["obs_idx"])
    assert pre.report["n_segments"] == 2 and pre.report["n_gaps"] == 1
    assert track.loc[track["is_speed_outlier"], "obs_idx"].tolist() == [SPIKE_ROW]
    assert track["t"].is_monotonic_increasing
    assert pre.nodes_xy.shape == (9, 2)
    # first node is the grid origin; it should project ~3 m south of the first GPS fix
    assert abs(pre.nodes_xy[0, 1] - (track["y"].iloc[0] - 3)) < 0.5


def test_drop_speed_outliers_option(synthetic_root):
    raw = load_trip("train", "00000001", synthetic_root)
    pre = preprocess_trip(raw, PreprocessConfig(drop_speed_outliers=True))
    assert SPIKE_ROW not in set(pre.track["obs_idx"])


def test_run_preprocessing_cli(synthetic_root, tmp_path):
    out = tmp_path / "pre"
    run_preprocessing.main(["--partition", "train", "--data-root", str(synthetic_root), "--output-dir", str(out)])
    report = pd.read_csv(out / "preprocessing_report.csv", dtype={"trip_id": str})
    assert report["trip_id"].tolist() == ["00000001"]
    assert (out / "00000001.csv").exists() and (out / "preprocess_config.json").exists()
