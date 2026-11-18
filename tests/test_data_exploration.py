import pandas as pd

from src.common.io import load_trip
from src.data_exploration import run_eda
from src.data_exploration.inventory import build_inventory, summarise_inventory
from src.data_exploration.statistics import gps_to_route_distance, trip_statistics
from src.data_exploration.validation import validate_trip


def test_inventory_counts(synthetic_root):
    inv = build_inventory(data_root=synthetic_root)
    assert len(inv) == 2 and inv["all_files_present"].all()
    assert inv["track_has_crlf"].all()
    assert set(inv["track_column_counts"]) == {"3"}
    summary = summarise_inventory(inv)
    assert summary["n_trips"].sum() == 2


def test_validation_finds_known_problems(synthetic_root):
    codes = {i.code for i in validate_trip(load_trip("train", "00000001", synthetic_root))}
    assert "DUPLICATE_TIMESTAMP" in codes
    assert "TIME_GAP" in codes
    assert "ROUTE_DISCONTINUITY" not in codes  # the synthetic route is connected
    assert "ARC_NODE_OUT_OF_RANGE" not in codes


def test_trip_statistics(synthetic_root):
    trip = load_trip("train", "00000001", synthetic_root)
    stats = trip_statistics(trip)
    assert stats["n_nodes"] == 9 and stats["n_arcs"] == 24
    assert stats["n_weak_components"] == 1 and stats["reciprocal_arc_frac"] == 1.0
    assert stats["route_continuity_breaks"] == 0
    assert stats["max_gap_s"] > 60
    assert stats["projection_max_rel_error"] < 0.01
    # Every fix except the 300 m spike is within ~3 m of the route.
    dist = gps_to_route_distance(trip)
    assert (dist < 5).sum() == len(dist) - 1


def test_run_eda_end_to_end(synthetic_root, tmp_path):
    out = tmp_path / "eda"
    run_eda.main(["--data-root", str(synthetic_root), "--output-dir", str(out)])
    for name in ("inventory.csv", "validation_issues.csv", "trip_statistics.csv", "data_quality_notes.md"):
        assert (out / name).exists()
    stats = pd.read_csv(out / "trip_statistics.csv", dtype={"trip_id": str})
    assert set(stats["trip_id"]) == {"00000001", "00000002"}
