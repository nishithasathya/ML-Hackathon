import numpy as np
import pytest

from src.common import config
from src.common.geo import LocalProjection, angle_diff_deg, bearing_from_xy, haversine_m, point_segment_distance
from src.common.io import DataFormatError, list_trips, load_trip, read_arcs


def test_haversine_one_degree_latitude():
    assert haversine_m(0, 0, 0, 1) == pytest.approx(111_195, rel=1e-3)


def test_projection_round_trip_and_scale():
    proj = LocalProjection(37.6, 55.75)
    x, y = proj.forward(37.61, 55.76)
    lon, lat = proj.inverse(x, y)
    assert lon == pytest.approx(37.61) and lat == pytest.approx(55.76)
    assert np.hypot(x, y) == pytest.approx(haversine_m(37.6, 55.75, 37.61, 55.76), rel=1e-3)


def test_point_segment_distance():
    d, t = point_segment_distance(5, 3, 0, 0, 10, 0)
    assert d == pytest.approx(3) and t == pytest.approx(0.5)
    d, t = point_segment_distance(-4, 3, 0, 0, 10, 0)  # beyond end A -> distance to A
    assert d == pytest.approx(5) and t == 0
    d, _ = point_segment_distance(3, 4, 0, 0, 0, 0)  # zero-length segment
    assert d == pytest.approx(5)


def test_bearings():
    assert bearing_from_xy(0, 1) == pytest.approx(0)
    assert bearing_from_xy(1, 0) == pytest.approx(90)
    assert angle_diff_deg(350, 10) == pytest.approx(20)
    assert angle_diff_deg(0, 180) == pytest.approx(180)


def test_evaluator_partition_is_refused(synthetic_root):
    with pytest.raises(ValueError):
        config.check_partition("evaluator_test")
    with pytest.raises(ValueError):
        list_trips("evaluator_test", synthetic_root)


def test_load_trip_reads_crlf_and_shapes(synthetic_root):
    assert list_trips("train", synthetic_root) == ["00000001"]
    trip = load_trip("train", "00000001", synthetic_root)
    assert trip.track.shape[1] == 3 and trip.nodes.shape == (9, 2)
    assert trip.arcs.dtype.kind == "i" and trip.route.ndim == 1
    assert not np.isnan(trip.track).any()


def test_read_arcs_rejects_non_integer(tmp_path):
    bad = tmp_path / "x.arcs"
    bad.write_text("0\t1.5\n")
    with pytest.raises(DataFormatError):
        read_arcs(bad)
