"""Shared pytest fixtures.

Tests use a tiny SYNTHETIC trip written to a temporary folder, so they run
without the real dataset and finish in seconds. The synthetic trip is a
3 x 3 street grid (100 m spacing) with a known route and a GPS track that has
a known spike, a known time gap and a known duplicate timestamp - so every
check has a correct answer we can assert. It is test scaffolding only and is
never used as evidence about the real data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

LON0, LAT0 = 10.0, 50.0
M_PER_DEG_LAT = 111_195.0
DLAT = 100.0 / M_PER_DEG_LAT  # 100 m north
DLON = 100.0 / (M_PER_DEG_LAT * np.cos(np.radians(LAT0)))  # 100 m east

SPIKE_ROW = 5  # raw row index of the GPS spike
DUPLICATE_ROW = 12  # raw row index that repeats the previous timestamp
GAP_AFTER_ROW = 25  # a 60 s gap starts after this raw row
GAP_SECONDS = 60


def node_id(col: int, row: int) -> int:
    return row * 3 + col


def build_grid():
    nodes = np.array([[LON0 + c * DLON, LAT0 + r * DLAT] for r in range(3) for c in range(3)])
    arcs = []
    for r in range(3):
        for c in range(3):
            if c < 2:
                arcs += [(node_id(c, r), node_id(c + 1, r)), (node_id(c + 1, r), node_id(c, r))]
            if r < 2:
                arcs += [(node_id(c, r), node_id(c, r + 1)), (node_id(c, r + 1), node_id(c, r))]
    return nodes, np.array(arcs)


def arc_id(arcs: np.ndarray, a: int, b: int) -> int:
    return int(np.flatnonzero((arcs[:, 0] == a) & (arcs[:, 1] == b))[0])


def build_track():
    """Drive east along row 0 (0 -> 200 m), then north along column 2, 10 m per second, 3 m GPS offset."""
    xy = [(x, 3.0) for x in range(0, 201, 10)] + [(203.0, y) for y in range(10, 101, 10)]
    xy = np.array(xy, dtype=float)
    xy[SPIKE_ROW, 1] = 300.0  # one fix thrown 300 m north
    t = np.arange(1, len(xy) + 1, dtype=float)
    t[GAP_AFTER_ROW + 1:] += GAP_SECONDS
    t[DUPLICATE_ROW] = t[DUPLICATE_ROW - 1]
    lon = LON0 + xy[:, 0] / (M_PER_DEG_LAT * np.cos(np.radians(LAT0)))
    lat = LAT0 + xy[:, 1] / M_PER_DEG_LAT
    return np.column_stack([lon, lat, t])


def write_trip(root: Path, partition: str, trip_id: str) -> None:
    nodes, arcs = build_grid()
    route = [arc_id(arcs, node_id(0, 0), node_id(1, 0)), arc_id(arcs, node_id(1, 0), node_id(2, 0)),
             arc_id(arcs, node_id(2, 0), node_id(2, 1))]
    folder = root / partition / trip_id
    folder.mkdir(parents=True)
    # CRLF line endings, like the real .track files
    (folder / f"{trip_id}.track").write_bytes(
        "".join(f"{lo:.6f}\t{la:.6f}\t{t:.6f}\r\n" for lo, la, t in build_track()).encode())
    (folder / f"{trip_id}.nodes").write_text("".join(f"{lo:.8f}\t{la:.8f}\n" for lo, la in nodes))
    (folder / f"{trip_id}.arcs").write_text("".join(f"{a}\t{b}\n" for a, b in arcs))
    (folder / f"{trip_id}.route").write_text("".join(f"{r}\n" for r in route))


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    write_trip(tmp_path, "train", "00000001")
    write_trip(tmp_path, "student_test", "00000002")
    return tmp_path


@pytest.fixture
def grid():
    nodes, arcs = build_grid()
    return {"nodes": nodes, "arcs": arcs,
            "route": [arc_id(arcs, node_id(0, 0), node_id(1, 0)), arc_id(arcs, node_id(1, 0), node_id(2, 0)),
                      arc_id(arcs, node_id(2, 0), node_id(2, 1))],
            "arc_id": lambda a, b: arc_id(arcs, a, b), "node_id": node_id}
