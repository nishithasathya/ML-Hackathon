"""Rule-based data-quality checks for one trip.

Each check produces a ``ValidationIssue`` only when something is found, so an
empty list means the trip passed every check. Severity levels:

* ``error``   - the file cannot be used as-is (e.g. arc refers to a missing node).
* ``warning`` - usable, but needs handling in preprocessing (e.g. duplicate timestamps).
* ``info``    - a property worth knowing (e.g. trip starts at t != 1).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from src.common.config import ALLOWED_PARTITIONS, PathLike
from src.common.io import RawTrip, iter_trips

# Gap threshold used only for *reporting* here. Preprocessing has its own
# configurable threshold (see src/preprocessing/pipeline.py).
REPORT_GAP_S = 30.0


@dataclass
class ValidationIssue:
    partition: str
    trip_id: str
    file: str
    severity: str
    code: str
    count: int
    message: str


def _invalid_coordinate_mask(lonlat: np.ndarray) -> np.ndarray:
    lon, lat = lonlat[:, 0], lonlat[:, 1]
    return (np.isnan(lon) | np.isnan(lat) | (np.abs(lon) > 180) | (np.abs(lat) > 90)
            | ((lon == 0) & (lat == 0)))


def validate_trip(trip: RawTrip, gap_threshold_s: float = REPORT_GAP_S) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    def add(file: str, severity: str, code: str, count: int, message: str) -> None:
        if count:
            issues.append(ValidationIssue(trip.partition, trip.trip_id, file, severity, code, int(count), message))

    # ---------- .track ----------
    track = trip.track
    if track.shape[0] == 0:
        add("track", "error", "EMPTY_TRACK", 1, "track file has no observations")
    else:
        add("track", "warning", "INVALID_COORD", _invalid_coordinate_mask(track[:, :2]).sum(),
            "NaN, out-of-range or (0,0) coordinates")
        t = track[:, 2]
        add("track", "warning", "NAN_TIMESTAMP", np.isnan(t).sum(), "missing timestamps")
        dt = np.diff(t)
        add("track", "warning", "NON_INCREASING_TIME", (dt < 0).sum(), "timestamp decreases")
        add("track", "warning", "DUPLICATE_TIMESTAMP", (dt == 0).sum(), "consecutive equal timestamps")
        add("track", "info", "TIME_GAP", (dt > gap_threshold_s).sum(),
            f"intervals longer than {gap_threshold_s:g}s (largest {np.nanmax(dt) if dt.size else 0:g}s)")
        add("track", "info", "NON_INTEGER_TIME", (t != np.round(t)).sum(), "fractional timestamps")
        if not np.isnan(t[0]) and t[0] != 1:
            add("track", "info", "START_TIME_NOT_1", 1, f"first timestamp is {t[0]:g}")
        same_pos = (np.diff(track[:, 0]) == 0) & (np.diff(track[:, 1]) == 0)
        add("track", "info", "REPEATED_POSITION", same_pos.sum(), "consecutive identical positions")

    # ---------- .nodes ----------
    nodes = trip.nodes
    add("nodes", "error", "INVALID_NODE_COORD", _invalid_coordinate_mask(nodes).sum() if nodes.size else 0,
        "NaN or out-of-range node coordinates")
    if nodes.size:
        add("nodes", "info", "DUPLICATE_NODE_COORD",
            int(pd.DataFrame(nodes).duplicated().sum()), "distinct node ids share coordinates")

    # ---------- .arcs ----------
    arcs = trip.arcs
    n_nodes = trip.n_nodes
    if arcs.size:
        bad = (arcs < 0) | (arcs >= n_nodes)
        add("arcs", "error", "ARC_NODE_OUT_OF_RANGE", bad.any(axis=1).sum(), "arc refers to a non-existent node")
        add("arcs", "info", "SELF_LOOP", (arcs[:, 0] == arcs[:, 1]).sum(), "arc starts and ends at same node")
        add("arcs", "info", "DUPLICATE_ARC", int(pd.DataFrame(arcs).duplicated().sum()),
            "same (from,to) pair appears under several arc ids")
        degree = np.bincount(arcs.ravel()[(arcs.ravel() >= 0) & (arcs.ravel() < n_nodes)], minlength=n_nodes)
        add("nodes", "info", "ISOLATED_NODE", (degree == 0).sum(), "node not used by any arc")
    else:
        add("arcs", "error", "EMPTY_ARCS", 1, "arc file is empty")

    # ---------- .route ----------
    route = trip.route
    if route.size == 0:
        add("route", "warning", "EMPTY_ROUTE", 1, "no reference route")
    else:
        out_of_range = (route < 0) | (route >= trip.n_arcs)
        add("route", "error", "ROUTE_ARC_OUT_OF_RANGE", out_of_range.sum(), "route refers to a non-existent arc")
        valid = route[~out_of_range]
        if valid.size > 1:
            breaks = arcs[valid[:-1], 1] != arcs[valid[1:], 0]
            add("route", "warning", "ROUTE_DISCONTINUITY", breaks.sum(),
                "consecutive route arcs are not connected (to_node != next from_node)")
            add("route", "info", "ROUTE_REPEATED_ARC", valid.size - np.unique(valid).size,
                "route traverses the same arc more than once")

    return issues


def validate_partitions(partitions: Iterable[str] = ALLOWED_PARTITIONS,
                        data_root: Optional[PathLike] = None) -> pd.DataFrame:
    rows = []
    for partition in partitions:
        for trip in iter_trips(partition, data_root):
            rows.extend(asdict(issue) for issue in validate_trip(trip))
    columns = list(ValidationIssue.__dataclass_fields__)
    return pd.DataFrame(rows, columns=columns)
