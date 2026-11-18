"""Dataset inventory: which files exist, how big they are and what they look like.

This works on the raw text (not on parsed arrays) so it can report problems
such as missing files, inconsistent column counts or CRLF line endings that a
parser would silently hide.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

from src.common.config import ALLOWED_PARTITIONS, FILE_EXTENSIONS, PathLike
from src.common.io import list_trips, trip_file


def inspect_raw_file(path) -> Dict[str, object]:
    """Return existence, size, non-empty line count, column counts and line-ending style."""
    info: Dict[str, object] = {"exists": path.exists(), "size_bytes": 0, "n_lines": 0,
                               "column_counts": "", "has_crlf": False}
    if not path.exists():
        return info
    raw = path.read_bytes()
    info["size_bytes"] = len(raw)
    info["has_crlf"] = b"\r\n" in raw
    column_counts = set()
    n_lines = 0
    for line in raw.decode("utf-8", errors="replace").splitlines():
        tokens = line.split()
        if not tokens:
            continue  # blank lines are ignored
        n_lines += 1
        column_counts.add(len(tokens))
    info["n_lines"] = n_lines
    # e.g. "3" for a well-formed track, "2|3" would signal a malformed file.
    info["column_counts"] = "|".join(str(c) for c in sorted(column_counts))
    return info


def inventory_trip(partition: str, trip_id: str, data_root: Optional[PathLike] = None) -> Dict[str, object]:
    """One flat row describing all four files of a trip."""
    row: Dict[str, object] = {"partition": partition, "trip_id": trip_id}
    for ext in FILE_EXTENSIONS:
        info = inspect_raw_file(trip_file(partition, trip_id, ext, data_root))
        for key, value in info.items():
            row[f"{ext}_{key}"] = value
    row["all_files_present"] = all(row[f"{ext}_exists"] for ext in FILE_EXTENSIONS)
    return row


def build_inventory(partitions: Iterable[str] = ALLOWED_PARTITIONS,
                    data_root: Optional[PathLike] = None) -> pd.DataFrame:
    """Inventory every trip of the given partitions."""
    rows = [inventory_trip(partition, trip_id, data_root)
            for partition in partitions
            for trip_id in list_trips(partition, data_root)]
    return pd.DataFrame(rows)


def summarise_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Per-partition totals: trips, complete trips, observation/node/arc/route line counts, MB."""
    size_cols = [f"{ext}_size_bytes" for ext in FILE_EXTENSIONS]
    grouped = inventory.groupby("partition")
    summary = pd.DataFrame({
        "n_trips": grouped.size(),
        "n_complete_trips": grouped["all_files_present"].sum(),
        "total_gps_observations": grouped["track_n_lines"].sum(),
        "total_nodes": grouped["nodes_n_lines"].sum(),
        "total_arcs": grouped["arcs_n_lines"].sum(),
        "total_route_arcs": grouped["route_n_lines"].sum(),
        "total_size_mb": grouped[size_cols].sum().sum(axis=1) / 1e6,
        "trips_with_crlf_track": grouped["track_has_crlf"].sum(),
    })
    return summary.reset_index()
