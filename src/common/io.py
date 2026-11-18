"""Raw-file readers for the PathSnap dataset.

Verified file formats (checked against all 90 permitted trips):

* ``<id>.track`` - 3 whitespace-separated columns, no header:
  longitude, latitude, timestamp (seconds). Files use CRLF line endings.
* ``<id>.nodes`` - 2 columns: longitude, latitude. Row index = node id (0-based).
* ``<id>.arcs``  - 2 integer columns: from_node_id, to_node_id. Row index = arc id.
* ``<id>.route`` - 1 integer column: chronological sequence of reference arc ids.

Every trip has its OWN road sub-network, so node/arc ids are only meaningful
inside the trip they come from.

These readers do no cleaning. Unparseable values in .track/.nodes become NaN so
that validation and preprocessing can count and handle them. Arc and route
files must be integers because their row indices are identifiers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

from src.common.config import FILE_EXTENSIONS, PathLike, check_partition, get_data_root


class DataFormatError(ValueError):
    """Raised when a raw file does not have the documented structure."""


@dataclass
class RawTrip:
    """All raw information for one trip, exactly as read from disk."""

    trip_id: str
    partition: str
    track: np.ndarray  # (n_obs, 3) float: lon, lat, t
    nodes: np.ndarray  # (n_nodes, 2) float: lon, lat
    arcs: np.ndarray  # (n_arcs, 2) int: from_node, to_node
    route: np.ndarray  # (n_route,) int: arc ids in travel order
    directory: Path

    @property
    def n_observations(self) -> int:
        return int(self.track.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_arcs(self) -> int:
        return int(self.arcs.shape[0])


def _read_numeric_table(path: Path, n_columns: int) -> np.ndarray:
    """Read a header-less whitespace-separated numeric file into a float array.

    Non-numeric tokens are converted to NaN. An empty file returns shape (0, n_columns).
    """
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return np.empty((0, n_columns), dtype=float)
    # sep=r"\s+" also treats the '\r' of CRLF endings as whitespace.
    frame = pd.read_csv(path, sep=r"\s+", header=None, dtype=str, engine="c")
    if frame.shape[1] != n_columns:
        raise DataFormatError(f"{path.name}: expected {n_columns} columns, found {frame.shape[1]}")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.to_numpy(dtype=float)


def _to_int_array(values: np.ndarray, path: Path) -> np.ndarray:
    if np.isnan(values).any():
        raise DataFormatError(f"{path.name}: contains non-numeric values")
    if not np.all(values == np.round(values)):
        raise DataFormatError(f"{path.name}: contains non-integer ids")
    return values.astype(np.int64)


def read_track(path: PathLike) -> np.ndarray:
    """Return (n, 3) float array of lon, lat, timestamp."""
    return _read_numeric_table(Path(path), 3)


def read_nodes(path: PathLike) -> np.ndarray:
    """Return (n, 2) float array of lon, lat. Row index is the node id."""
    return _read_numeric_table(Path(path), 2)


def read_arcs(path: PathLike) -> np.ndarray:
    """Return (n, 2) int array of from_node, to_node. Row index is the arc id."""
    path = Path(path)
    return _to_int_array(_read_numeric_table(path, 2), path)


def read_route(path: PathLike) -> np.ndarray:
    """Return 1-D int array of arc ids in the order they were traversed."""
    path = Path(path)
    return _to_int_array(_read_numeric_table(path, 1), path).ravel()


def partition_dir(partition: str, data_root: Optional[PathLike] = None) -> Path:
    return get_data_root(data_root) / check_partition(partition)


def trip_dir(partition: str, trip_id: str, data_root: Optional[PathLike] = None) -> Path:
    return partition_dir(partition, data_root) / trip_id


def trip_file(partition: str, trip_id: str, extension: str, data_root: Optional[PathLike] = None) -> Path:
    if extension not in FILE_EXTENSIONS:
        raise ValueError(f"Unknown extension '{extension}'. Expected one of {FILE_EXTENSIONS}")
    return trip_dir(partition, trip_id, data_root) / f"{trip_id}.{extension}"


def list_trips(partition: str, data_root: Optional[PathLike] = None) -> List[str]:
    """Return the sorted trip ids (sub-directory names) of a partition."""
    folder = partition_dir(partition, data_root)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Partition folder not found: {folder}. Copy the dataset into data/raw/ "
            f"or set PATHSNAP_DATA_ROOT."
        )
    return sorted(p.name for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))


def load_trip(partition: str, trip_id: str, data_root: Optional[PathLike] = None) -> RawTrip:
    """Load the four raw files of one trip."""
    return RawTrip(
        trip_id=trip_id,
        partition=partition,
        track=read_track(trip_file(partition, trip_id, "track", data_root)),
        nodes=read_nodes(trip_file(partition, trip_id, "nodes", data_root)),
        arcs=read_arcs(trip_file(partition, trip_id, "arcs", data_root)),
        route=read_route(trip_file(partition, trip_id, "route", data_root)),
        directory=trip_dir(partition, trip_id, data_root),
    )


def iter_trips(partition: str, data_root: Optional[PathLike] = None,
               trip_ids: Optional[List[str]] = None) -> Iterator[RawTrip]:
    """Yield RawTrip objects one at a time (keeps memory low for big networks)."""
    for trip_id in trip_ids if trip_ids is not None else list_trips(partition, data_root):
        yield load_trip(partition, trip_id, data_root)
