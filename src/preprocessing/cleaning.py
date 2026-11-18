"""Step-by-step cleaning functions for a GPS track.

Design principle: we keep the ORIGINAL row number of every observation in the
``obs_idx`` column. Later phases must return one prediction per relevant GPS
observation, and the evaluator compares against the original rows, so rows are
only dropped when they are physically unusable (invalid coordinates, missing
or duplicate timestamps). Suspicious points (speed spikes) are FLAGGED, not
removed, by default.

Every function takes a DataFrame and returns a new DataFrame plus a count of
what it changed, so the pipeline can build a per-trip cleaning report.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from src.common.geo import LocalProjection

TRACK_COLUMNS = ["obs_idx", "lon", "lat", "t"]


def track_to_frame(track: np.ndarray) -> pd.DataFrame:
    """Convert the raw (n, 3) track array into a DataFrame with original row ids."""
    frame = pd.DataFrame(track, columns=["lon", "lat", "t"])
    frame.insert(0, "obs_idx", np.arange(len(frame), dtype=np.int64))
    return frame


def drop_invalid_coordinates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop rows with NaN, out-of-range or exactly (0, 0) coordinates.

    (0, 0) is a common "no fix" placeholder in GPS loggers; no trip in this
    dataset is near that point, so treating it as invalid is safe.
    """
    lon, lat = df["lon"], df["lat"]
    invalid = (lon.isna() | lat.isna() | (lon.abs() > 180) | (lat.abs() > 90) | ((lon == 0) & (lat == 0)))
    return df.loc[~invalid].copy(), int(invalid.sum())


def enforce_time_order(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int, int]:
    """Drop missing timestamps, sort by time (stable) and drop duplicate timestamps.

    Returns ``(df, n_missing_time, n_out_of_order, n_duplicate_time)``.
    For duplicates the first occurrence is kept: two fixes with the same time
    cannot both be used to compute a speed.
    """
    missing = df["t"].isna()
    df = df.loc[~missing]
    n_out_of_order = int((df["t"].diff() < 0).sum())
    df = df.sort_values("t", kind="mergesort")  # mergesort is stable: keeps file order for ties
    duplicate = df["t"].duplicated(keep="first")
    return df.loc[~duplicate].reset_index(drop=True), int(missing.sum()), n_out_of_order, int(duplicate.sum())


def add_projected_coordinates(df: pd.DataFrame, projection: LocalProjection) -> pd.DataFrame:
    """Add x, y in metres (local equirectangular projection, see src/common/geo.py)."""
    df = df.copy()
    df["x"], df["y"] = projection.forward(df["lon"].to_numpy(), df["lat"].to_numpy())
    return df


def add_time_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``t_rel`` (seconds since the first kept fix) and ``dt_prev`` (NaN for the first row)."""
    df = df.copy()
    df["t_rel"] = df["t"] - df["t"].iloc[0] if len(df) else df["t"]
    df["dt_prev"] = df["t"].diff()
    return df


def assign_gap_segments(df: pd.DataFrame, gap_threshold_s: float) -> Tuple[pd.DataFrame, int]:
    """Split the track into continuous segments wherever ``dt_prev > gap_threshold_s``.

    Speeds and headings must not be computed across a long gap, and a later
    HMM will need to know where the observation chain is broken.
    Returns ``(df, n_gaps)``.
    """
    df = df.copy()
    is_gap = (df["dt_prev"] > gap_threshold_s).fillna(False)
    df["is_segment_start"] = is_gap.copy()
    if len(df):
        df.loc[df.index[0], "is_segment_start"] = True
    df["segment_id"] = df["is_segment_start"].cumsum().astype(np.int64) - 1
    return df, int(is_gap.sum())


def flag_speed_outliers(df: pd.DataFrame, max_speed_mps: float) -> Tuple[pd.DataFrame, int]:
    """Flag isolated position spikes.

    A point is flagged when the implied speed both INTO it and OUT OF it
    (within the same segment) exceeds ``max_speed_mps``. Requiring both sides
    avoids flagging the legitimate point after a single jump, and catches the
    typical "one fix thrown off by multipath" pattern.
    Requires columns x, y, t, segment_id. Returns ``(df, n_flagged)``.
    """
    df = df.copy()
    same_seg_prev = df["segment_id"].eq(df["segment_id"].shift(1))
    same_seg_next = df["segment_id"].eq(df["segment_id"].shift(-1))
    step_in = np.hypot(df["x"].diff(), df["y"].diff())
    dt_in = df["t"].diff()
    speed_in = (step_in / dt_in).where(same_seg_prev)
    speed_out = speed_in.shift(-1).where(same_seg_next)
    df["speed_in_mps"] = speed_in
    df["is_speed_outlier"] = (speed_in > max_speed_mps) & (speed_out > max_speed_mps)
    return df, int(df["is_speed_outlier"].sum())
