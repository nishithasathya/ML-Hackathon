"""Preprocessing pipeline: RawTrip -> PreprocessedTrip.

``preprocess_trip`` is the main shared interface of this module. The feature
and visualisation modules call it; they must not re-implement cleaning.

Decisions are documented next to each default in ``PreprocessConfig``.
Nothing here is fitted to data (no means/scalers), so there is no risk of
leaking student-test information into training. Learned transformations
(e.g. feature scaling) belong to the model-training phase and must be fitted
on training trips only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from src.common.geo import LocalProjection
from src.common.io import RawTrip
from src.preprocessing.cleaning import (add_projected_coordinates, add_time_deltas, assign_gap_segments,
                                        drop_invalid_coordinates, enforce_time_order, flag_speed_outliers,
                                        track_to_frame)


@dataclass
class PreprocessConfig:
    # The data is nominally 1 Hz (EDA: most intervals are exactly 1 s). A gap
    # above 30 s means the vehicle may have passed several road segments
    # unobserved, so the track is split there.
    gap_threshold_s: float = 30.0
    # 70 m/s = 252 km/h: above any plausible road speed, so a fix implying it
    # on both sides is treated as a GPS spike.
    max_speed_mps: float = 70.0
    # Flag only by default: every observation should still get a prediction later.
    drop_speed_outliers: bool = False


@dataclass
class PreprocessedTrip:
    trip_id: str
    partition: str
    # Columns: obs_idx, lon, lat, t, x, y, t_rel, dt_prev, is_segment_start,
    #          segment_id, speed_in_mps, is_speed_outlier
    track: pd.DataFrame
    nodes_lonlat: np.ndarray  # (n_nodes, 2) unchanged from raw
    nodes_xy: np.ndarray  # (n_nodes, 2) metres in the same projection as the track
    arcs: np.ndarray  # (n_arcs, 2) unchanged: arc ids must stay aligned with .route
    route: np.ndarray  # reference arc ids (labels only - never use as a model input)
    projection: LocalProjection
    report: Dict[str, float] = field(default_factory=dict)


def preprocess_trip(raw: RawTrip, config: PreprocessConfig = PreprocessConfig()) -> PreprocessedTrip:
    report: Dict[str, float] = {"partition": raw.partition, "trip_id": raw.trip_id,
                                "n_raw_observations": raw.n_observations}

    df = track_to_frame(raw.track)
    df, report["n_dropped_invalid_coord"] = drop_invalid_coordinates(df)
    df, n_missing_t, n_unordered, n_dup = enforce_time_order(df)
    report["n_dropped_missing_time"] = n_missing_t
    report["n_out_of_order"] = n_unordered
    report["n_dropped_duplicate_time"] = n_dup
    if df.empty:
        raise ValueError(f"Trip {raw.trip_id}: no valid observations after cleaning")

    # One projection per trip, centred on the cleaned track; nodes use the same one.
    projection = LocalProjection.from_points(df["lon"].to_numpy(), df["lat"].to_numpy())
    df = add_projected_coordinates(df, projection)
    df = add_time_deltas(df)
    df, report["n_gaps"] = assign_gap_segments(df, config.gap_threshold_s)
    df, report["n_speed_outliers"] = flag_speed_outliers(df, config.max_speed_mps)
    if config.drop_speed_outliers:
        df = df.loc[~df["is_speed_outlier"]].reset_index(drop=True)
    report["n_segments"] = int(df["segment_id"].nunique())
    report["n_clean_observations"] = int(len(df))
    report["n_fractional_timestamps"] = int((df["t"] != np.round(df["t"])).sum())

    nx, ny = projection.forward(raw.nodes[:, 0], raw.nodes[:, 1])
    return PreprocessedTrip(
        trip_id=raw.trip_id, partition=raw.partition, track=df.reset_index(drop=True),
        nodes_lonlat=raw.nodes, nodes_xy=np.column_stack([nx, ny]), arcs=raw.arcs, route=raw.route,
        projection=projection, report=report,
    )


def config_as_dict(config: PreprocessConfig) -> Dict[str, object]:
    return asdict(config)
