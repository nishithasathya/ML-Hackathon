"""Per-trip descriptive statistics for the track, road network and reference route.

Column names produced by ``trip_statistics`` are a shared interface: the
visualisation module reads ``outputs/eda/trip_statistics.csv`` (see
docs/INTERFACES.md). Do not rename columns without telling the team.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.common.config import ALLOWED_PARTITIONS, PathLike
from src.common.geo import LocalProjection, haversine_m, point_segment_distance
from src.common.io import RawTrip, iter_trips


def _valid_track(trip: RawTrip) -> np.ndarray:
    track = trip.track
    return track[~np.isnan(track).any(axis=1)]


def track_statistics(trip: RawTrip) -> Dict[str, float]:
    track = _valid_track(trip)
    lon, lat, t = track[:, 0], track[:, 1], track[:, 2]
    stats: Dict[str, float] = {"n_observations": int(track.shape[0])}
    if track.shape[0] < 2:
        return stats
    dt = np.diff(t)
    step = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.nan)
    stats.update({
        "t_start": float(t[0]),
        "t_end": float(t[-1]),
        "duration_s": float(t[-1] - t[0]),
        "dt_median_s": float(np.median(dt)),
        "dt_mean_s": float(np.mean(dt)),
        "max_gap_s": float(np.max(dt)),
        "frac_dt_1s": float(np.mean(dt == 1)),
        "n_gaps_gt_5s": int(np.sum(dt > 5)),
        "n_gaps_gt_30s": int(np.sum(dt > 30)),
        "n_gaps_gt_300s": int(np.sum(dt > 300)),
        "n_duplicate_timestamps": int(np.sum(dt == 0)),
        "n_repeated_positions": int(np.sum(step == 0)),
        "path_length_km": float(step.sum() / 1000.0),
        "speed_median_mps": float(np.nanmedian(speed)),
        "speed_p95_mps": float(np.nanpercentile(speed, 95)),
        "speed_max_mps": float(np.nanmax(speed)),
        "n_steps_speed_gt_70mps": int(np.nansum(speed > 70)),
        "center_lon": float(lon.mean()),
        "center_lat": float(lat.mean()),
        "lon_min": float(lon.min()), "lon_max": float(lon.max()),
        "lat_min": float(lat.min()), "lat_max": float(lat.max()),
        "bbox_width_km": float(haversine_m(lon.min(), lat.mean(), lon.max(), lat.mean()) / 1000.0),
        "bbox_height_km": float(haversine_m(lon.mean(), lat.min(), lon.mean(), lat.max()) / 1000.0),
    })
    # How wrong is the local equirectangular projection for this trip?
    # Compare projected step lengths with haversine step lengths.
    proj = LocalProjection.from_points(lon, lat)
    x, y = proj.forward(lon, lat)
    proj_step = np.hypot(np.diff(x), np.diff(y))
    moving = step > 1.0
    stats["projection_max_rel_error"] = (float(np.max(np.abs(proj_step[moving] - step[moving]) / step[moving]))
                                         if moving.any() else 0.0)
    return stats


def network_statistics(trip: RawTrip) -> Dict[str, float]:
    nodes, arcs = trip.nodes, trip.arcs
    n_nodes, n_arcs = trip.n_nodes, trip.n_arcs
    stats: Dict[str, float] = {"n_nodes": n_nodes, "n_arcs": n_arcs}
    if n_arcs == 0 or n_nodes == 0:
        return stats
    graph = coo_matrix((np.ones(n_arcs), (arcs[:, 0], arcs[:, 1])), shape=(n_nodes, n_nodes)).tocsr()
    n_weak, weak_labels = connected_components(graph, directed=True, connection="weak")
    n_strong, strong_labels = connected_components(graph, directed=True, connection="strong")
    out_degree = np.bincount(arcs[:, 0], minlength=n_nodes)
    in_degree = np.bincount(arcs[:, 1], minlength=n_nodes)
    pair_set = set(map(tuple, arcs.tolist()))
    reciprocal = np.mean([(b, a) in pair_set for a, b in arcs.tolist()])
    arc_len = haversine_m(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1], nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1])
    stats.update({
        "n_isolated_nodes": int(np.sum((out_degree + in_degree) == 0)),
        "n_weak_components": int(n_weak),
        "largest_weak_component_frac": float(np.bincount(weak_labels).max() / n_nodes),
        "n_strong_components": int(n_strong),
        "largest_strong_component_frac": float(np.bincount(strong_labels).max() / n_nodes),
        "n_dead_end_nodes": int(np.sum((out_degree == 0) & (in_degree > 0))),
        "reciprocal_arc_frac": float(reciprocal),
        "n_self_loops": int(np.sum(arcs[:, 0] == arcs[:, 1])),
        "n_duplicate_arcs": int(n_arcs - len(pair_set)),
        "arc_length_median_m": float(np.median(arc_len)),
        "arc_length_max_m": float(np.max(arc_len)),
        "network_total_length_km": float(arc_len.sum() / 1000.0),
    })
    return stats


def route_statistics(trip: RawTrip) -> Dict[str, float]:
    route, arcs, nodes = trip.route, trip.arcs, trip.nodes
    stats: Dict[str, float] = {"n_route_arcs": int(route.size)}
    if route.size == 0:
        return stats
    ra = arcs[route]
    seg_len = haversine_m(nodes[ra[:, 0], 0], nodes[ra[:, 0], 1], nodes[ra[:, 1], 0], nodes[ra[:, 1], 1])
    stats.update({
        "n_unique_route_arcs": int(np.unique(route).size),
        "route_continuity_breaks": int(np.sum(ra[:-1, 1] != ra[1:, 0])) if route.size > 1 else 0,
        "route_length_km": float(seg_len.sum() / 1000.0),
    })
    return stats


def gps_to_route_distance(trip: RawTrip, chunk: int = 2000) -> np.ndarray:
    """Distance (m) from every valid GPS observation to the nearest reference-route arc.

    This measures GPS noise *relative to the reference route*; it is only
    possible for partitions where the route is supplied (train, student_test).
    """
    track = _valid_track(trip)
    if track.shape[0] == 0 or trip.route.size == 0:
        return np.empty(0)
    proj = LocalProjection.from_points(track[:, 0], track[:, 1])
    px, py = proj.forward(track[:, 0], track[:, 1])
    nx, ny = proj.forward(trip.nodes[:, 0], trip.nodes[:, 1])
    ra = trip.arcs[trip.route]
    ax, ay, bx, by = nx[ra[:, 0]], ny[ra[:, 0]], nx[ra[:, 1]], ny[ra[:, 1]]
    out = np.empty(px.size)
    for start in range(0, px.size, chunk):  # chunking bounds memory to chunk x n_route
        sl = slice(start, start + chunk)
        dist, _ = point_segment_distance(px[sl, None], py[sl, None], ax[None], ay[None], bx[None], by[None])
        out[sl] = dist.min(axis=1)
    return out


def trip_statistics(trip: RawTrip) -> Dict[str, float]:
    row: Dict[str, float] = {"partition": trip.partition, "trip_id": trip.trip_id}
    row.update(track_statistics(trip))
    row.update(network_statistics(trip))
    row.update(route_statistics(trip))
    dist = gps_to_route_distance(trip)
    if dist.size:
        row.update({
            "gps_route_dist_median_m": float(np.median(dist)),
            "gps_route_dist_p95_m": float(np.percentile(dist, 95)),
            "gps_route_dist_max_m": float(dist.max()),
            "frac_gps_gt_50m_from_route": float(np.mean(dist > 50)),
        })
    return row


def compute_trip_statistics(partitions: Iterable[str] = ALLOWED_PARTITIONS,
                            data_root: Optional[PathLike] = None, verbose: bool = False) -> pd.DataFrame:
    rows = []
    for partition in partitions:
        for trip in iter_trips(partition, data_root):
            if verbose:
                print(f"  statistics: {partition}/{trip.trip_id}", flush=True)
            rows.append(trip_statistics(trip))
    return pd.DataFrame(rows)


def partition_summary(stats: pd.DataFrame) -> pd.DataFrame:
    """describe() of the main numeric columns, per partition, in long format."""
    cols = ["n_observations", "duration_s", "path_length_km", "dt_median_s", "max_gap_s", "frac_dt_1s",
            "speed_p95_mps", "n_nodes", "n_arcs", "n_weak_components", "n_strong_components",
            "reciprocal_arc_frac", "n_route_arcs", "route_length_km", "gps_route_dist_median_m",
            "gps_route_dist_p95_m", "projection_max_rel_error"]
    cols = [c for c in cols if c in stats.columns]
    summary = stats.groupby("partition")[cols].describe().stack(level=0, future_stack=True) \
        if _pandas_supports_future_stack() else stats.groupby("partition")[cols].describe().stack(level=0)
    return summary.reset_index().rename(columns={"level_1": "statistic"})


def _pandas_supports_future_stack() -> bool:
    major, minor = (int(v) for v in pd.__version__.split(".")[:2])
    return (major, minor) >= (2, 1)
