"""Static matplotlib figures for the first-phase report.

Inputs are the CSV files written by the other modules (see docs/INTERFACES.md):
    outputs/eda/trip_statistics.csv     (data_exploration.run_eda)
    outputs/eda/validation_issues.csv   (data_exploration.run_eda)
    outputs/features/train/radius_sweep.csv (features.run_features --sweep)
plus the raw tracks for the sampling-interval figure. Every figure therefore
shows numbers that were actually computed from the dataset.

Always read CSVs through ``read_csv_keep_ids`` so trip ids such as
"00000010" keep their leading zeros.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")  # file output only; works on machines without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import ALLOWED_PARTITIONS, PathLike
from src.common.geo import LocalProjection
from src.common.io import RawTrip, list_trips, read_track, trip_file

PARTITION_COLOURS = {"train": "#2a78d6", "student_test": "#eb6834"}
SINGLE_COLOUR = "#2a78d6"
MAP_COLOURS = {"network": "#b0b0b0", "route": "#2e7d32", "gps": "#1565c0"}


def read_csv_keep_ids(path: PathLike) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"trip_id": str})


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_trip_static(raw: RawTrip, path: Path, network_buffer_m: float = 150.0) -> Path:
    """GPS points over the nearby network and reference route, in local metres (equal aspect)."""
    from src.visualization.folium_maps import nearby_arc_ids  # same network subset as the Folium map

    proj = LocalProjection.from_points(raw.track[:, 0], raw.track[:, 1])
    nx, ny = proj.forward(raw.nodes[:, 0], raw.nodes[:, 1])
    gx, gy = proj.forward(raw.track[:, 0], raw.track[:, 1])
    fig, ax = plt.subplots(figsize=(7, 7))
    for arc_ids, colour, width, label in ((nearby_arc_ids(raw, network_buffer_m), MAP_COLOURS["network"], 0.5,
                                           "Road network (near track)"),
                                          (raw.route, MAP_COLOURS["route"], 2.5, "Reference route")):
        a, b = raw.arcs[arc_ids, 0], raw.arcs[arc_ids, 1]
        xs = np.column_stack([nx[a], nx[b], np.full(a.size, np.nan)]).ravel()
        ys = np.column_stack([ny[a], ny[b], np.full(a.size, np.nan)]).ravel()
        ax.plot(xs, ys, color=colour, linewidth=width, label=label)
    ax.scatter(gx, gy, s=6, color=MAP_COLOURS["gps"], label="GPS observations", zorder=3)
    ax.set_aspect("equal")
    pad = 200
    ax.set_xlim(np.nanmin(gx) - pad, np.nanmax(gx) + pad)
    ax.set_ylim(np.nanmin(gy) - pad, np.nanmax(gy) + pad)
    _style(ax, f"Trip {raw.trip_id} ({raw.partition}): GPS vs reference route", "east (m)", "north (m)")
    ax.legend(loc="best", fontsize=8, frameon=True)
    return _save(fig, path)


def plot_trip_locations(stats: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for partition, group in stats.groupby("partition"):
        ax.scatter(group["center_lon"], group["center_lat"], s=28, color=PARTITION_COLOURS.get(partition, "#666"),
                   label=f"{partition} ({len(group)} trips)", edgecolor="white", linewidth=0.8)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    _style(ax, "Trip locations (centre of each GPS track)", "longitude", "latitude")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_distribution(stats: pd.DataFrame, column: str, path: Path, title: str, xlabel: str,
                      log_x: bool = False, bins: int = 30) -> Path:
    """Histogram of one per-trip statistic, partitions side by side."""
    values = stats[column].dropna()
    positive = values[values > 0]
    if log_x and len(positive):
        edges = np.logspace(np.log10(positive.min()), np.log10(positive.max()), bins)
    else:
        edges = np.linspace(values.min(), values.max(), bins)
    fig, ax = plt.subplots(figsize=(7, 4))
    series = [stats.loc[stats["partition"] == p, column].dropna() for p in ALLOWED_PARTITIONS]
    ax.hist(series, bins=edges, color=[PARTITION_COLOURS[p] for p in ALLOWED_PARTITIONS],
            label=list(ALLOWED_PARTITIONS), edgecolor="white", linewidth=0.5)
    if log_x:
        ax.set_xscale("log")
    _style(ax, title, xlabel, "number of trips")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_network_size(stats: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for partition, group in stats.groupby("partition"):
        ax.scatter(group["n_nodes"], group["n_arcs"], s=22, color=PARTITION_COLOURS.get(partition, "#666"),
                   label=partition, edgecolor="white", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "Road-network size per trip", "nodes", "arcs")
    ax.legend(fontsize=8)
    return _save(fig, path)


def interval_category_counts(partitions: Iterable[str] = ALLOWED_PARTITIONS,
                             data_root: Optional[PathLike] = None) -> pd.DataFrame:
    """Count GPS sampling intervals per category, per partition, from the raw tracks."""
    bins = [-np.inf, 0, 1.5, 2.5, 5, 30, 300, np.inf]
    labels = ["<=0 s", "1 s", "2 s", "3-5 s", "6-30 s", "31-300 s", ">300 s"]
    rows = []
    for partition in partitions:
        dts = [np.diff(read_track(trip_file(partition, t, "track", data_root))[:, 2])
               for t in list_trips(partition, data_root)]
        counts = pd.cut(np.concatenate(dts), bins=bins, labels=labels).value_counts().reindex(labels)
        rows.append(counts.rename(partition))
    return pd.DataFrame(rows).T


def plot_sampling_intervals(counts: pd.DataFrame, path: Path) -> Path:
    share = counts / counts.sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.4
    xs = np.arange(len(share.index))
    for k, partition in enumerate(share.columns):
        ax.bar(xs + (k - 0.5) * width, share[partition], width=width - 0.03,
               color=PARTITION_COLOURS.get(partition, "#666"), label=partition)
    ax.set_xticks(xs, share.index)
    ax.set_yscale("log")
    _style(ax, "GPS sampling intervals (share of all intervals, log scale)", "interval between fixes", "share")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_validation_issues(issues: pd.DataFrame, path: Path) -> Path:
    """Number of trips affected by each validation issue code (missing/invalid-value analysis)."""
    counts = issues.groupby("code")["trip_id"].nunique().sort_values()
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(counts) + 1)))
    ax.barh(counts.index, counts.values, color=SINGLE_COLOUR, height=0.6)
    for y, v in enumerate(counts.values):
        ax.text(v + 0.5, y, str(v), va="center", fontsize=8, color="#333")
    _style(ax, "Data-quality checks: trips affected per issue", "number of trips", "")
    return _save(fig, path)


def plot_radius_sweep(sweep: pd.DataFrame, path: Path) -> Path:
    """Candidate miss rate vs radius (from run_features --sweep), one line per subset."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colours = ["#2a78d6", "#eb6834"]
    for colour, (subset, group) in zip(colours, sweep.groupby("subset", sort=False)):
        ax.plot(group["radius_m"], group["candidate_miss_rate"] * 100, marker="o", color=colour, linewidth=2,
                label=subset)
        for r, m, c in zip(group["radius_m"], group["candidate_miss_rate"] * 100, group["mean_candidates_per_obs"]):
            ax.annotate(f"{c:.1f} cand.", (r, m), textcoords="offset points", xytext=(4, 6), fontsize=7, color="#555")
    _style(ax, "Candidate miss rate vs search radius (train trips)", "radius (m)", "miss rate (%)")
    ax.legend(fontsize=8)
    return _save(fig, path)
