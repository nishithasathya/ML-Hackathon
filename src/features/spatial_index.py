"""Spatial index over road arcs, used to find candidate arcs near a GPS fix.

Problem: a trip network can have >300,000 arcs, and checking every arc for
every GPS point is far too slow. Arcs are line segments (some several km long),
so indexing only their endpoints or midpoints would miss long arcs.

Approach: sample points along every arc at most ``sample_spacing_m`` apart and
put the samples in a KD-tree. For a GPS point P and search radius r, any arc
whose closest point to P is within r has a sample within r + spacing/2 of P
(the closest point lies between two samples that are <= spacing apart). So
querying the tree with r + spacing/2 never misses an arc; the exact
point-to-segment distance is then computed only for those arcs.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.spatial import cKDTree

from src.common.geo import bearing_from_xy, point_segment_distance


class ArcIndex:
    def __init__(self, nodes_xy: np.ndarray, arcs: np.ndarray, sample_spacing_m: float = 20.0):
        if sample_spacing_m <= 0:
            raise ValueError("sample_spacing_m must be positive")
        self.sample_spacing_m = float(sample_spacing_m)
        self.arcs = arcs
        self.ax = nodes_xy[arcs[:, 0], 0]
        self.ay = nodes_xy[arcs[:, 0], 1]
        self.bx = nodes_xy[arcs[:, 1], 0]
        self.by = nodes_xy[arcs[:, 1], 1]
        self.length_m = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing_deg = bearing_from_xy(self.bx - self.ax, self.by - self.ay)

        # Number of samples per arc: both endpoints plus enough interior points.
        n_samples = np.ceil(self.length_m / self.sample_spacing_m).astype(np.int64) + 1
        self.sample_arc = np.repeat(np.arange(len(arcs)), n_samples)
        offsets = np.concatenate([[0], np.cumsum(n_samples)[:-1]])
        position = np.arange(self.sample_arc.size) - np.repeat(offsets, n_samples)
        frac = position / np.maximum(np.repeat(n_samples, n_samples) - 1, 1)
        sx = self.ax[self.sample_arc] + frac * (self.bx - self.ax)[self.sample_arc]
        sy = self.ay[self.sample_arc] + frac * (self.by - self.ay)[self.sample_arc]
        self.tree = cKDTree(np.column_stack([sx, sy]))

    @property
    def n_arcs(self) -> int:
        return int(self.arcs.shape[0])

    def arcs_within(self, x: np.ndarray, y: np.ndarray, radius_m: float) -> List[np.ndarray]:
        """For each point, the ids of all arcs whose segment is within ``radius_m`` (exact)."""
        query_r = radius_m + self.sample_spacing_m / 2.0
        hits = self.tree.query_ball_point(np.column_stack([x, y]), r=query_r)
        result: List[np.ndarray] = []
        for px, py, sample_ids in zip(x, y, hits):
            if not sample_ids:
                result.append(np.empty(0, dtype=np.int64))
                continue
            arc_ids = np.unique(self.sample_arc[sample_ids])
            dist, _ = point_segment_distance(px, py, self.ax[arc_ids], self.ay[arc_ids],
                                             self.bx[arc_ids], self.by[arc_ids])
            result.append(arc_ids[dist <= radius_m])
        return result

    def distance_to_arcs(self, x: float, y: float, arc_ids: np.ndarray):
        """Exact distance (m) and position along the arc (0..1) from one point to given arcs."""
        return point_segment_distance(x, y, self.ax[arc_ids], self.ay[arc_ids], self.bx[arc_ids], self.by[arc_ids])
