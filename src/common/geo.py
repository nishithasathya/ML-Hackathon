"""Geographic helpers shared by every module.

Why not plain Euclidean distance on degrees?
    Longitude/latitude are angles. One degree of longitude shrinks with
    cos(latitude) (about 0.57 at 55 N, e.g. Moscow, where many trips are), so
    Euclidean distance in degrees distorts east-west distances badly.

What we use instead:
    * ``haversine_m`` - great-circle distance, used for reporting lengths.
    * ``LocalProjection`` - an equirectangular projection centred on each trip.
      It turns lon/lat into metres (x east, y north) so that point-to-segment
      distances, headings and spatial indexing are simple vector maths.
      Each trip covers at most a few tens of kilometres, so the error of
      this projection is small; the EDA module measures it per trip
      (``projection_max_rel_error``) instead of assuming it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

EARTH_RADIUS_M = 6_371_008.8  # mean Earth radius (IUGG)


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in metres between (lon1, lat1) and (lon2, lat2). Vectorised."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, dtype=float)) for v in (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass(frozen=True)
class LocalProjection:
    """Equirectangular projection around (lon0, lat0); outputs metres."""

    lon0: float
    lat0: float

    @classmethod
    def from_points(cls, lon, lat) -> "LocalProjection":
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        if lon.size == 0 or np.all(np.isnan(lon)):
            raise ValueError("Cannot build a projection from an empty set of points")
        return cls(float(np.nanmean(lon)), float(np.nanmean(lat)))

    @property
    def _cos_lat0(self) -> float:
        return float(np.cos(np.radians(self.lat0)))

    def forward(self, lon, lat) -> Tuple[np.ndarray, np.ndarray]:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x = np.radians(lon - self.lon0) * self._cos_lat0 * EARTH_RADIUS_M
        y = np.radians(lat - self.lat0) * EARTH_RADIUS_M
        return x, y

    def inverse(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        lon = self.lon0 + np.degrees(x / (EARTH_RADIUS_M * self._cos_lat0))
        lat = self.lat0 + np.degrees(y / EARTH_RADIUS_M)
        return lon, lat


def point_segment_distance(px, py, ax, ay, bx, by) -> Tuple[np.ndarray, np.ndarray]:
    """Distance from points P to segments AB in projected metres (broadcasting).

    Returns ``(distance, t)`` where ``t`` in [0, 1] is the position of the
    closest point along AB (0 = at A, 1 = at B). Zero-length segments are
    treated as the single point A.
    """
    px, py, ax, ay, bx, by = (np.asarray(v, dtype=float) for v in (px, py, ax, ay, bx, by))
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    safe = np.where(length_sq > 0, length_sq, 1.0)
    t = np.where(length_sq > 0, ((px - ax) * dx + (py - ay) * dy) / safe, 0.0)
    t = np.clip(t, 0.0, 1.0)
    distance = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
    return distance, t


def bearing_from_xy(dx, dy) -> np.ndarray:
    """Compass bearing in degrees [0, 360): 0 = north, 90 = east."""
    return np.degrees(np.arctan2(np.asarray(dx, dtype=float), np.asarray(dy, dtype=float))) % 360.0


def angle_diff_deg(a, b) -> np.ndarray:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    diff = np.abs((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360.0)
    return np.minimum(diff, 360.0 - diff)
