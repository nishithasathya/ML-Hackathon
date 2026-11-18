"""Turn the reference route into a per-observation reference arc.

The .route file is an ordered list of arc ids; it does NOT say which GPS
observation belongs to which arc. Classifiers in later phases need a label per
observation, so we derive one here.

Why not simply "nearest route arc"? Routes often use both directions of the
same street or pass the same place twice; geometrically identical arcs would
then be confused. Instead we use a monotonic alignment (dynamic programming):

    cost[i, j] = d(obs_i, route_arc_j) + min_{k <= j} cost[i-1, k]

i.e. observation i is matched to route position j, positions may never go
backwards in time, and the total GPS-to-arc distance is minimised. Several
observations may share one arc, and very short arcs may receive no
observation, which is expected at 1 Hz sampling.

Limitations (reported, not hidden): the labels inherit any error in the
reference route itself (11 routes contain continuity breaks), and for trips
with a systematic GPS offset (e.g. ~130 m) the alignment is less trustworthy;
``ref_distance_m`` is kept so these cases can be filtered or analysed.

IMPORTANT: labels are targets only. They must never be used as model inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.geo import point_segment_distance


def align_track_to_route(track: pd.DataFrame, nodes_xy: np.ndarray, arcs: np.ndarray,
                         route: np.ndarray) -> pd.DataFrame:
    """Return obs_idx, ref_route_pos, ref_arc_id, ref_distance_m for every observation."""
    n_obs, n_route = len(track), int(route.size)
    columns = ["obs_idx", "ref_route_pos", "ref_arc_id", "ref_distance_m"]
    if n_obs == 0 or n_route == 0:
        return pd.DataFrame(columns=columns)

    ra = arcs[route]
    ax, ay = nodes_xy[ra[:, 0], 0], nodes_xy[ra[:, 0], 1]
    bx, by = nodes_xy[ra[:, 1], 0], nodes_xy[ra[:, 1], 1]
    px, py = track["x"].to_numpy(), track["y"].to_numpy()

    # Back-pointers: for observation i and route position j, which position the
    # previous observation used. uint16 keeps memory low (routes have < 65k arcs).
    ptr_dtype = np.uint16 if n_route < np.iinfo(np.uint16).max else np.uint32
    back = np.zeros((n_obs, n_route), dtype=ptr_dtype)
    positions = np.arange(n_route)

    cost, _ = point_segment_distance(px[0], py[0], ax, ay, bx, by)
    for i in range(1, n_obs):
        running_min = np.minimum.accumulate(cost)
        # Latest index achieving the prefix minimum = a valid argmin of cost[:j+1].
        argmin = np.maximum.accumulate(np.where(cost == running_min, positions, 0))
        back[i] = argmin
        dist, _ = point_segment_distance(px[i], py[i], ax, ay, bx, by)
        cost = dist + running_min

    path = np.empty(n_obs, dtype=np.int64)
    path[-1] = int(np.argmin(cost))
    for i in range(n_obs - 1, 0, -1):
        path[i - 1] = back[i, path[i]]

    ref_dist, _ = point_segment_distance(px, py, ax[path], ay[path], bx[path], by[path])
    return pd.DataFrame({"obs_idx": track["obs_idx"].to_numpy(), "ref_route_pos": path,
                         "ref_arc_id": route[path], "ref_distance_m": ref_dist})
