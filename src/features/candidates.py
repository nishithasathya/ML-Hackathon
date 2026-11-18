"""Candidate generation: which road arcs could each GPS observation belong to?

Later phases will train classifiers that score (observation, candidate arc)
pairs, and an HMM whose hidden states are these candidates. In this phase we
only generate and measure the candidate sets.

For every observation we keep the ``max_candidates`` nearest arcs within
``radius_m``. Note: most roads are stored as two opposite arcs with identical
geometry (EDA: reciprocal_arc_frac is usually > 0.9), so they tie on distance;
the heading feature (features.py) is what can separate them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.spatial_index import ArcIndex

CANDIDATE_COLUMNS = ["obs_idx", "arc_id", "distance_m", "offset_frac", "distance_rank"]


@dataclass
class CandidateConfig:
    radius_m: float = 50.0  # see the radius sweep in run_features.py --sweep for the evidence
    max_candidates: int = 10
    sample_spacing_m: float = 20.0


def generate_candidates(track: pd.DataFrame, index: ArcIndex, config: CandidateConfig = CandidateConfig()) -> pd.DataFrame:
    """Return one row per (observation, candidate arc), sorted by obs then distance.

    ``track`` needs columns obs_idx, x, y (output of preprocessing).
    ``distance_rank`` is 1 for the nearest arc. Observations with no arc within
    the radius simply have no rows.
    """
    x = track["x"].to_numpy()
    y = track["y"].to_numpy()
    obs_ids = track["obs_idx"].to_numpy()
    near = index.arcs_within(x, y, config.radius_m)

    parts = []
    for obs, px, py, arc_ids in zip(obs_ids, x, y, near):
        if arc_ids.size == 0:
            continue
        dist, frac = index.distance_to_arcs(px, py, arc_ids)
        order = np.lexsort((arc_ids, dist))[: config.max_candidates]  # by distance, then arc id
        parts.append(np.column_stack([np.full(order.size, obs), arc_ids[order], dist[order], frac[order],
                                      np.arange(1, order.size + 1)]))
    if not parts:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    table = pd.DataFrame(np.vstack(parts), columns=CANDIDATE_COLUMNS)
    for col in ("obs_idx", "arc_id", "distance_rank"):
        table[col] = table[col].astype(np.int64)
    return table
