"""Feature extraction for (observation, candidate arc) pairs.

Three groups of features are built:

1. Observation features (motion of the vehicle, from the GPS track only):
   time since start, time to previous/next fix, step distance, speed,
   smoothed speed, acceleration, heading, heading change, cumulative distance.
   Computed within gap segments so that nothing is measured across a long gap.
2. Arc attributes (from the road network only): length, bearing, how many
   arcs continue from its end node, whether the opposite-direction arc exists.
3. Pair features: GPS-to-arc distance, distance rank, distance relative to
   the nearest candidate, position along the arc, and the angle between the
   vehicle heading and the arc bearing (the main signal for telling apart the
   two directions of the same street).

4. Temporal-context pair features (phase 2): how far the PREVIOUS and NEXT GPS
   fixes are from the same candidate arc, and how far the vehicle moves along
   the arc's direction between them. A correct arc is usually close to the
   neighbouring fixes too, and the vehicle moves forward along it; the
   opposite-direction twin has the same distances but negative motion.

The reference label (``is_reference``) is attached as a separate column and is
never used to compute any feature. ``FEATURE_COLUMNS`` is the single list of
model inputs used by training AND inference; ``LABEL_COLUMNS`` must never be a
model input.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.geo import angle_diff_deg, bearing_from_xy, point_segment_distance
from src.features.candidates import CandidateConfig, generate_candidates
from src.features.labels import align_track_to_route
from src.features.spatial_index import ArcIndex

# Heading is undefined when the vehicle is (almost) stationary; below this
# displacement we report NaN instead of a noise-driven angle.
MIN_HEADING_DISPLACEMENT_M = 2.0
SPEED_SMOOTHING_WINDOW = 5  # observations (about 5 s at 1 Hz)

# Identifier columns: needed to map scores back to (observation, road arc); never model inputs.
ID_COLUMNS = ["trip_id", "obs_idx", "arc_id"]
# Label / diagnostic columns derived from the reference route: NEVER model inputs.
LABEL_COLUMNS = ["is_reference", "ref_distance_m", "ref_arc_id"]
# Model inputs. Trip-progress columns (t_rel, cum_dist_m, segment_id) and absolute
# directions (heading_deg, arc_bearing_deg) are deliberately excluded: they cannot
# generalise to an unseen trip.
FEATURE_COLUMNS = [
    "distance_m", "distance_rank", "dist_minus_nearest_m", "offset_frac", "n_candidates",
    "heading_diff_deg", "heading_cos", "along_arc_motion_m", "prev_obs_dist_m", "next_obs_dist_m",
    "arc_length_m", "arc_to_out_degree", "arc_has_reverse",
    "speed_mps", "speed_smooth_mps", "accel_mps2", "heading_change_deg", "step_dist_m",
    "dt_prev", "dt_next", "is_speed_outlier", "is_segment_start",
]


def model_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """The exact model input matrix (same function for training and inference)."""
    return table[FEATURE_COLUMNS].astype(float)


def observation_features(track: pd.DataFrame) -> pd.DataFrame:
    """Per-observation motion features. ``track`` is PreprocessedTrip.track."""
    df = track[["obs_idx", "x", "y", "t", "t_rel", "dt_prev", "segment_id", "is_segment_start",
                "is_speed_outlier"]].copy()
    seg = df["segment_id"]
    same_prev = seg.eq(seg.shift(1))
    same_next = seg.eq(seg.shift(-1))

    dx_prev = df["x"].diff().where(same_prev)
    dy_prev = df["y"].diff().where(same_prev)
    df["step_dist_m"] = np.hypot(dx_prev, dy_prev)
    df["dt_prev"] = df["dt_prev"].where(same_prev)
    df["dt_next"] = (-df["t"].diff(-1)).where(same_next)
    df["speed_mps"] = df["step_dist_m"] / df["dt_prev"]
    df["speed_smooth_mps"] = (df.groupby("segment_id")["speed_mps"]
                              .transform(lambda s: s.rolling(SPEED_SMOOTHING_WINDOW, center=True, min_periods=1).median()))
    df["accel_mps2"] = (df["speed_mps"].diff() / df["dt_prev"]).where(same_prev & same_prev.shift(1, fill_value=False))

    # Heading from the previous to the next fix (central difference); one-sided at segment ends.
    x_next = df["x"].shift(-1).where(same_next, df["x"])
    y_next = df["y"].shift(-1).where(same_next, df["y"])
    x_prev = df["x"].shift(1).where(same_prev, df["x"])
    y_prev = df["y"].shift(1).where(same_prev, df["y"])
    hdx, hdy = x_next - x_prev, y_next - y_prev
    heading = pd.Series(bearing_from_xy(hdx, hdy), index=df.index)
    df["heading_deg"] = heading.where(np.hypot(hdx, hdy) >= MIN_HEADING_DISPLACEMENT_M)
    df["heading_change_deg"] = pd.Series(angle_diff_deg(df["heading_deg"], df["heading_deg"].shift(1)),
                                         index=df.index).where(same_prev)
    df["cum_dist_m"] = df["step_dist_m"].fillna(0.0).cumsum()
    return df.drop(columns=["x", "y", "t"])


def arc_attributes(index: ArcIndex) -> pd.DataFrame:
    """Static per-arc attributes derived from network geometry and topology."""
    arcs = index.arcs
    n_nodes = int(arcs.max()) + 1 if arcs.size else 0
    out_degree = np.bincount(arcs[:, 0], minlength=n_nodes)
    pair_set = set(map(tuple, arcs.tolist()))
    has_reverse = np.fromiter(((b, a) in pair_set for a, b in arcs.tolist()), dtype=bool, count=len(arcs))
    return pd.DataFrame({
        "arc_id": np.arange(len(arcs)),
        "arc_length_m": index.length_m,
        "arc_bearing_deg": index.bearing_deg,
        "arc_to_out_degree": out_degree[arcs[:, 1]],  # number of ways to continue after this arc
        "arc_has_reverse": has_reverse,
    })


def candidate_features(candidates: pd.DataFrame, obs_feats: pd.DataFrame, arc_attrs: pd.DataFrame) -> pd.DataFrame:
    table = candidates.merge(arc_attrs, on="arc_id", how="left").merge(obs_feats, on="obs_idx", how="left")
    grouped = table.groupby("obs_idx")["distance_m"]
    table["n_candidates"] = grouped.transform("size")
    table["dist_minus_nearest_m"] = table["distance_m"] - grouped.transform("min")
    table["heading_diff_deg"] = angle_diff_deg(table["heading_deg"], table["arc_bearing_deg"])
    table.loc[table["heading_deg"].isna(), "heading_diff_deg"] = np.nan
    return table


def add_context_features(table: pd.DataFrame, track: pd.DataFrame, index: ArcIndex) -> pd.DataFrame:
    """Distances of the previous/next fix (same gap segment) to each candidate arc, and along-arc motion."""
    if table.empty:
        for col in ("prev_obs_dist_m", "next_obs_dist_m", "along_arc_motion_m", "heading_cos"):
            table[col] = pd.Series(dtype=float)
        return table
    pos_of = pd.Series(np.arange(len(track)), index=track["obs_idx"].to_numpy())
    pos = pos_of.reindex(table["obs_idx"].to_numpy()).to_numpy()
    x, y, seg = track["x"].to_numpy(), track["y"].to_numpy(), track["segment_id"].to_numpy()
    n = len(track)
    arc = table["arc_id"].to_numpy()
    ax, ay, bx, by = index.ax[arc], index.ay[arc], index.bx[arc], index.by[arc]

    def neighbour(offset: int):
        npos = pos + offset
        ok = (npos >= 0) & (npos < n)
        npos_c = np.clip(npos, 0, n - 1)
        ok &= seg[npos_c] == seg[pos]
        return npos_c, ok

    out = {}
    for name, offset in (("prev_obs_dist_m", -1), ("next_obs_dist_m", 1)):
        npos, ok = neighbour(offset)
        d, _ = point_segment_distance(x[npos], y[npos], ax, ay, bx, by)
        out[name] = np.where(ok, d, np.nan)
    p_prev, ok_prev = neighbour(-1)
    p_next, ok_next = neighbour(1)
    # Displacement prev -> next (falls back to the fix itself at segment ends), projected on the arc direction.
    x0 = np.where(ok_prev, x[p_prev], x[pos]); y0 = np.where(ok_prev, y[p_prev], y[pos])
    x1 = np.where(ok_next, x[p_next], x[pos]); y1 = np.where(ok_next, y[p_next], y[pos])
    length = np.hypot(bx - ax, by - ay)
    safe = np.where(length > 0, length, 1.0)
    along = ((x1 - x0) * (bx - ax) + (y1 - y0) * (by - ay)) / safe
    out["along_arc_motion_m"] = np.where(length > 0, along, 0.0)
    for name, values in out.items():
        table[name] = values
    table["heading_cos"] = np.cos(np.radians(table["heading_diff_deg"]))
    return table


def build_trip_features(pre, config: CandidateConfig = CandidateConfig(), with_labels: bool = True,
                        index: Optional[ArcIndex] = None,
                        labels: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Full feature table for one PreprocessedTrip, plus a summary of candidate quality.

    ``with_labels=False`` is how inference will call this in later phases:
    the route is then not read at all. ``index`` and ``labels`` may be passed
    in to avoid recomputing them (used by the radius sweep).
    """
    index = index or ArcIndex(pre.nodes_xy, pre.arcs, config.sample_spacing_m)
    candidates = generate_candidates(pre.track, index, config)
    table = candidate_features(candidates, observation_features(pre.track), arc_attributes(index))
    table = add_context_features(table, pre.track, index)
    table.insert(0, "trip_id", str(pre.trip_id))

    n_obs = len(pre.track)
    n_with = int(candidates["obs_idx"].nunique()) if len(candidates) else 0
    summary: Dict[str, float] = {
        "partition": pre.partition, "trip_id": pre.trip_id, "n_observations": n_obs,
        "radius_m": config.radius_m, "max_candidates": config.max_candidates,
        "n_candidate_rows": int(len(candidates)),
        "frac_obs_without_candidates": 1.0 - n_with / n_obs if n_obs else np.nan,
        "mean_candidates_per_obs": len(candidates) / n_obs if n_obs else np.nan,
    }
    if with_labels:
        if labels is None:
            labels = align_track_to_route(pre.track, pre.nodes_xy, pre.arcs, pre.route)
        table = table.merge(labels[["obs_idx", "ref_arc_id", "ref_distance_m"]], on="obs_idx", how="left")
        # Arcs with the same (from, to) node pair are the same road piece (1,903 duplicate arcs
        # in 60 trips): both count as the reference, otherwise identical inputs get opposite labels.
        ref_ok = table["ref_arc_id"].notna().to_numpy()
        ref_ids = table["ref_arc_id"].fillna(0).astype(np.int64).to_numpy()
        cand = table["arc_id"].to_numpy()
        same_pair = (pre.arcs[cand, 0] == pre.arcs[ref_ids, 0]) & (pre.arcs[cand, 1] == pre.arcs[ref_ids, 1])
        table["is_reference"] = ref_ok & ((cand == ref_ids) | same_pair)
        hit = table.groupby("obs_idx")["is_reference"].any().reindex(labels["obs_idx"], fill_value=False)
        nearest_hit = table.loc[table["distance_rank"] == 1].set_index("obs_idx")["is_reference"] \
            .reindex(labels["obs_idx"], fill_value=False)
        # Diagnostic: when the nearest arc is wrong, is it just the opposite-direction twin of the reference?
        nearest = table.loc[table["distance_rank"] == 1, ["obs_idx", "arc_id", "ref_arc_id"]].dropna()
        ref = nearest["ref_arc_id"].astype(np.int64).to_numpy()
        arc = nearest["arc_id"].to_numpy()
        is_twin = (pre.arcs[arc, 0] == pre.arcs[ref, 1]) & (pre.arcs[arc, 1] == pre.arcs[ref, 0]) & (arc != ref)
        summary.update({
            "nearest_arc_is_reverse_of_reference_rate": float(is_twin.sum() / n_obs) if n_obs else np.nan,
            "candidate_miss_rate": float(1.0 - hit.mean()),
            # A geometric heuristic, NOT a trained model: how often the closest arc is the reference arc.
            "nearest_arc_is_reference_rate": float(nearest_hit.mean()),
            "ref_distance_median_m": float(labels["ref_distance_m"].median()),
        })
        # ref_arc_id is kept (label column) so evaluation can score observations without
        # any correct candidate; it is listed in LABEL_COLUMNS and never used as a feature.
    return table, summary
