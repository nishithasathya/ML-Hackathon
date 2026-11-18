"""Build the candidate-level datasets used for training and evaluation.

Formulation (binary candidate scoring)
--------------------------------------
Every row is one (GPS observation, candidate road arc) pair produced by
``src.features.build_trip_features``. The target is ``is_reference`` (1 if the
candidate is the reference arc for that observation). A model outputs
P(candidate is correct | features); the observation's prediction is the
highest-scoring candidate, and the HMM consumes the probabilities as emissions.

Why not multiclass? Candidate sets differ per observation and arc ids are local
to each trip, so "arc 1234" is not a class that means anything across trips.
Scoring candidates keeps the candidate -> road-arc mapping explicit.

Leakage rules enforced here
---------------------------
* Only ``train`` and ``student_test`` are readable (``check_partition``).
* Model inputs are exactly ``FEATURE_COLUMNS``; label columns are separate.
* Trip ids stay strings.
* The rules that reduce the TRAINING set are decided from training data only
  (offset-trip rule from the train EDA, stride subsampling) and never touch the
  student-test evaluation set, which is always scored on every candidate.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.config import RANDOM_SEED, check_partition, get_processed_dir
from src.common.io import iter_trips, list_trips
from src.features.candidates import CandidateConfig
from src.features.features import build_trip_features, model_matrix
from src.features.labels import align_track_to_route
from src.preprocessing.pipeline import PreprocessConfig, preprocess_trip


@dataclass
class TrainingSubsetConfig:
    # Fixes are 1 s apart, so neighbouring observations are near-duplicates. Keeping every
    # k-th observation (with ALL of its candidates) cuts cost without changing the label mix.
    observation_stride: int = 3
    # Train-only data-quality rule (phase-1 EDA): trips whose median GPS-to-reference distance
    # exceeds this are excluded from FITTING because their labels are unreliable (00000033, 00000054).
    max_median_ref_distance_m: float = 30.0
    # Observations with no correct candidate carry no ranking signal for the classifier.
    drop_observations_without_positive: bool = True


def build_partition(partition: str, candidate_config: CandidateConfig = CandidateConfig(),
                    preprocess_config: PreprocessConfig = PreprocessConfig(), data_root=None,
                    trip_ids: Optional[List[str]] = None, use_cache: bool = True,
                    verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (candidate feature table with labels, per-observation label table) for a partition."""
    check_partition(partition)
    tag = f"{partition}_r{int(candidate_config.radius_m)}_k{candidate_config.max_candidates}"
    cache = get_processed_dir() / f"{tag}.pkl"
    if use_cache and trip_ids is None and data_root is None and cache.exists():
        feats, obs = pd.read_pickle(cache)
        return feats, obs
    feats_parts, obs_parts = [], []
    start = time.perf_counter()
    for raw in iter_trips(partition, data_root, trip_ids):
        pre = preprocess_trip(raw, preprocess_config)
        labels = align_track_to_route(pre.track, pre.nodes_xy, pre.arcs, pre.route)
        table, _ = build_trip_features(pre, candidate_config, with_labels=True, labels=labels)
        feats_parts.append(table)
        obs = labels[["obs_idx", "ref_arc_id", "ref_distance_m"]].copy()
        obs.insert(0, "trip_id", str(raw.trip_id))
        hit = table.groupby("obs_idx")["is_reference"].any()
        obs["has_reference_candidate"] = obs["obs_idx"].map(hit).fillna(False).astype(bool)
        obs = obs.merge(pre.track[["obs_idx", "segment_id"]], on="obs_idx", how="left")
        obs_parts.append(obs)
        if verbose:
            print(f"  {partition}/{raw.trip_id}: {len(table)} candidate rows", flush=True)
    feats = pd.concat(feats_parts, ignore_index=True)
    obs = pd.concat(obs_parts, ignore_index=True)
    for col in ("arc_has_reverse", "is_speed_outlier", "is_segment_start"):
        feats[col] = feats[col].astype(float)
    feats["is_reference"] = feats["is_reference"].astype(bool)
    if verbose:
        print(f"{partition}: {len(feats)} rows, {len(obs)} observations "
              f"({time.perf_counter() - start:.1f}s)", flush=True)
    if trip_ids is None and data_root is None:
        pd.to_pickle((feats, obs), cache)
    return feats, obs


def offset_trips(obs: pd.DataFrame, max_median_m: float) -> List[str]:
    med = obs.groupby("trip_id")["ref_distance_m"].median()
    return sorted(med[med > max_median_m].index.astype(str))


def training_subset(feats: pd.DataFrame, obs: pd.DataFrame,
                    config: TrainingSubsetConfig = TrainingSubsetConfig()) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Rows used to FIT models (evaluation always uses all rows)."""
    excluded = offset_trips(obs, config.max_median_ref_distance_m)
    keep = ~feats["trip_id"].isin(excluded)
    if config.observation_stride > 1:
        keep &= (feats["obs_idx"] % config.observation_stride) == 0
    if config.drop_observations_without_positive:
        has_pos = feats.groupby(["trip_id", "obs_idx"])["is_reference"].transform("any")
        keep &= has_pos
    subset = feats.loc[keep].reset_index(drop=True)
    info = {"excluded_offset_trips": excluded, "n_rows_all": int(len(feats)), "n_rows_fit": int(len(subset)),
            "positive_rate_all": float(feats["is_reference"].mean()),
            "positive_rate_fit": float(subset["is_reference"].mean()), **asdict(config)}
    return subset, info


def xy(frame: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Model inputs and target. The ONLY place features are selected for models."""
    return model_matrix(frame), frame["is_reference"].to_numpy(dtype=int)


def rng() -> np.random.Generator:
    return np.random.default_rng(RANDOM_SEED)
