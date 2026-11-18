"""End-to-end prediction for ONE unseen trip, without any reference labels.

    raw (.track/.nodes/.arcs only) -> preprocess -> candidates + features
    -> classifier probabilities -> HMM/Viterbi -> predicted arc per observation

This is the function the API calls. It never reads a .route file: the RawTrip
it receives comes from ``load_trip_for_inference`` and has an empty route.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from src.common.graph import RoadGraph
from src.common.io import RawTrip
from src.evaluation.metrics import select_top1
from src.features.features import build_trip_features, model_matrix
from src.features.spatial_index import ArcIndex
from src.hmm.decoder import decode_trip
from src.inference.artifacts import Bundle
from src.preprocessing.pipeline import PreprocessedTrip, preprocess_trip


@dataclass
class TripPrediction:
    trip_id: str
    pre: PreprocessedTrip
    graph: RoadGraph
    candidates: pd.DataFrame  # one row per (observation, candidate) with "prob"
    classifier: pd.DataFrame  # obs_idx, arc_id, offset_frac, prob (top-1 per observation)
    hmm: pd.DataFrame  # obs_idx, arc_id, offset_frac, prob (Viterbi path)
    timings_s: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, float] = field(default_factory=dict)


def predict_trip(raw: RawTrip, bundle: Bundle, log=None) -> TripPrediction:
    if raw.route.size:
        raise ValueError("predict_trip must receive a trip loaded WITHOUT its reference route")
    say = log or (lambda msg: None)
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    pre = preprocess_trip(raw, bundle.preprocess_config)
    timings["preprocess"] = time.perf_counter() - t0
    say(f"preprocessed: {len(pre.track)} observations, {pre.report['n_segments']} segments")

    t0 = time.perf_counter()
    index = ArcIndex(pre.nodes_xy, pre.arcs, bundle.candidate_config.sample_spacing_m)
    table, summary = build_trip_features(pre, bundle.candidate_config, with_labels=False, index=index)
    timings["candidates_features"] = time.perf_counter() - t0
    say(f"candidates: {len(table)} rows ({summary['mean_candidates_per_obs']:.2f} per observation)")

    t0 = time.perf_counter()
    table["prob"] = bundle.model.predict_proba(model_matrix(table))[:, 1] if len(table) else []
    timings["classifier"] = time.perf_counter() - t0
    say(f"classifier scored {len(table)} candidates")

    cands = table[["trip_id", "obs_idx", "arc_id", "offset_frac", "distance_rank", "prob"]]
    clf = select_top1(cands.rename(columns={"prob": "score"})).rename(columns={"score": "prob"})
    clf = clf[["obs_idx", "arc_id", "offset_frac", "prob"]]

    t0 = time.perf_counter()
    graph = RoadGraph(pre.nodes_xy, pre.arcs)
    hmm, hmm_stats = decode_trip(cands, pre.track, graph, bundle.hmm_config)
    timings["hmm_viterbi"] = time.perf_counter() - t0
    timings["total"] = sum(timings.values())
    say(f"HMM/Viterbi decoded {hmm_stats['n_decoded']} observations in {hmm_stats['n_chains']} chains")

    stats = {"n_observations": int(len(pre.track)), "n_candidates": int(len(table)),
             "n_segments": int(pre.report["n_segments"]),
             "frac_obs_without_candidates": float(summary["frac_obs_without_candidates"]),
             **{k: v for k, v in hmm_stats.items() if k != "decode_time_s"}}
    return TripPrediction(trip_id=str(raw.trip_id), pre=pre, graph=graph, candidates=table, classifier=clf, hmm=hmm,
                          timings_s=timings, stats=stats)
