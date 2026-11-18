"""Evaluation metrics shared by model comparison, HMM experiments and the API evaluator.

Definitions (used identically everywhere; see docs/METRICS.md):

Candidate-level (binary classification of (observation, candidate arc) rows,
label ``is_reference``):
    accuracy, precision, recall, F1 (positive class), macro F1 (mean of the F1 of
    both classes), log loss, confusion matrix. Probabilities are thresholded at 0.5.

Observation-level Top-1 accuracy:
    for every GPS observation, the candidate with the highest score is chosen
    (ties -> nearer candidate). Correct when that arc is the reference arc (or
    an identical duplicate arc). Denominator = ALL observations that have a
    reference label, so observations without any candidate count as errors.

Candidate miss rate:
    share of labelled observations whose reference arc is not among their candidates.

Route continuity:
    over consecutive observation pairs (t, t+1) in the SAME gap segment that
    both have a prediction: share of pairs where the predicted position at t+1
    can be reached from the predicted position at t along the directed network
    within a physically plausible distance (v_max * dt + 2 * tolerance).
    Pairs split by a GPS gap are excluded from the denominator.

Sequence-level route metrics (per trip, length-weighted, duplicates merged):
    route precision = length of predicted arcs that are on the reference route / length of predicted arcs
    route recall    = length of reference-route arcs that were predicted / length of the reference route
    route F1        = harmonic mean. Arcs are counted once each (as a set).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score)

from src.common.graph import RoadGraph

MAX_SPEED_MPS = 60.0  # train p95 speeds peak at 65.8 m/s on a highway trip; 60 m/s + tolerance covers it
POSITION_TOLERANCE_M = 25.0  # GPS noise allowance at each end (train p95 GPS-to-route distance is 14 m)


def classification_metrics(y_true, proba, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_top1(candidates: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """One row per (trip_id, obs_idx): the highest-scoring candidate (ties -> lower distance_rank)."""
    if candidates.empty:
        return candidates.copy()
    ordered = candidates.sort_values(["trip_id", "obs_idx", score_col, "distance_rank"],
                                     ascending=[True, True, False, True], kind="mergesort")
    return ordered.drop_duplicates(["trip_id", "obs_idx"], keep="first").reset_index(drop=True)


def observation_accuracy(chosen: pd.DataFrame, observations: pd.DataFrame) -> Dict[str, float]:
    """Top-1 accuracy of chosen candidates.

    ``chosen``: trip_id, obs_idx, is_reference (one row per observation that got a prediction).
    ``observations``: trip_id, obs_idx, has_reference_candidate (one row per labelled observation).
    """
    merged = observations[["trip_id", "obs_idx", "has_reference_candidate"]].merge(
        chosen[["trip_id", "obs_idx", "is_reference"]], on=["trip_id", "obs_idx"], how="left")
    correct = merged["is_reference"].fillna(False).astype(bool)
    covered = merged["has_reference_candidate"].astype(bool)
    return {
        "top1_accuracy": float(correct.mean()) if len(merged) else float("nan"),
        "top1_accuracy_covered": float(correct[covered].mean()) if covered.any() else float("nan"),
        "candidate_miss_rate": float(1.0 - covered.mean()) if len(merged) else float("nan"),
        "n_observations": int(len(merged)),
        "n_predicted": int(merged["is_reference"].notna().sum()),
    }


def route_continuity(pred: pd.DataFrame, track: pd.DataFrame, graph: RoadGraph) -> Dict[str, float]:
    """Continuity of one trip's predicted positions.

    ``pred``: obs_idx, arc_id, offset_frac (one row per predicted observation).
    ``track``: PreprocessedTrip.track (obs_idx, t, x, y, segment_id).
    """
    df = track[["obs_idx", "t", "x", "y", "segment_id"]].merge(pred[["obs_idx", "arc_id", "offset_frac"]],
                                                                 on="obs_idx", how="left")
    n_pairs = n_ok = 0
    arc = df["arc_id"].to_numpy()
    frac = df["offset_frac"].to_numpy()
    t = df["t"].to_numpy()
    seg = df["segment_id"].to_numpy()
    for i in range(len(df) - 1):
        if seg[i] != seg[i + 1] or np.isnan(arc[i]) or np.isnan(arc[i + 1]):
            continue
        n_pairs += 1
        budget = MAX_SPEED_MPS * max(t[i + 1] - t[i], 1.0) + 2 * POSITION_TOLERANCE_M
        d = graph.route_distance(int(arc[i]), float(frac[i]), int(arc[i + 1]), float(frac[i + 1]), budget)
        n_ok += np.isfinite(d)
    return {"route_continuity": n_ok / n_pairs if n_pairs else float("nan"),
            "n_transitions": int(n_pairs), "n_breaks": int(n_pairs - n_ok)}


def route_overlap(pred_arcs, reference_route, graph: RoadGraph) -> Dict[str, float]:
    """Length-weighted precision/recall/F1 between predicted arcs and the reference route (as sets)."""
    pred_arcs = np.asarray([a for a in pred_arcs if a is not None and not pd.isna(a)], dtype=np.int64)
    ref = np.asarray(reference_route, dtype=np.int64)
    if pred_arcs.size == 0 or ref.size == 0:
        return {"route_precision": float("nan"), "route_recall": float("nan"), "route_f1": float("nan")}
    pred_keys, pi = np.unique(graph.arc_key[pred_arcs], return_index=True)
    ref_keys, ri = np.unique(graph.arc_key[ref], return_index=True)
    pred_len = graph.length_m[pred_arcs[pi]]
    ref_len = graph.length_m[ref[ri]]
    in_ref = np.isin(pred_keys, ref_keys)
    in_pred = np.isin(ref_keys, pred_keys)
    precision = pred_len[in_ref].sum() / pred_len.sum() if pred_len.sum() > 0 else float("nan")
    recall = ref_len[in_pred].sum() / ref_len.sum() if ref_len.sum() > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"route_precision": float(precision), "route_recall": float(recall), "route_f1": float(f1)}


def observation_table(features: pd.DataFrame, labels: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per-observation label table from a labelled feature table (+ the label alignment for
    observations that have no candidates at all)."""
    has = features.groupby(["trip_id", "obs_idx"])["is_reference"].any().rename("has_reference_candidate")
    has = has.reset_index()
    if labels is not None:
        has = labels[["trip_id", "obs_idx"]].merge(has, on=["trip_id", "obs_idx"], how="left")
        has["has_reference_candidate"] = has["has_reference_candidate"].fillna(False).astype(bool)
    return has
