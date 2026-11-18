"""Server-side evaluator: compares a finished prediction with the reference route.

Runs AFTER prediction is complete and only on Laptop 1. The reference route is
read here, turned into per-observation labels, and reduced to summary numbers.
Neither the route nor the labels leave this module: the API returns metrics only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import load_reference_route
from src.evaluation.metrics import route_continuity, route_overlap
from src.features.labels import align_track_to_route
from src.inference.predictor import TripPrediction


def _top1(pred: pd.DataFrame, labels: pd.DataFrame, graph) -> float:
    merged = labels[["obs_idx", "ref_arc_id"]].merge(pred[["obs_idx", "arc_id"]], on="obs_idx", how="left")
    ok = merged["arc_id"].notna().to_numpy()
    ref = merged["ref_arc_id"].to_numpy(dtype=np.int64)
    arc = merged["arc_id"].fillna(0).to_numpy(dtype=np.int64)
    correct = ok & (graph.arc_key[arc] == graph.arc_key[ref])
    return float(correct.mean()) if len(merged) else float("nan")


def evaluate_prediction(result: TripPrediction, data_root: Path) -> dict:
    route = load_reference_route(data_root, result.trip_id)
    if route is None or route.size == 0:
        return {"trip_id": result.trip_id, "available": False,
                "detail": "No reference route on the server for this trip; metrics not computed."}
    pre, graph = result.pre, result.graph
    labels = align_track_to_route(pre.track, pre.nodes_xy, pre.arcs, route)
    out = {"trip_id": result.trip_id, "available": True, "n_labelled_observations": int(len(labels))}
    for name, pred in (("hmm", result.hmm), ("classifier_only", result.classifier)):
        cont = route_continuity(pred, pre.track, graph)
        overlap = route_overlap(pred.sort_values("obs_idx")["arc_id"].to_numpy(), route, graph)
        out[name] = {"top1_accuracy": _top1(pred, labels, graph), "route_continuity": cont["route_continuity"],
                     "n_breaks": cont["n_breaks"], **overlap}
    return out
