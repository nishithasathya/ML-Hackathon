"""Route-level evaluation over a whole partition (continuity + sequence metrics).

Works on any per-observation prediction table (classifier top-1 or HMM output):
columns trip_id, obs_idx, arc_id, offset_frac.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.graph import RoadGraph
from src.common.io import iter_trips
from src.evaluation.metrics import route_continuity, route_overlap
from src.preprocessing.pipeline import PreprocessConfig, PreprocessedTrip, preprocess_trip


def trip_contexts(partition: str, trip_ids: Optional[List[str]] = None, data_root=None,
                  preprocess_config: PreprocessConfig = PreprocessConfig()
                  ) -> Iterator[Tuple[PreprocessedTrip, RoadGraph]]:
    for raw in iter_trips(partition, data_root, trip_ids):
        pre = preprocess_trip(raw, preprocess_config)
        yield pre, RoadGraph(pre.nodes_xy, pre.arcs)


def evaluate_routes(predictions: Dict[str, pd.DataFrame], contexts) -> pd.DataFrame:
    """Per-trip continuity and route precision/recall/F1 for one or more prediction sets.

    ``predictions``: {method_name: table(trip_id, obs_idx, arc_id, offset_frac)}.
    ``contexts``: iterable of (PreprocessedTrip, RoadGraph) (materialised list recommended).
    """
    rows = []
    for pre, graph in contexts:
        tid = str(pre.trip_id)
        for method, table in predictions.items():
            pred = table[table["trip_id"] == tid]
            rec = {"trip_id": tid, "method": method}
            rec.update(route_continuity(pred, pre.track, graph))
            ordered = pred.sort_values("obs_idx")["arc_id"].to_numpy()
            rec.update(route_overlap(ordered, pre.route, graph))
            rows.append(rec)
    return pd.DataFrame(rows)


def summarise_routes(per_trip: pd.DataFrame) -> pd.DataFrame:
    """Partition-level summary per method: continuity pooled over transitions, route metrics averaged over trips."""
    out = []
    for method, g in per_trip.groupby("method", sort=False):
        trans = g["n_transitions"].sum()
        out.append({"method": method,
                    "route_continuity": float((trans - g["n_breaks"].sum()) / trans) if trans else np.nan,
                    "n_breaks": int(g["n_breaks"].sum()), "n_transitions": int(trans),
                    "route_precision_mean": float(g["route_precision"].mean()),
                    "route_recall_mean": float(g["route_recall"].mean()),
                    "route_f1_mean": float(g["route_f1"].mean())})
    return pd.DataFrame(out)
