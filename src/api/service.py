"""Prediction service: loads artefacts once, serves trips from the configured directory."""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from src.api.evaluator import evaluate_prediction
from src.api.settings import Settings
from src.common.io import list_trip_dirs, load_trip_for_inference
from src.inference.artifacts import ArtifactError, Bundle, load_bundle
from src.inference.predictor import predict_trip
from src.visualization.folium_maps import nearby_arc_ids

log = logging.getLogger("pathsnap.api")


class ServiceUnavailable(RuntimeError):
    pass


class TripNotFound(KeyError):
    pass


def _clean(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


class PredictionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bundle: Optional[Bundle] = None
        self.load_error: Optional[str] = None
        self.results: Dict[str, dict] = {}
        self._lock = threading.Lock()

    # -------------------------------------------------------------- startup
    def load(self) -> None:
        try:
            start = time.perf_counter()
            self.bundle = load_bundle(self.settings.artifact_dir)
            meta = self.bundle.metadata
            log.info("model loaded: key=%s bundle=%s sha256=%s... (%.2fs) from %s", meta["model"]["key"],
                     meta["bundle_version"], meta["model"]["sha256"][:12], time.perf_counter() - start,
                     self.settings.artifact_dir)
        except ArtifactError as exc:
            self.load_error = str(exc)
            log.error("model NOT loaded: %s", exc)
        root = self.settings.eval_data_root
        if root is None:
            log.warning("PATHSNAP_EVAL_DATA_ROOT is not set: no evaluation trips can be served")
        elif not root.is_dir():
            log.warning("evaluation data folder does not exist: %s", root)
        else:
            log.info("serving '%s' trips from %s: %s", self.settings.dataset_label, root, self.trip_ids())

    # -------------------------------------------------------------- trips
    def trip_ids(self) -> List[str]:
        root = self.settings.eval_data_root
        if root is None or not root.is_dir():
            return []
        return list_trip_dirs(root)

    def require_ready(self) -> None:
        if self.bundle is None:
            raise ServiceUnavailable(self.load_error or "Model bundle not loaded")
        if self.settings.eval_data_root is None:
            raise ServiceUnavailable("Evaluation data not configured: set PATHSNAP_EVAL_DATA_ROOT on Laptop 1")
        if not self.settings.eval_data_root.is_dir():
            raise ServiceUnavailable(f"Evaluation data folder not found: {self.settings.eval_data_root}")

    # -------------------------------------------------------------- predict
    def predict(self, trip_id: str) -> dict:
        self.require_ready()
        if trip_id not in self.trip_ids():
            raise TripNotFound(trip_id)
        assert self.bundle is not None
        log.info("[%s] loading trip (track, nodes, arcs only - no reference route)", trip_id)
        raw = load_trip_for_inference(self.settings.eval_data_root, trip_id, partition=self.settings.dataset_label)
        log.info("[%s] %d GPS fixes, %d nodes, %d arcs", trip_id, raw.n_observations, raw.n_nodes, raw.n_arcs)
        with self._lock:  # one prediction at a time keeps logs readable and memory bounded
            result = predict_trip(raw, self.bundle, log=lambda m: log.info("[%s] %s", trip_id, m))
        log.info("[%s] inference finished in %.2fs (%s)", trip_id, result.timings_s["total"],
                 ", ".join(f"{k}={v:.2f}s" for k, v in result.timings_s.items() if k != "total"))

        evaluation = _clean(evaluate_prediction(result, self.settings.eval_data_root))
        if evaluation.get("available"):
            log.info("[%s] server-side evaluation: HMM top1=%.4f continuity=%.4f | classifier-only top1=%.4f",
                     trip_id, evaluation["hmm"]["top1_accuracy"], evaluation["hmm"]["route_continuity"] or 0,
                     evaluation["classifier_only"]["top1_accuracy"])
        self.results[trip_id] = evaluation
        return self._response(raw, result, evaluation)

    def _response(self, raw, result, evaluation: dict) -> dict:
        pre = result.pre
        nodes = raw.nodes
        hmm = result.hmm.sort_values("obs_idx")
        seq = hmm["arc_id"].to_numpy()
        seq = seq[np.r_[True, seq[1:] != seq[:-1]]].tolist() if len(seq) else []
        lines, current = [], []
        for a in seq:  # merge connected consecutive arcs into polylines
            u, v = raw.arcs[a]
            if current and current[-1] == [float(nodes[u, 1]), float(nodes[u, 0])]:
                current.append([float(nodes[v, 1]), float(nodes[v, 0])])
            else:
                if current:
                    lines.append(current)
                current = [[float(nodes[u, 1]), float(nodes[u, 0])], [float(nodes[v, 1]), float(nodes[v, 0])]]
        if current:
            lines.append(current)
        net_ids = nearby_arc_ids(raw, buffer_m=120.0, max_arcs=self.settings.max_network_arcs)
        network = [[[float(nodes[u, 1]), float(nodes[u, 0])], [float(nodes[v, 1]), float(nodes[v, 0])]]
                   for u, v in raw.arcs[net_ids]]
        obs = pre.track[["obs_idx", "lat", "lon"]].merge(
            result.hmm[["obs_idx", "arc_id"]].rename(columns={"arc_id": "arc_id_hmm"}), on="obs_idx", how="left").merge(
            result.classifier[["obs_idx", "arc_id"]].rename(columns={"arc_id": "arc_id_classifier"}), on="obs_idx",
            how="left")
        observations = [{"obs_idx": int(r.obs_idx), "lat": float(r.lat), "lon": float(r.lon),
                         "arc_id_hmm": None if np.isnan(r.arc_id_hmm) else int(r.arc_id_hmm),
                         "arc_id_classifier": None if np.isnan(r.arc_id_classifier) else int(r.arc_id_classifier)}
                        for r in obs.itertuples()]
        return {"trip_id": result.trip_id, "dataset_label": self.settings.dataset_label,
                "model_key": self.bundle.metadata["model"]["key"], "n_observations": result.stats["n_observations"],
                "n_candidates": result.stats["n_candidates"], "n_segments": result.stats["n_segments"],
                "timings_s": {k: round(v, 4) for k, v in result.timings_s.items()},
                "predicted_arc_sequence": [int(a) for a in seq], "predicted_route_lines": lines,
                "network_lines": network, "observations": observations, "evaluation": evaluation}
