"""Request/response schemas. Note what is NOT here: no reference route, no labels."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_key: Optional[str] = None
    bundle_version: Optional[str] = None
    dataset_label: str
    eval_data_configured: bool
    n_trips_available: int
    detail: Optional[str] = None


class TripInfo(BaseModel):
    trip_id: str


class TripListResponse(BaseModel):
    dataset_label: str
    count: int
    expected_count: int
    trips: List[TripInfo]
    warning: Optional[str] = None


class PredictRequest(BaseModel):
    trip_id: str = Field(..., pattern=r"^[A-Za-z0-9_\-]{1,64}$", examples=["00000052"])


class ObservationPrediction(BaseModel):
    obs_idx: int
    lat: float
    lon: float
    arc_id_hmm: Optional[int] = None
    arc_id_classifier: Optional[int] = None


class MethodMetrics(BaseModel):
    top1_accuracy: Optional[float] = None
    route_continuity: Optional[float] = None
    route_precision: Optional[float] = None
    route_recall: Optional[float] = None
    route_f1: Optional[float] = None
    n_breaks: Optional[int] = None


class EvaluationResult(BaseModel):
    trip_id: str
    available: bool
    detail: Optional[str] = None
    n_labelled_observations: Optional[int] = None
    hmm: Optional[MethodMetrics] = None
    classifier_only: Optional[MethodMetrics] = None


class PredictResponse(BaseModel):
    trip_id: str
    dataset_label: str
    model_key: str
    n_observations: int
    n_candidates: int
    n_segments: int
    timings_s: Dict[str, float]
    predicted_arc_sequence: List[int]
    predicted_route_lines: List[List[List[float]]]  # polylines of [lat, lon]
    network_lines: List[List[List[float]]]  # nearby road network, [lat, lon] pairs
    observations: List[ObservationPrediction]
    evaluation: EvaluationResult
