"""HTTP client for the PathSnap API (used by the Streamlit UI on Laptop 2).

The UI never reads data or model files: everything it shows comes from these calls.
``client`` can be any httpx-compatible client (tests pass FastAPI's TestClient).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_API_URL = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000")


class ApiError(RuntimeError):
    pass


class PathSnapClient:
    def __init__(self, base_url: str = DEFAULT_API_URL, timeout_s: float = 300.0, client: Optional[Any] = None):
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def _get(self, path: str) -> Dict[str, Any]:
        return self._handle(self._client.get(path))

    def _handle(self, response) -> Dict[str, Any]:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = response.text
            raise ApiError(f"HTTP {response.status_code}: {detail}")
        return response.json()

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def trips(self) -> List[str]:
        return [t["trip_id"] for t in self._get("/api/evaluation/trips")["trips"]]

    def trip_listing(self) -> Dict[str, Any]:
        return self._get("/api/evaluation/trips")

    def predict(self, trip_id: str) -> Dict[str, Any]:
        return self._handle(self._client.post("/api/predict", json={"trip_id": trip_id}))

    def results(self, trip_id: str) -> Dict[str, Any]:
        return self._get(f"/api/evaluation/results/{trip_id}")


def metrics_rows(evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the API's evaluation block into table rows for display."""
    if not evaluation or not evaluation.get("available"):
        return []
    rows = []
    for key, label in (("hmm", "Classifier + HMM/Viterbi (final)"), ("classifier_only", "Classifier only")):
        m = evaluation.get(key) or {}
        rows.append({"method": label, "Top-1 accuracy": m.get("top1_accuracy"),
                     "Route continuity": m.get("route_continuity"), "Route precision": m.get("route_precision"),
                     "Route recall": m.get("route_recall"), "Route F1": m.get("route_f1"),
                     "Continuity breaks": m.get("n_breaks")})
    return rows
