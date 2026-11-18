"""FastAPI backend: health, trip listing, prediction, errors, and that no labels leak out."""
import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.settings import Settings


def _client(eval_root, artifact_dir, tmp_path):
    settings = Settings(eval_data_root=eval_root, dataset_label="synthetic (test)", artifact_dir=artifact_dir,
                        cors_origins=["*"], log_dir=tmp_path / "logs")
    return TestClient(create_app(settings))


@pytest.fixture
def client(multi_root, tiny_bundle, tmp_path):
    with _client(multi_root / "student_test", tiny_bundle, tmp_path) as c:
        yield c


def test_health_and_trip_listing(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] and h["n_trips_available"] == 2
    listing = client.get("/api/evaluation/trips").json()
    assert [t["trip_id"] for t in listing["trips"]] == ["00000020", "00000021"]
    assert listing["warning"]  # 2 trips instead of the expected 10 is reported, not hidden


def test_predict_returns_route_and_metrics_but_no_reference(client):
    r = client.post("/api/predict", json={"trip_id": "00000020"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_observations"] > 0 and body["predicted_route_lines"] and body["observations"]
    assert body["evaluation"]["available"] and 0 <= body["evaluation"]["hmm"]["top1_accuracy"] <= 1
    text = json.dumps(body)
    assert "ref_arc_id" not in text and "is_reference" not in text and '"route"' not in text
    stored = client.get("/api/evaluation/results/00000020").json()
    assert stored["hmm"]["top1_accuracy"] == body["evaluation"]["hmm"]["top1_accuracy"]


def test_invalid_and_unknown_trip_ids(client):
    assert client.post("/api/predict", json={"trip_id": "../../etc"}).status_code == 422
    assert client.post("/api/predict", json={"trip_id": "99999999"}).status_code == 404
    assert client.get("/api/evaluation/results/00000021").status_code == 404


def test_missing_artifacts_gives_503(multi_root, tmp_path):
    with _client(multi_root / "student_test", tmp_path / "no_artifacts", tmp_path) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/api/predict", json={"trip_id": "00000020"}).status_code == 503


def test_missing_evaluation_data_fails_clearly(tiny_bundle, tmp_path):
    with _client(None, tiny_bundle, tmp_path) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.get("/api/evaluation/trips").status_code == 503
        assert c.post("/api/predict", json={"trip_id": "00000020"}).status_code == 503


def test_requests_are_logged(client, tmp_path):
    client.get("/health")
    log = (tmp_path / "logs" / "api.log").read_text()
    assert "REQUEST  GET /health" in log and "model loaded" in log
