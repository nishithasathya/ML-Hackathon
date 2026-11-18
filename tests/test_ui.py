"""Evaluator UI: API client against the real FastAPI app, and the Streamlit page itself."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.settings import Settings

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
sys.path.insert(0, str(FRONTEND))
from api_client import ApiError, PathSnapClient, metrics_rows  # noqa: E402


@pytest.fixture
def api(multi_root, tiny_bundle, tmp_path):
    settings = Settings(eval_data_root=multi_root / "student_test", dataset_label="synthetic (test)",
                        artifact_dir=tiny_bundle, cors_origins=["*"], log_dir=tmp_path / "logs")
    with TestClient(create_app(settings)) as c:
        yield PathSnapClient("http://testserver", client=c)


def test_client_lists_trips_and_predicts(api):
    assert api.health()["status"] == "ok"
    trips = api.trips()
    assert trips == ["00000020", "00000021"]
    res = api.predict(trips[1])
    rows = metrics_rows(res["evaluation"])
    assert [r["method"] for r in rows] == ["Classifier + HMM/Viterbi (final)", "Classifier only"]
    assert api.results(trips[1])["available"]


def test_client_raises_readable_errors(api):
    with pytest.raises(ApiError, match="404"):
        api.predict("12345678")


def test_streamlit_page_reports_unreachable_api():
    from streamlit.testing.v1 import AppTest
    import os
    os.environ["PATHSNAP_API_URL"] = "http://127.0.0.1:9"  # nothing listens on the discard port
    app = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=30).run()
    assert any("Cannot reach the API" in e.value for e in app.error)
