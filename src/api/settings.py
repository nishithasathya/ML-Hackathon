"""API configuration, read from environment variables (see .env.example).

PATHSNAP_EVAL_DATA_ROOT   folder with the trips to serve: <root>/<trip_id>/<trip_id>.{track,nodes,arcs,route}
                          (the organisers' 10 evaluator trips on Laptop 1)
PATHSNAP_DATASET_LABEL    label shown in the UI, e.g. "evaluator_test" or "student_test (rehearsal)"
PATHSNAP_ARTIFACT_DIR     exported model bundle (default: <repo>/artifacts)
PATHSNAP_API_HOST / PORT  bind address for uvicorn (0.0.0.0 to accept Laptop 2)
PATHSNAP_CORS_ORIGINS     comma-separated allowed origins (default "*")
PATHSNAP_LOG_DIR          where api.log is written (default: <repo>/logs)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.common.config import PROJECT_ROOT, get_artifact_dir, get_eval_data_root

EXPECTED_EVALUATION_TRIPS = 10


@dataclass
class Settings:
    eval_data_root: Optional[Path]
    dataset_label: str
    artifact_dir: Path
    cors_origins: List[str]
    log_dir: Path
    max_network_arcs: int = 4000

    @classmethod
    def from_env(cls) -> "Settings":
        cors = os.environ.get("PATHSNAP_CORS_ORIGINS", "*")
        return cls(eval_data_root=get_eval_data_root(),
                   dataset_label=os.environ.get("PATHSNAP_DATASET_LABEL", "evaluator_test"),
                   artifact_dir=get_artifact_dir(),
                   cors_origins=[c.strip() for c in cors.split(",") if c.strip()],
                   log_dir=Path(os.environ.get("PATHSNAP_LOG_DIR", PROJECT_ROOT / "logs")))
