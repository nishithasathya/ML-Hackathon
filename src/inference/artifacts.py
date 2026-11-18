"""Export and load the inference artefact bundle.

artifacts/
    model.joblib     fitted scikit-learn Pipeline (imputer/scaler inside, if any)
    bundle.json      everything else inference needs: feature list, candidate,
                     preprocessing and HMM configuration, model metadata, seed,
                     version, training-set description and the model file hash.

The API loads this directory once at start-up (``PATHSNAP_ARTIFACT_DIR``
overrides the location). Nothing is retrained at inference time.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.common.config import RANDOM_SEED, get_artifact_dir
from src.features.candidates import CandidateConfig
from src.features.features import FEATURE_COLUMNS
from src.hmm.decoder import HMMConfig
from src.preprocessing.pipeline import PreprocessConfig

BUNDLE_VERSION = "2.0.0"


class ArtifactError(RuntimeError):
    """Raised when the artefact bundle is missing or inconsistent."""


@dataclass
class Bundle:
    model: Any
    feature_columns: list
    candidate_config: CandidateConfig
    preprocess_config: PreprocessConfig
    hmm_config: HMMConfig
    metadata: Dict[str, Any]
    directory: Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_bundle(model_path: Path, model_key: str, model_params: Dict[str, Any], hmm_config: HMMConfig,
                  metrics: Dict[str, Any], candidate_config: CandidateConfig = CandidateConfig(),
                  preprocess_config: PreprocessConfig = PreprocessConfig(), training: Optional[Dict[str, Any]] = None,
                  directory: Optional[Path] = None) -> Path:
    out = Path(directory) if directory else get_artifact_dir()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "model.joblib"
    shutil.copyfile(model_path, target)
    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"key": model_key, "params": model_params, "file": "model.joblib", "sha256": _sha256(target)},
        "feature_columns": FEATURE_COLUMNS,
        "candidate_config": asdict(candidate_config),
        "preprocess_config": asdict(preprocess_config),
        "hmm_config": hmm_config.as_dict(),
        "seed": RANDOM_SEED,
        "training": training or {},
        "metrics": metrics,
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=2, default=str))
    return out


def load_bundle(directory: Optional[Path] = None) -> Bundle:
    folder = Path(directory) if directory else get_artifact_dir()
    meta_path, model_path = folder / "bundle.json", folder / "model.joblib"
    if not meta_path.exists() or not model_path.exists():
        raise ArtifactError(f"Artefact bundle not found in {folder}: need bundle.json and model.joblib. "
                            f"Run `python scripts/build_final_artifacts.py` or set PATHSNAP_ARTIFACT_DIR.")
    meta = json.loads(meta_path.read_text())
    if meta["model"].get("sha256") and meta["model"]["sha256"] != _sha256(model_path):
        raise ArtifactError("model.joblib does not match the hash recorded in bundle.json")
    if meta["feature_columns"] != FEATURE_COLUMNS:
        raise ArtifactError("Feature list in bundle.json differs from the code's FEATURE_COLUMNS; "
                            "the model was trained with a different feature version.")
    return Bundle(model=joblib.load(model_path), feature_columns=meta["feature_columns"],
                  candidate_config=CandidateConfig(**meta["candidate_config"]),
                  preprocess_config=PreprocessConfig(**meta["preprocess_config"]),
                  hmm_config=HMMConfig(**meta["hmm_config"]), metadata=meta, directory=folder)
