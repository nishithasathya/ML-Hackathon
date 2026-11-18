"""Artefact bundle + end-to-end label-free prediction (integration lead, inference round)."""
import json

import pytest

from src.common.io import load_trip, load_trip_for_inference
from src.inference.artifacts import ArtifactError, load_bundle
from src.inference.predictor import predict_trip


def test_bundle_loads_and_predicts_without_labels(tiny_bundle, multi_root):
    bundle = load_bundle(tiny_bundle)
    raw = load_trip_for_inference(multi_root / "student_test", "00000020")
    result = predict_trip(raw, bundle)
    # every fix that has candidates is decoded (the synthetic 300 m spike has none)
    assert len(result.hmm) == result.candidates["obs_idx"].nunique() == len(result.pre.track) - 1
    assert set(result.timings_s) >= {"preprocess", "candidates_features", "classifier", "hmm_viterbi", "total"}
    assert "is_reference" not in result.candidates.columns
    labelled = load_trip("student_test", "00000020", multi_root)
    with pytest.raises(ValueError):
        predict_trip(labelled, bundle)  # refuses a trip that carries its route


def test_bundle_integrity_checks(tiny_bundle, tmp_path):
    with pytest.raises(ArtifactError):
        load_bundle(tmp_path / "does_not_exist")
    meta = json.loads((tiny_bundle / "bundle.json").read_text())
    meta["model"]["sha256"] = "0" * 64
    (tiny_bundle / "bundle.json").write_text(json.dumps(meta))
    with pytest.raises(ArtifactError):
        load_bundle(tiny_bundle)
