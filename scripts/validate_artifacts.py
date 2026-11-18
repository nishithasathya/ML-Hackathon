"""Check that the inference bundle loads and runs, without retraining anything.

    python scripts/validate_artifacts.py [--trip-dir data/raw/student_test --trip 00000052]

1. loads artifacts/ (hash and feature-list checks happen inside load_bundle);
2. optionally runs the full label-free prediction on one trip and prints timings.
Exit code 0 = OK, 1 = problem (message printed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import list_trip_dirs, load_trip_for_inference  # noqa: E402
from src.inference.artifacts import ArtifactError, load_bundle  # noqa: E402
from src.inference.predictor import predict_trip  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--trip-dir", default=None, help="folder containing <trip_id>/ sub-folders")
    parser.add_argument("--trip", default=None)
    args = parser.parse_args(argv)
    try:
        bundle = load_bundle(Path(args.artifact_dir) if args.artifact_dir else None)
    except ArtifactError as exc:
        print(f"FAIL: {exc}")
        return 1
    meta = bundle.metadata
    print(f"OK: model '{meta['model']['key']}' (bundle {meta['bundle_version']}, created {meta['created_utc']})")
    print(f"    {len(bundle.feature_columns)} features, radius {bundle.candidate_config.radius_m} m, "
          f"HMM w={bundle.hmm_config.emission_weight} beta={bundle.hmm_config.transition_beta_m} m")
    if args.trip_dir:
        trip = args.trip or list_trip_dirs(args.trip_dir)[0]
        raw = load_trip_for_inference(args.trip_dir, trip)
        result = predict_trip(raw, bundle, log=lambda m: print(f"    {m}"))
        print(f"OK: trip {trip}: {len(result.hmm)} observations decoded in {result.timings_s['total']:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
