"""Command-line entry point: clean every trip of a partition and save the result.

Usage (from the repository root):
    python -m src.preprocessing.run_preprocessing --partition train
    python -m src.preprocessing.run_preprocessing --partition student_test --gap-threshold 30

Outputs (default outputs/preprocessed/<partition>/):
    <trip_id>.csv               cleaned track (columns listed in pipeline.PreprocessedTrip)
    preprocessing_report.csv    one row per trip: what was dropped / flagged
    preprocess_config.json      the exact configuration used (for reproducibility)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common.config import ALLOWED_PARTITIONS, get_output_dir
from src.common.io import iter_trips
from src.preprocessing.pipeline import PreprocessConfig, config_as_dict, preprocess_trip


def main(argv=None) -> None:
    defaults = PreprocessConfig()
    parser = argparse.ArgumentParser(description="PathSnap preprocessing")
    parser.add_argument("--partition", required=True, choices=ALLOWED_PARTITIONS)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--trips", nargs="*", default=None, help="optional subset of trip ids")
    parser.add_argument("--gap-threshold", type=float, default=defaults.gap_threshold_s)
    parser.add_argument("--max-speed", type=float, default=defaults.max_speed_mps)
    parser.add_argument("--drop-speed-outliers", action="store_true")
    args = parser.parse_args(argv)

    config = PreprocessConfig(gap_threshold_s=args.gap_threshold, max_speed_mps=args.max_speed,
                              drop_speed_outliers=args.drop_speed_outliers)
    out = Path(args.output_dir) if args.output_dir else get_output_dir("preprocessed", args.partition)
    out.mkdir(parents=True, exist_ok=True)

    reports = []
    for raw in iter_trips(args.partition, args.data_root, args.trips):
        pre = preprocess_trip(raw, config)
        pre.track.to_csv(out / f"{raw.trip_id}.csv", index=False)
        reports.append(pre.report)
        print(f"  {raw.trip_id}: {pre.report['n_clean_observations']} obs, {pre.report['n_segments']} segment(s), "
              f"{pre.report['n_speed_outliers']} speed outlier(s)")

    report = pd.DataFrame(reports)
    report.to_csv(out / "preprocessing_report.csv", index=False)
    (out / "preprocess_config.json").write_text(json.dumps(config_as_dict(config), indent=2))
    numeric = report.drop(columns=["partition", "trip_id"]).sum()
    print("Totals:\n" + numeric.to_string())
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
