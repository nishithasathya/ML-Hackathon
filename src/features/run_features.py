"""Command-line entry point for candidate generation and feature extraction.

Usage (from the repository root):
    python -m src.features.run_features --partition train
    python -m src.features.run_features --partition student_test
    python -m src.features.run_features --partition train --sweep 25 50 75 100

Outputs (default outputs/features/<partition>/):
    <trip_id>_features.csv.gz   one row per (observation, candidate arc) + is_reference label
    feature_summary.csv         per trip: candidate counts, candidate miss rate, ...
    radius_sweep.csv            (--sweep only) miss rate vs radius, TRAIN partition only

The sweep is how the candidate radius is chosen: it uses training trips only,
so the student-test trips stay untouched for later model comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import ALLOWED_PARTITIONS, TRAIN, get_output_dir
from src.common.io import iter_trips
from src.features.candidates import CandidateConfig
from src.features.features import build_trip_features
from src.features.labels import align_track_to_route
from src.features.spatial_index import ArcIndex
from src.preprocessing.pipeline import preprocess_trip


OFFSET_TRIP_THRESHOLD_M = 30.0


def run_sweep(radii, max_candidates: int, data_root, trips, out: Path) -> None:
    rows = []
    for raw in iter_trips(TRAIN, data_root, trips):
        pre = preprocess_trip(raw)
        # Index and reference labels do not depend on the radius: build once, reuse.
        index = ArcIndex(pre.nodes_xy, pre.arcs)
        labels = align_track_to_route(pre.track, pre.nodes_xy, pre.arcs, pre.route)
        for radius in radii:
            _, summary = build_trip_features(pre, CandidateConfig(radius_m=radius, max_candidates=max_candidates),
                                             index=index, labels=labels)
            rows.append(summary)
        print(f"  sweep {raw.trip_id} done", flush=True)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / "radius_sweep_per_trip.csv", index=False)
    # Trips whose GPS is systematically far from the reference route (median
    # aligned distance > OFFSET_TRIP_THRESHOLD_M) dominate the miss rate, so the
    # sweep is reported both with and without them.
    offset = sweep["ref_distance_median_m"] > OFFSET_TRIP_THRESHOLD_M
    tables = [_weighted_sweep(sweep).assign(subset="all_train_trips"),
              _weighted_sweep(sweep[~offset]).assign(subset=f"excluding_offset_trips(>{OFFSET_TRIP_THRESHOLD_M:g}m)")]
    agg = pd.concat(tables, ignore_index=True)
    agg.to_csv(out / "radius_sweep.csv", index=False)
    print("Offset trips:", sorted(sweep.loc[offset, "trip_id"].unique()))
    print(agg.to_string(index=False))


def _weighted_sweep(sweep: pd.DataFrame) -> pd.DataFrame:
    """Observation-weighted averages per radius."""
    rows = []
    for radius, g in sweep.groupby("radius_m"):
        w = g["n_observations"]
        rows.append({"radius_m": radius, "n_trips": len(g), "n_observations": int(w.sum()),
                     "candidate_miss_rate": (g.candidate_miss_rate * w).sum() / w.sum(),
                     "frac_obs_without_candidates": (g.frac_obs_without_candidates * w).sum() / w.sum(),
                     "mean_candidates_per_obs": (g.mean_candidates_per_obs * w).sum() / w.sum()})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    defaults = CandidateConfig()
    parser = argparse.ArgumentParser(description="PathSnap candidate generation and features")
    parser.add_argument("--partition", required=True, choices=ALLOWED_PARTITIONS)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--trips", nargs="*", default=None)
    parser.add_argument("--radius", type=float, default=defaults.radius_m)
    parser.add_argument("--max-candidates", type=int, default=defaults.max_candidates)
    parser.add_argument("--sweep", nargs="*", type=float, default=None,
                        help="radii to compare (train partition only); no feature files are written")
    args = parser.parse_args(argv)

    out = Path(args.output_dir) if args.output_dir else get_output_dir("features", args.partition)
    out.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        if args.partition != TRAIN:
            parser.error("--sweep is only allowed on the train partition")
        run_sweep(args.sweep, args.max_candidates, args.data_root, args.trips, out)
        return

    config = CandidateConfig(radius_m=args.radius, max_candidates=args.max_candidates)
    summaries = []
    for raw in iter_trips(args.partition, args.data_root, args.trips):
        table, summary = build_trip_features(preprocess_trip(raw), config)
        table.to_csv(out / f"{raw.trip_id}_features.csv.gz", index=False, compression="gzip")
        summaries.append(summary)
        print(f"  {raw.trip_id}: {summary['n_candidate_rows']} rows, miss rate {summary['candidate_miss_rate']:.3f}",
              flush=True)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out / "feature_summary.csv", index=False)
    weights = summary_df["n_observations"]
    for col in ("candidate_miss_rate", "nearest_arc_is_reference_rate", "nearest_arc_is_reverse_of_reference_rate",
                "mean_candidates_per_obs"):
        print(f"{col} (observation-weighted): {(summary_df[col] * weights).sum() / weights.sum():.4f}")
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
