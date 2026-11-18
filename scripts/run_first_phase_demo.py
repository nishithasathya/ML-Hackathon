"""First-phase demonstration: run the whole implemented pipeline on ONE trip.

    python scripts/run_first_phase_demo.py --partition train --trip 00000010

Steps (each prints what it found):
    1. load raw files           (src.common.io)
    2. validate                 (src.data_exploration.validation)
    3. statistics               (src.data_exploration.statistics)
    4. preprocess               (src.preprocessing.pipeline)
    5. candidates + features    (src.features)
    6. Folium map + static plot (src.visualization)

No model is trained and no route is predicted: that is deliberately left for
the next evaluation phases.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow "python scripts/run_first_phase_demo.py" from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import ALLOWED_PARTITIONS, get_output_dir  # noqa: E402
from src.common.io import load_trip  # noqa: E402
from src.data_exploration.statistics import trip_statistics  # noqa: E402
from src.data_exploration.validation import validate_trip  # noqa: E402
from src.features.candidates import CandidateConfig  # noqa: E402
from src.features.features import build_trip_features  # noqa: E402
from src.preprocessing.pipeline import preprocess_trip  # noqa: E402
from src.visualization.folium_maps import build_trip_map, save_trip_map  # noqa: E402
from src.visualization.plots import plot_trip_static  # noqa: E402


def step(n: int, title: str) -> float:
    print(f"\n[{n}/6] {title}")
    return time.perf_counter()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="PathSnap first-phase demo on one trip")
    parser.add_argument("--partition", default="train", choices=ALLOWED_PARTITIONS)
    parser.add_argument("--trip", default="00000010")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--radius", type=float, default=CandidateConfig().radius_m)
    args = parser.parse_args(argv)
    out = get_output_dir("demo")

    t0 = step(1, "Loading raw files")
    raw = load_trip(args.partition, args.trip, args.data_root)
    print(f"  {raw.n_observations} GPS fixes, {raw.n_nodes} nodes, {raw.n_arcs} arcs, {raw.route.size} route arcs "
          f"({time.perf_counter() - t0:.2f}s)")

    step(2, "Validating")
    issues = validate_trip(raw)
    for issue in issues:
        print(f"  [{issue.severity}] {issue.file}/{issue.code}: {issue.count} - {issue.message}")
    if not issues:
        print("  no issues found")

    step(3, "Trip statistics")
    stats = trip_statistics(raw)
    for key in ("duration_s", "path_length_km", "frac_dt_1s", "max_gap_s", "speed_p95_mps", "n_weak_components",
                "reciprocal_arc_frac", "route_continuity_breaks", "gps_route_dist_median_m", "gps_route_dist_p95_m"):
        print(f"  {key:28s} {stats.get(key, float('nan')):.3f}")

    step(4, "Preprocessing")
    pre = preprocess_trip(raw)
    for key, value in pre.report.items():
        if key not in ("partition", "trip_id"):
            print(f"  {key:28s} {value}")

    t0 = step(5, f"Candidate generation (radius {args.radius:g} m) and features")
    table, summary = build_trip_features(pre, CandidateConfig(radius_m=args.radius))
    print(f"  {len(table)} (observation, candidate) rows x {table.shape[1]} columns "
          f"({time.perf_counter() - t0:.2f}s)")
    for key in ("mean_candidates_per_obs", "frac_obs_without_candidates", "candidate_miss_rate",
                "nearest_arc_is_reference_rate", "nearest_arc_is_reverse_of_reference_rate", "ref_distance_median_m"):
        print(f"  {key:42s} {summary[key]:.4f}")
    feature_path = out / f"{args.partition}_{args.trip}_features.csv"
    table.to_csv(feature_path, index=False)
    print(f"  feature table -> {feature_path}")

    step(6, "Visualisation")
    # Show the candidate arcs of five evenly spaced observations on the map.
    positions = [round((len(pre.track) - 1) * k / 4) for k in range(5)]
    sample_obs = pre.track["obs_idx"].iloc[positions].tolist()
    shown = table[table["obs_idx"].isin(sample_obs)]
    map_path = save_trip_map(build_trip_map(raw, track=pre.track, candidates=shown),
                             out / f"{args.partition}_{args.trip}_map.html")
    png_path = plot_trip_static(raw, out / f"{args.partition}_{args.trip}.png")
    print(f"  map  -> {map_path}\n  plot -> {png_path}")
    print("\nDemo finished. No model was trained; prediction is planned for later phases.")


if __name__ == "__main__":
    main()
