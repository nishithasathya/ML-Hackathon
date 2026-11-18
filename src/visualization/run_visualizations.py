"""Command-line entry point for maps and figures.

Usage (from the repository root):
    python -m src.visualization.run_visualizations maps --partition train --trips 00000010 00000033
    python -m src.visualization.run_visualizations figures

``maps``    writes outputs/maps/<partition>_<trip>.html (Folium) and
            outputs/figures/trip_<partition>_<trip>.png (static) for each trip.
``figures`` writes dataset-level PNGs to outputs/figures/. It needs the EDA
            outputs (run src.data_exploration.run_eda first) and, for the radius
            figure, outputs/features/train/radius_sweep.csv. Missing inputs are
            skipped with a message rather than faked.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.common.config import ALLOWED_PARTITIONS, get_output_dir
from src.common.io import list_trips, load_trip
from src.preprocessing.pipeline import preprocess_trip
from src.visualization import plots
from src.visualization.folium_maps import build_trip_map, save_trip_map


def make_maps(partition: str, trips, data_root, out_root) -> None:
    trip_ids = trips or list_trips(partition, data_root)[:3]
    for trip_id in trip_ids:
        raw = load_trip(partition, trip_id, data_root)
        pre = preprocess_trip(raw)
        html = save_trip_map(build_trip_map(raw, track=pre.track),
                             get_output_dir("maps", root=out_root) / f"{partition}_{trip_id}.html")
        png = plots.plot_trip_static(raw, get_output_dir("figures", root=out_root) / f"trip_{partition}_{trip_id}.png")
        print(f"  {trip_id}: {html.name}, {png.name}")


def make_figures(data_root, eda_dir: Path, sweep_csv: Path, out_root) -> None:
    fig_dir = get_output_dir("figures", root=out_root)
    stats_csv = eda_dir / "trip_statistics.csv"
    if stats_csv.exists():
        stats = plots.read_csv_keep_ids(stats_csv)
        plots.plot_trip_locations(stats, fig_dir / "trip_locations.png")
        plots.plot_distribution(stats, "n_observations", fig_dir / "observations_per_trip.png",
                                "GPS observations per trip", "observations", log_x=True)
        plots.plot_distribution(stats, "gps_route_dist_median_m", fig_dir / "gps_route_distance.png",
                                "Median GPS distance to reference route", "metres (log scale)", log_x=True)
        plots.plot_distribution(stats, "max_gap_s", fig_dir / "largest_gap_per_trip.png",
                                "Largest time gap per trip", "seconds (log scale)", log_x=True)
        plots.plot_network_size(stats, fig_dir / "network_size.png")
        print("  dataset figures written")
    else:
        print(f"  skipped dataset figures: {stats_csv} not found (run run_eda first)")

    issues_csv = eda_dir / "validation_issues.csv"
    if issues_csv.exists():
        issues = plots.read_csv_keep_ids(issues_csv)
        if len(issues):
            plots.plot_validation_issues(issues, fig_dir / "validation_issues.png")
            print("  validation figure written")
    else:
        print(f"  skipped validation figure: {issues_csv} not found")

    counts = plots.interval_category_counts(ALLOWED_PARTITIONS, data_root)
    counts.to_csv(fig_dir / "sampling_interval_counts.csv")
    plots.plot_sampling_intervals(counts, fig_dir / "sampling_intervals.png")
    print("  sampling-interval figure written")

    if sweep_csv.exists():
        plots.plot_radius_sweep(plots.read_csv_keep_ids(sweep_csv), fig_dir / "radius_sweep.png")
        print("  radius-sweep figure written")
    else:
        print(f"  skipped radius sweep: {sweep_csv} not found (run run_features --sweep)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="PathSnap visualisations")
    parser.add_argument("command", choices=["maps", "figures"])
    parser.add_argument("--partition", default="train", choices=ALLOWED_PARTITIONS)
    parser.add_argument("--trips", nargs="*", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None, help="default: outputs/")
    args = parser.parse_args(argv)

    if args.command == "maps":
        make_maps(args.partition, args.trips, args.data_root, args.output_root)
    else:
        make_figures(args.data_root, get_output_dir("eda", root=args.output_root, create=False),
                     get_output_dir("features", "train", root=args.output_root, create=False) / "radius_sweep.csv",
                     args.output_root)


if __name__ == "__main__":
    main()
