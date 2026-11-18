"""Folium maps of PREDICTED routes for development trips (student_test or train).

    python -m src.visualization.prediction_maps --partition student_test --trips 00000052 00000040

Uses the exported artefact bundle and the label-free prediction pipeline, then
draws: road network, reference route (development partition only - this is
allowed for analysis; the evaluator UI never receives it), raw GPS fixes,
classifier-only arcs and the HMM/Viterbi route, with a layer control and legend.
Output: results/maps/<partition>_<trip>_prediction.html
"""
from __future__ import annotations

import argparse

from src.common.config import get_data_root, get_results_dir
from src.common.io import load_trip, load_trip_for_inference
from src.inference.artifacts import load_bundle
from src.inference.predictor import predict_trip
from src.visualization.folium_maps import build_trip_map, save_trip_map


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--partition", default="student_test")
    parser.add_argument("--trips", nargs="+", required=True)
    args = parser.parse_args(argv)
    bundle = load_bundle()
    out = get_results_dir("maps")
    for trip in args.trips:
        raw_labelled = load_trip(args.partition, trip)  # for the reference layer only
        raw = load_trip_for_inference(get_data_root() / args.partition, trip)
        result = predict_trip(raw, bundle)
        fmap = build_trip_map(raw_labelled, track=result.pre.track,
                              predicted_arc_ids=result.hmm.sort_values("obs_idx")["arc_id"].tolist(),
                              classifier_arc_ids=result.classifier["arc_id"].tolist())
        path = save_trip_map(fmap, out / f"{args.partition}_{trip}_prediction.html")
        print(f"  {trip}: {path}")


if __name__ == "__main__":
    main()
