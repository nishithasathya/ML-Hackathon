"""HMM/Viterbi experiments: decoder tuning and classifier-only vs classifier+HMM.

    python -m src.hmm.run_hmm --model hgb --probs tuned

1. Decoder parameters (emission weight w, transition scale beta) are chosen on
   TRAINING trips using OUT-OF-FOLD classifier probabilities (each trip scored
   by a model that never saw it), so the decoder is tuned on honest scores.
   Tuning trips: every 3rd training trip (sorted ids), offset trips excluded.
2. The chosen decoder is applied to the student_test trips, using the final
   classifier's probabilities, and compared with classifier-only top-1 on:
   Top-1 accuracy, candidate-level macro F1 of the decisions, route continuity,
   route precision/recall/F1 and inference time.

Outputs: results/hmm/{decoder_grid.csv, hmm_config.json, comparison.csv, per_trip.csv}
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.common.config import STUDENT_TEST, TRAIN, get_output_dir, get_results_dir
from src.evaluation.metrics import observation_accuracy, route_continuity, route_overlap, select_top1
from src.evaluation.route_eval import trip_contexts
from src.hmm.decoder import HMMConfig, decode_trip
from src.models.dataset import build_partition, offset_trips

GRID = {"emission_weight": [0.25, 0.5, 1.0, 2.0], "transition_beta_m": [2.5, 5.0, 10.0, 20.0, 40.0]}
CAND_COLS = ["trip_id", "obs_idx", "arc_id", "offset_frac", "distance_rank", "is_reference"]


def decision_macro_f1(feats: pd.DataFrame, chosen: pd.DataFrame) -> float:
    """Candidate-level macro F1 when the chosen candidate of each observation is labelled 1, the rest 0."""
    picked = feats[["trip_id", "obs_idx", "arc_id"]].merge(chosen[["trip_id", "obs_idx", "arc_id"]].assign(_p=1),
                                                          on=["trip_id", "obs_idx", "arc_id"], how="left")
    return float(f1_score(feats["is_reference"].astype(int), picked["_p"].fillna(0).astype(int), average="macro"))


def run_methods(feats: pd.DataFrame, obs: pd.DataFrame, contexts, config: HMMConfig,
                with_routes: bool = True) -> Dict[str, object]:
    """Decode all trips; return chosen tables + per-trip metrics for classifier-only and HMM."""
    chosen_clf, chosen_hmm, per_trip = [], [], []
    by_trip = {t: g for t, g in feats.groupby("trip_id", sort=False)}
    for pre, graph in contexts:
        tid = str(pre.trip_id)
        cands = by_trip.get(tid)
        if cands is None:
            continue
        clf = select_top1(cands.rename(columns={"prob": "score"}))
        start = time.perf_counter()
        hmm, stats = decode_trip(cands, pre.track, graph, config)
        hmm_s = time.perf_counter() - start
        hmm = hmm.merge(cands[["obs_idx", "arc_id", "is_reference", "distance_rank"]], on=["obs_idx", "arc_id"],
                        how="left").drop_duplicates("obs_idx")
        hmm.insert(0, "trip_id", tid)
        chosen_clf.append(clf)
        chosen_hmm.append(hmm)
        if with_routes:
            o = obs[obs["trip_id"] == tid]
            for method, ch in (("classifier_only", clf), ("classifier_hmm", hmm)):
                rec = {"trip_id": tid, "method": method, **observation_accuracy(ch, o),
                       **route_continuity(ch, pre.track, graph),
                       **route_overlap(ch.sort_values("obs_idx")["arc_id"].to_numpy(), pre.route, graph)}
                rec["decode_time_s"] = hmm_s if method == "classifier_hmm" else 0.0
                per_trip.append(rec)
    return {"clf": pd.concat(chosen_clf, ignore_index=True), "hmm": pd.concat(chosen_hmm, ignore_index=True),
            "per_trip": pd.DataFrame(per_trip)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="hgb")
    parser.add_argument("--probs", default="tuned", choices=["tuned", "untuned"])
    args = parser.parse_args(argv)
    out = get_results_dir("hmm")
    model_dir = get_output_dir("models")
    prefix = f"tuned_{args.model}" if args.probs == "tuned" else args.model

    train_feats, train_obs = build_partition(TRAIN)
    test_feats, test_obs = build_partition(STUDENT_TEST)
    train_feats = train_feats[CAND_COLS].assign(prob=np.load(model_dir / f"oof_{prefix}.npy"))
    test_feats = test_feats[CAND_COLS].assign(prob=np.load(model_dir / f"student_test_{prefix}.npy"))

    # ---- 1. decoder tuning on training trips (out-of-fold probabilities) ----
    skip = set(offset_trips(train_obs, 30.0))
    tune_trips = [t for i, t in enumerate(sorted(train_obs["trip_id"].unique())) if i % 3 == 0 and t not in skip]
    tune_ctx = list(trip_contexts(TRAIN, tune_trips))
    tune_feats = train_feats[train_feats["trip_id"].isin(tune_trips)]
    tune_obs = train_obs[train_obs["trip_id"].isin(tune_trips)]
    base = observation_accuracy(select_top1(tune_feats.rename(columns={"prob": "score"})), tune_obs)["top1_accuracy"]
    grid_rows = []
    for w, beta in itertools.product(GRID["emission_weight"], GRID["transition_beta_m"]):
        cfg = HMMConfig(emission_weight=w, transition_beta_m=beta)
        start = time.perf_counter()
        res = run_methods(tune_feats, tune_obs, tune_ctx, cfg, with_routes=False)
        acc = observation_accuracy(res["hmm"], tune_obs)["top1_accuracy"]
        grid_rows.append({"emission_weight": w, "transition_beta_m": beta, "train_oof_top1_hmm": acc,
                          "train_oof_top1_classifier_only": base, "time_s": time.perf_counter() - start})
        print(f"  w={w} beta={beta}: HMM top1={acc:.4f} (classifier-only {base:.4f})", flush=True)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(out / "decoder_grid.csv", index=False)
    best = grid.sort_values("train_oof_top1_hmm", ascending=False).iloc[0]
    config = HMMConfig(emission_weight=float(best["emission_weight"]), transition_beta_m=float(best["transition_beta_m"]))
    (out / "hmm_config.json").write_text(json.dumps({**config.as_dict(), "selected_on": "train OOF probabilities",
                                                     "tuning_trips": tune_trips, "classifier": prefix}, indent=2))

    # ---- 2. student_test: classifier-only vs classifier + HMM ----
    test_ctx = list(trip_contexts(STUDENT_TEST))
    res = run_methods(test_feats, test_obs, test_ctx, config)
    per_trip = res["per_trip"]
    per_trip.to_csv(out / "per_trip.csv", index=False)
    rows = []
    for method, chosen in (("classifier_only", res["clf"]), ("classifier_hmm", res["hmm"])):
        g = per_trip[per_trip["method"] == method]
        trans = g["n_transitions"].sum()
        rows.append({"method": method, **{k: v for k, v in observation_accuracy(chosen, test_obs).items()},
                     "decision_macro_f1": decision_macro_f1(test_feats, chosen),
                     "route_continuity": (trans - g["n_breaks"].sum()) / trans, "n_breaks": int(g["n_breaks"].sum()),
                     "route_precision_mean": g["route_precision"].mean(), "route_recall_mean": g["route_recall"].mean(),
                     "route_f1_mean": g["route_f1"].mean(), "decode_time_s_total": g["decode_time_s"].sum(),
                     "trips_improved_top1": None})
    comp = pd.DataFrame(rows)
    pv = per_trip.pivot(index="trip_id", columns="method", values="top1_accuracy")
    comp.loc[comp["method"] == "classifier_hmm", "trips_improved_top1"] = int((pv["classifier_hmm"] > pv["classifier_only"]).sum())
    comp.to_csv(out / "comparison.csv", index=False)
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
