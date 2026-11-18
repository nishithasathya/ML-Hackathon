"""Train, cross-validate and compare the five candidate-scoring models.

Run from the repository root:

    python -m src.models.train_compare                 # all five models
    python -m src.models.train_compare --models hgb rf # a subset

What it does (all numbers written to results/models/, nothing typed by hand):
1. builds the train / student_test candidate tables (cached in data/processed/);
2. 5-fold GroupKFold over the 80 training trips (folds split by trip id);
3. refits each model on the training subset of all 80 trips;
4. scores every student_test candidate; Top-1, macro F1, log loss, confusion matrix;
5. route continuity + route precision/recall/F1 of the top-1 predictions (student_test);
6. saves model files, out-of-fold and student-test probabilities, histories, timings.
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.common.config import RANDOM_SEED, STUDENT_TEST, TRAIN, get_output_dir, get_results_dir
from src.evaluation.metrics import select_top1
from src.evaluation.route_eval import evaluate_routes, summarise_routes, trip_contexts
from src.models.cv import cross_validate, evaluate_scores, score_candidates, trip_folds
from src.models.dataset import TrainingSubsetConfig, build_partition, training_subset, xy
from src.models.model_zoo import DEFAULT_PARAMS, MODEL_INFO, OBJECTIVES, make_model, training_history


def top1_table(feats: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    scored = feats[["trip_id", "obs_idx", "arc_id", "offset_frac", "distance_rank", "is_reference"]].assign(score=proba)
    return select_top1(scored)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODEL_INFO))
    parser.add_argument("--stride", type=int, default=TrainingSubsetConfig.observation_stride)
    args = parser.parse_args(argv)

    np.random.seed(RANDOM_SEED)
    res_dir = get_results_dir("models")
    model_dir = get_output_dir("models")

    train_feats, train_obs = build_partition(TRAIN)
    test_feats, test_obs = build_partition(STUDENT_TEST)
    subset_cfg = TrainingSubsetConfig(observation_stride=args.stride)
    fit_rows, subset_info = training_subset(train_feats, train_obs, subset_cfg)
    folds = trip_folds(train_obs["trip_id"].unique().tolist())
    (res_dir / "dataset_info.json").write_text(json.dumps({
        **subset_info, "n_train_trips": int(train_obs["trip_id"].nunique()),
        "n_student_test_trips": int(test_obs["trip_id"].nunique()),
        "n_student_test_rows": int(len(test_feats)), "folds": [v for _, v in folds],
        "seed": RANDOM_SEED, "python": platform.python_version(), "sklearn": sklearn.__version__}, indent=2))
    print(f"fit rows: {len(fit_rows)} (positive rate {subset_info['positive_rate_fit']:.3f}); "
          f"excluded trips: {subset_info['excluded_offset_trips']}", flush=True)

    contexts = list(trip_contexts(STUDENT_TEST))
    rows, fold_tables, top1_sets = [], [], {}
    for key in args.models:
        print(f"== {key}: {MODEL_INFO[key]['name']}", flush=True)
        fold_df, oof = cross_validate(key, None, fit_rows, train_feats, train_obs, folds)
        fold_tables.append(fold_df)
        np.save(model_dir / f"oof_{key}.npy", oof)

        model = make_model(key)
        X, y = xy(fit_rows)
        start = time.perf_counter()
        model.fit(X, y)
        fit_s = time.perf_counter() - start
        joblib.dump(model, model_dir / f"{key}.joblib", compress=3)
        hist = training_history(model)
        if hist:
            (res_dir / f"history_{key}.json").write_text(json.dumps(hist))

        proba, pred_s = score_candidates(model, test_feats)
        np.save(model_dir / f"student_test_{key}.npy", proba)
        test_metrics = evaluate_scores(test_feats, test_obs, proba)
        top1_sets[key] = top1_table(test_feats, proba)
        X_tr_all, y_tr_all = xy(fit_rows)
        train_proba = model.predict_proba(X_tr_all)[:, 1]
        train_metrics = evaluate_scores(fit_rows, train_obs.merge(fit_rows[["trip_id", "obs_idx"]].drop_duplicates()),
                                        train_proba)

        cv = fold_df
        rows.append({
            "model": key, "name": MODEL_INFO[key]["name"], "family": MODEL_INFO[key]["family"],
            "params": json.dumps(DEFAULT_PARAMS[key], default=str), "objective": OBJECTIVES[key],
            "cv_macro_f1_mean": cv["macro_f1"].mean(), "cv_macro_f1_std": cv["macro_f1"].std(),
            "cv_top1_mean": cv["top1_accuracy"].mean(), "cv_top1_std": cv["top1_accuracy"].std(),
            "cv_log_loss_mean": cv["log_loss"].mean(),
            "train_fit_top1": train_metrics["top1_accuracy"], "train_fit_macro_f1": train_metrics["macro_f1"],
            "train_fit_log_loss": train_metrics["log_loss"],
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "fit_time_s": fit_s, "test_inference_time_s": pred_s,
            "inference_ms_per_1k_candidates": 1000 * pred_s / len(test_feats) * 1000,
        })
        print(f"   student_test: top1={test_metrics['top1_accuracy']:.4f} macroF1={test_metrics['macro_f1']:.4f} "
              f"logloss={test_metrics['log_loss']:.4f}", flush=True)

    # Non-learned reference point (NOT one of the five models): always pick the nearest arc.
    nearest = top1_table(test_feats, -test_feats["distance_m"].to_numpy())
    top1_sets["nearest_arc_baseline"] = nearest

    routes = evaluate_routes({k: v for k, v in top1_sets.items()}, contexts)
    routes.to_csv(res_dir / "student_test_route_metrics_per_trip.csv", index=False)
    route_summary = summarise_routes(routes)
    route_summary.to_csv(res_dir / "student_test_route_metrics.csv", index=False)

    comparison = pd.DataFrame(rows).merge(route_summary.rename(columns={"method": "model"}), on="model", how="left")
    comparison.to_csv(res_dir / "model_comparison.csv", index=False)
    pd.concat(fold_tables).to_csv(res_dir / "cv_folds.csv", index=False)
    nearest_acc = float(nearest.merge(test_obs[["trip_id", "obs_idx"]], how="right")["is_reference"].fillna(False).mean())
    (res_dir / "baseline_nearest_arc.json").write_text(json.dumps({"student_test_top1_accuracy": nearest_acc}, indent=2))
    print(comparison[["model", "cv_macro_f1_mean", "cv_top1_mean", "test_top1_accuracy", "test_macro_f1",
                      "test_log_loss", "route_continuity", "fit_time_s", "test_inference_time_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
