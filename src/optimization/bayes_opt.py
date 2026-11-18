"""Bayesian hyperparameter optimisation of the selected model (Optuna, TPE sampler).

    python -m src.optimization.bayes_opt --model hgb --trials 40

Objective (maximised): mean observation-level Top-1 accuracy over the same five
trip-grouped folds used for model comparison (GroupKFold on the 80 training
trips). Top-1 is what the map-matcher actually outputs per GPS fix and what the
HMM builds on; candidate-level macro F1 depends on a 0.5 threshold that the
system never uses. Macro F1 and log loss of every trial are recorded as well.

Cost control: trials fit on every 6th observation of the training trips
(all candidates of a kept observation) instead of every 3rd. They are still
evaluated on ALL candidate rows of the held-out trips. The best configuration
is then re-run with the standard setting, so tuned and untuned models are
compared under identical conditions.

The evaluator-testing partition is never touched. Student-test data is used
only AFTER the search, to report tuned vs untuned performance.

Outputs: results/optimisation/ (study.db, trials.csv, best_params.json,
search_space.json, tuned_vs_untuned.csv) and outputs/models/tuned_<model>.joblib.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable, Dict

import joblib
import numpy as np
import optuna
import pandas as pd

from src.common.config import RANDOM_SEED, STUDENT_TEST, TRAIN, get_output_dir, get_results_dir
from src.models.cv import cross_validate, evaluate_scores, score_candidates, trip_folds
from src.models.dataset import TrainingSubsetConfig, build_partition, training_subset, xy
from src.models.model_zoo import DEFAULT_PARAMS, make_model

# Search spaces: (description for the report, sampler function).
SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    "hgb": {
        "learning_rate": "log-uniform [0.02, 0.3]",
        "max_iter": "int [100, 800] (early stopping still active)",
        "max_leaf_nodes": "int [15, 127]",
        "min_samples_leaf": "int log [10, 400]",
        "l2_regularization": "log-uniform [1e-4, 10]",
        "max_features": "uniform [0.4, 1.0] (feature subsampling per split)",
        "class_weight": "{None, 'balanced'}",
    },
    "rf": {
        "n_estimators": "int [100, 300]",
        "max_depth": "{None} or int [8, 32]",
        "min_samples_leaf": "int log [1, 50]",
        "max_features": "uniform [0.2, 0.8]",
        "class_weight": "{None, 'balanced_subsample'}",
    },
    "mlp": {
        "hidden_layer_sizes": "{(32,), (64,32), (128,64), (128,64,32)}",
        "alpha": "log-uniform [1e-6, 1e-2]",
        "learning_rate_init": "log-uniform [1e-4, 1e-2]",
        "batch_size": "{256, 512, 1024}",
    },
}


def sample_params(key: str, trial: optuna.Trial) -> Dict[str, Any]:
    if key == "hgb":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 800),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 400, log=True),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
                "max_features": trial.suggest_float("max_features", 0.4, 1.0),
                "class_weight": trial.suggest_categorical("class_weight", ["none", "balanced"])}
    if key == "rf":
        depth = trial.suggest_categorical("max_depth_mode", ["none", "limited"])
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": None if depth == "none" else trial.suggest_int("max_depth", 8, 32),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
                "max_features": trial.suggest_float("max_features", 0.2, 0.8),
                "class_weight": trial.suggest_categorical("class_weight", ["none", "balanced_subsample"])}
    if key == "mlp":
        sizes = {"32": (32,), "64-32": (64, 32), "128-64": (128, 64), "128-64-32": (128, 64, 32)}
        return {"hidden_layer_sizes": sizes[trial.suggest_categorical("hidden", list(sizes))],
                "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024])}
    raise KeyError(f"No search space defined for '{key}'")


def clean_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    if out.get("class_weight") == "none":
        out["class_weight"] = None
    return out


def make_objective(key: str, fit_rows, all_rows, obs, folds) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        params = clean_params(sample_params(key, trial))
        start = time.perf_counter()
        fold_df, _ = cross_validate(key, params, fit_rows, all_rows, obs, folds, verbose=False)
        trial.set_user_attr("cv_macro_f1", float(fold_df["macro_f1"].mean()))
        trial.set_user_attr("cv_log_loss", float(fold_df["log_loss"].mean()))
        trial.set_user_attr("cv_top1_std", float(fold_df["top1_accuracy"].std()))
        trial.set_user_attr("params", json.dumps(params, default=str))
        trial.set_user_attr("duration_s", time.perf_counter() - start)
        value = float(fold_df["top1_accuracy"].mean())
        print(f"  trial {trial.number}: top1={value:.4f} ({time.perf_counter() - start:.0f}s) {params}", flush=True)
        return value
    return objective


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="hgb", choices=sorted(SEARCH_SPACES))
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--search-stride", type=int, default=6)
    parser.add_argument("--finish-only", action="store_true",
                        help="do not search: load the existing study.db (e.g. a search stopped early) and run the "
                             "tuned-vs-untuned comparison with its best trial")
    args = parser.parse_args(argv)

    out = get_results_dir("optimisation")
    model_dir = get_output_dir("models")
    train_feats, train_obs = build_partition(TRAIN)
    test_feats, test_obs = build_partition(STUDENT_TEST)
    folds = trip_folds(train_obs["trip_id"].unique().tolist())
    search_rows, _ = training_subset(train_feats, train_obs, TrainingSubsetConfig(observation_stride=args.search_stride))
    std_rows, _ = training_subset(train_feats, train_obs, TrainingSubsetConfig())

    (out / "search_space.json").write_text(json.dumps({
        "model": args.model, "space": SEARCH_SPACES[args.model], "objective": "maximise mean 5-fold GroupKFold Top-1",
        "sampler": "optuna.samplers.TPESampler(seed=42, n_startup_trials=10)", "n_trials": args.trials,
        "search_fit_stride": args.search_stride}, indent=2))
    db = out / "study.db"
    if args.finish_only:
        study = optuna.load_study(study_name=f"pathsnap_{args.model}", storage=f"sqlite:///{db}")
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        search_s = sum(t.user_attrs.get("duration_s", 0.0) for t in done)
    else:
        if db.exists():
            db.unlink()
        study = optuna.create_study(study_name=f"pathsnap_{args.model}", direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=10),
                                    storage=f"sqlite:///{db}")
        start = time.perf_counter()
        study.optimize(make_objective(args.model, search_rows, train_feats, train_obs, folds), n_trials=args.trials)
        search_s = time.perf_counter() - start
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration", "state"))
    trials = trials[trials["state"] == "COMPLETE"]
    trials.to_csv(out / "trials.csv", index=False)
    best = clean_params(sample_params(args.model, optuna.trial.FixedTrial(study.best_params)))
    (out / "best_params.json").write_text(json.dumps({
        "model": args.model, "best_trial": study.best_trial.number, "best_cv_top1_search": study.best_value,
        "best_params": best, "search_time_s": search_s,
        "n_complete_trials": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
        "stopped_early": bool(args.finish_only)}, indent=2, default=str))

    # Tuned vs untuned under identical conditions (standard stride, same folds, same test set).
    rows = []
    for label, params in (("untuned", None), ("tuned", best)):
        fold_df, oof = cross_validate(args.model, params, std_rows, train_feats, train_obs, folds, verbose=False)
        model = make_model(args.model, params)
        X, y = xy(std_rows)
        t0 = time.perf_counter()
        model.fit(X, y)
        fit_s = time.perf_counter() - t0
        proba, pred_s = score_candidates(model, test_feats)
        test = evaluate_scores(test_feats, test_obs, proba)
        if label == "tuned":
            joblib.dump(model, model_dir / f"tuned_{args.model}.joblib", compress=3)
            np.save(model_dir / f"oof_tuned_{args.model}.npy", oof)
            np.save(model_dir / f"student_test_tuned_{args.model}.npy", proba)
        rows.append({"variant": label, "params": json.dumps(params or DEFAULT_PARAMS[args.model], default=str),
                     "cv_top1_mean": fold_df["top1_accuracy"].mean(), "cv_top1_std": fold_df["top1_accuracy"].std(),
                     "cv_macro_f1_mean": fold_df["macro_f1"].mean(), "cv_log_loss_mean": fold_df["log_loss"].mean(),
                     "test_top1_accuracy": test["top1_accuracy"], "test_macro_f1": test["macro_f1"],
                     "test_log_loss": test["log_loss"], "fit_time_s": fit_s, "test_inference_time_s": pred_s})
        print(f"{label}: {rows[-1]}", flush=True)
    pd.DataFrame(rows).to_csv(out / "tuned_vs_untuned.csv", index=False)


if __name__ == "__main__":
    main()
