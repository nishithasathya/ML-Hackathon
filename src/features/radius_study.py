"""Re-evaluate the candidate search radius WITH a trained model (phase 2).

    python -m src.features.radius_study --model hgb --radii 25 50 75 100

Phase 1 chose 50 m from the candidate miss rate alone. A larger radius lowers the
miss rate but adds more wrong candidates, so the useful question is: which radius
gives the best Top-1 after scoring? For every radius the full train/student_test
candidate tables are rebuilt, the model (default hyperparameters) is
cross-validated with the same trip-grouped folds, and the decision uses the
TRAINING CV Top-1 only. Student-test numbers are reported for information.

Decision: radii whose CV Top-1 is within 0.5 percentage points of the best are
treated as tied (the same margin as the model-selection rule) and the SMALLEST
tied radius is kept: fewer candidates means fewer distractors and a cheaper HMM
(transition cost grows with candidates squared). Note: this tie rule was added
after the first run of the study, mirroring the model-selection rule; the
report states this explicitly.

Output: results/candidates/radius_study.csv  (``python -m src.features.radius_study --decide-only``
re-applies the decision to an existing CSV without recomputing)
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from src.common.config import STUDENT_TEST, TRAIN, get_results_dir
from src.features.candidates import CandidateConfig


TIE_MARGIN = 0.005


def decide(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in ("selected", "best_cv", "tied_with_best") if c in df.columns])
    best = df["cv_top1_mean"].max()
    df["best_cv"] = df["cv_top1_mean"] == best
    df["tied_with_best"] = df["cv_top1_mean"] >= best - TIE_MARGIN
    df["selected"] = df["radius_m"] == df.loc[df["tied_with_best"], "radius_m"].min()
    return df


def main(argv=None) -> None:
    # Imported here: src.models depends on src.features, not the other way round at import time.
    from src.models.cv import cross_validate, evaluate_scores, score_candidates, trip_folds
    from src.models.dataset import build_partition, training_subset, xy
    from src.models.model_zoo import make_model

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="hgb")
    parser.add_argument("--radii", type=float, nargs="+", default=[25, 50, 75, 100])
    parser.add_argument("--decide-only", action="store_true")
    args = parser.parse_args(argv)
    path = get_results_dir("candidates") / "radius_study.csv"
    if args.decide_only:
        out = decide(pd.read_csv(path))
        out.to_csv(path, index=False)
        print(out.to_string(index=False))
        return
    rows = []
    for r in args.radii:
        cfg = CandidateConfig(radius_m=r)
        start = time.perf_counter()
        tr_f, tr_o = build_partition(TRAIN, cfg, verbose=False)
        te_f, te_o = build_partition(STUDENT_TEST, cfg, verbose=False)
        build_s = time.perf_counter() - start
        fit_rows, _ = training_subset(tr_f, tr_o)
        folds = trip_folds(tr_o["trip_id"].unique().tolist())
        cv, _ = cross_validate(args.model, None, fit_rows, tr_f, tr_o, folds, verbose=False)
        model = make_model(args.model)
        model.fit(*xy(fit_rows))
        proba, pred_s = score_candidates(model, te_f)
        test = evaluate_scores(te_f, te_o, proba)
        rows.append({"radius_m": r, "train_candidate_miss_rate": 1 - tr_o["has_reference_candidate"].mean(),
                     "train_candidates_per_obs": len(tr_f) / len(tr_o), "cv_top1_mean": cv["top1_accuracy"].mean(),
                     "cv_top1_std": cv["top1_accuracy"].std(), "cv_macro_f1_mean": cv["macro_f1"].mean(),
                     "student_test_candidate_miss_rate": test["candidate_miss_rate"],
                     "student_test_top1": test["top1_accuracy"], "feature_build_time_s": build_s,
                     "student_test_inference_s": pred_s})
        print(rows[-1], flush=True)
    out = decide(pd.DataFrame(rows))
    out.to_csv(path, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
