"""Select the model to tune, by a rule fixed BEFORE looking at the results.

    python -m src.models.select_model

Rule (cross-validation on training trips only; student_test is reported, not used):
1. highest mean CV Top-1 accuracy (the task metric);
2. models within 0.5 percentage points of the best are treated as tied, and the
   tie is broken by lower mean CV log loss (better probabilities matter because
   the HMM uses them as emissions), then by lower inference time.
Writes results/models/selected_model.json with the evidence.
"""
from __future__ import annotations

import json

import pandas as pd

from src.common.config import get_results_dir

TIE_MARGIN = 0.005


def select(comparison: pd.DataFrame) -> dict:
    best = comparison["cv_top1_mean"].max()
    tied = comparison[comparison["cv_top1_mean"] >= best - TIE_MARGIN]
    chosen = tied.sort_values(["cv_log_loss_mean", "test_inference_time_s"]).iloc[0]
    cols = ["model", "cv_top1_mean", "cv_top1_std", "cv_macro_f1_mean", "cv_log_loss_mean", "test_top1_accuracy",
            "test_inference_time_s"]
    return {"selected": chosen["model"], "rule": __doc__.split("Rule")[1].strip(),
            "tied_within_margin": tied["model"].tolist(),
            "candidates": comparison[cols].to_dict(orient="records")}


def main() -> None:
    res = get_results_dir("models")
    comparison = pd.read_csv(res / "model_comparison.csv")
    decision = select(comparison)
    (res / "selected_model.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
