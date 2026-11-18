"""Export the final inference bundle to artifacts/ from recorded experiment outputs.

    python scripts/build_final_artifacts.py

Reads (all produced by earlier pipeline steps, nothing typed by hand):
    results/models/selected_model.json         which model was selected
    results/optimisation/best_params.json      tuned hyperparameters
    results/optimisation/tuned_vs_untuned.csv  tuned model metrics
    results/hmm/hmm_config.json, comparison.csv decoder configuration + results
    outputs/models/tuned_<model>.joblib         the tuned, refitted model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import get_output_dir, get_results_dir  # noqa: E402
from src.hmm.decoder import HMMConfig  # noqa: E402
from src.inference.artifacts import export_bundle  # noqa: E402


def main() -> None:
    res = get_results_dir()
    selected = json.loads((res / "models" / "selected_model.json").read_text())["selected"]
    best = json.loads((res / "optimisation" / "best_params.json").read_text())
    if best["model"] != selected:
        raise SystemExit(f"Optimised model ({best['model']}) differs from the selected model ({selected})")
    hmm_meta = json.loads((res / "hmm" / "hmm_config.json").read_text())
    hmm_cfg = HMMConfig(**{k: hmm_meta[k] for k in HMMConfig().as_dict()})
    tuned = pd.read_csv(res / "optimisation" / "tuned_vs_untuned.csv").set_index("variant").loc["tuned"]
    hmm = pd.read_csv(res / "hmm" / "comparison.csv").set_index("method")
    dataset = json.loads((res / "models" / "dataset_info.json").read_text())
    metrics = {"student_test_classifier_top1": float(tuned["test_top1_accuracy"]),
               "student_test_classifier_macro_f1": float(tuned["test_macro_f1"]),
               "student_test_classifier_log_loss": float(tuned["test_log_loss"]),
               "cv_top1_mean": float(tuned["cv_top1_mean"]),
               "student_test_hmm_top1": float(hmm.loc["classifier_hmm", "top1_accuracy"]),
               "student_test_hmm_route_continuity": float(hmm.loc["classifier_hmm", "route_continuity"])}
    training = {"partitions": ["train"], "excluded_trips": dataset["excluded_offset_trips"],
                "observation_stride": dataset["observation_stride"], "n_fit_rows": dataset["n_rows_fit"]}
    out = export_bundle(get_output_dir("models") / f"tuned_{selected}.joblib", selected, best["best_params"],
                        hmm_cfg, metrics, training=training)
    print(f"Artefacts written to {out}")


if __name__ == "__main__":
    main()
