"""Phase-2 experiment figures, generated ONLY from recorded result files.

    python -m src.visualization.experiment_plots

Reads results/{models,optimisation,hmm,candidates}/*.csv|json and writes PNGs to
results/figures/. Loss curves are drawn only for models whose training
genuinely recorded a history (HistGradientBoosting, MLP); nothing is simulated.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.config import get_results_dir  # noqa: E402

BLUE, ORANGE, GREY, GREEN = "#2a6fdb", "#e8743b", "#8a8f98", "#2e9d5b"


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _bar_labels(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")


def model_plots(res: Path, out: Path) -> None:
    comp = pd.read_csv(res / "models" / "model_comparison.csv")
    names = comp["name"].tolist()
    x = np.arange(len(comp))

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bars = ax.bar(x, comp["cv_macro_f1_mean"], yerr=comp["cv_macro_f1_std"], color=BLUE, capsize=4)
    _bar_labels(ax, bars)
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylim(0.7, 1.0)
    _style(ax, "Cross-validation macro F1 (5-fold GroupKFold by trip, mean ± std)", ylabel="macro F1")
    _save(fig, out / "cv_macro_f1.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    w = 0.38
    b1 = ax.bar(x - w / 2, comp["test_top1_accuracy"], w, label="Top-1 accuracy", color=BLUE)
    b2 = ax.bar(x + w / 2, comp["test_macro_f1"], w, label="Macro F1 (threshold 0.5)", color=ORANGE)
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    base = json.loads((res / "models" / "baseline_nearest_arc.json").read_text())["student_test_top1_accuracy"]
    ax.axhline(base, color=GREY, linestyle="--", linewidth=1, label=f"nearest-arc heuristic Top-1 = {base:.3f}")
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylim(0.4, 1.08)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    _style(ax, "Student-test performance (10 trips, all candidates)")
    _save(fig, out / "student_test_top1_macro_f1.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    w = 0.38
    b1 = ax.bar(x - w / 2, comp["cv_log_loss_mean"], w, label="CV (train trips)", color=BLUE)
    b2 = ax.bar(x + w / 2, comp["test_log_loss"], w, label="Student test", color=ORANGE)
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    ax.set_xticks(x, names, rotation=15)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Log loss (lower is better)", ylabel="log loss")
    _save(fig, out / "log_loss.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, col, title in ((axes[0], "fit_time_s", "Training time (final fit)"),
                           (axes[1], "inference_ms_per_1k_candidates", "Inference time per 1,000 candidates")):
        bars = ax.bar(x, comp[col], color=BLUE)
        _bar_labels(ax, bars, "{:.2f}")
        ax.set_yscale("log")
        ax.set_xticks(x, [n.split(" (")[0] for n in names], rotation=25, fontsize=8)
        _style(ax, title, ylabel="seconds" if col == "fit_time_s" else "milliseconds (log scale)")
    _save(fig, out / "timings.png")

    fig, axes = plt.subplots(1, len(comp), figsize=(2.6 * len(comp), 2.8))
    for ax, (_, r) in zip(axes, comp.iterrows()):
        cm = np.array([[r["test_tn"], r["test_fp"]], [r["test_fn"], r["test_tp"]]])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j]):,}\n({norm[i, j]:.2f})", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=7)
        ax.set_yticks([0, 1], ["true 0", "true 1"], fontsize=7)
        ax.set_title(r["name"].split(" (")[0], fontsize=8)
    fig.suptitle("Candidate-level confusion matrices, student test (threshold 0.5)", fontsize=10, x=0.01, ha="left")
    _save(fig, out / "confusion_matrices.png")

    for key in ("hgb", "mlp"):
        path = res / "models" / f"history_{key}.json"
        if not path.exists():
            continue
        hist = json.loads(path.read_text())
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(hist["train_loss"], color=BLUE, label="training loss")
        if "validation_loss" in hist:
            ax.plot(hist["validation_loss"], color=ORANGE, label="validation loss (internal 10% split)")
        if "validation_accuracy" in hist:
            ax2 = ax.twinx()
            ax2.plot(hist["validation_accuracy"], color=ORANGE, label="validation accuracy (internal 10% split)")
            ax2.set_ylabel("accuracy")
            ax2.legend(frameon=False, fontsize=8, loc="center right")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        label = "boosting iteration" if key == "hgb" else "epoch"
        _style(ax, f"{'HistGradientBoosting' if key == 'hgb' else 'MLP'}: recorded training history", label, "log loss")
        _save(fig, out / f"training_history_{key}.png")


def optimisation_plots(res: Path, out: Path) -> None:
    trials = pd.read_csv(res / "optimisation" / "trials.csv")
    trials = trials[trials["state"] == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.scatter(trials["number"], trials["value"], color=BLUE, s=18, label="trial objective")
    ax.plot(trials["number"], trials["value"].cummax(), color=ORANGE, label="best so far")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Bayesian optimisation (Optuna TPE): CV Top-1 per trial", "trial", "mean CV Top-1")
    _save(fig, out / "optuna_history.png")

    tvu = pd.read_csv(res / "optimisation" / "tuned_vs_untuned.csv")
    metrics = [("cv_top1_mean", "CV Top-1"), ("test_top1_accuracy", "Test Top-1"), ("cv_macro_f1_mean", "CV macro F1"),
               ("test_macro_f1", "Test macro F1")]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(metrics))
    w = 0.38
    for k, (variant, colour) in enumerate((("untuned", GREY), ("tuned", BLUE))):
        row = tvu[tvu["variant"] == variant].iloc[0]
        bars = ax.bar(x + (k - 0.5) * w, [row[m] for m, _ in metrics], w, label=variant, color=colour)
        _bar_labels(ax, bars)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0.6, 1.0)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Tuned vs untuned (same folds, same student-test set)")
    _save(fig, out / "tuned_vs_untuned.png")


def hmm_plots(res: Path, out: Path) -> None:
    comp = pd.read_csv(res / "hmm" / "comparison.csv").set_index("method")
    metrics = [("top1_accuracy", "Top-1 accuracy"), ("decision_macro_f1", "Decision macro F1"),
               ("route_continuity", "Route continuity"), ("route_f1_mean", "Route F1 (mean/trip)")]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(metrics))
    w = 0.38
    for k, (method, colour, label) in enumerate((("classifier_only", GREY, "classifier only"),
                                                 ("classifier_hmm", BLUE, "classifier + HMM/Viterbi"))):
        bars = ax.bar(x + (k - 0.5) * w, [comp.loc[method, m] for m, _ in metrics], w, label=label, color=colour)
        _bar_labels(ax, bars)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0.6, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, "Student test: classifier-only vs classifier + HMM")
    _save(fig, out / "hmm_vs_classifier.png")

    per = pd.read_csv(res / "hmm" / "per_trip.csv", dtype={"trip_id": str})
    pv = per.pivot(index="trip_id", columns="method", values="top1_accuracy")
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(pv))
    ax.bar(x - 0.2, pv["classifier_only"], 0.4, color=GREY, label="classifier only")
    ax.bar(x + 0.2, pv["classifier_hmm"], 0.4, color=BLUE, label="classifier + HMM")
    ax.set_xticks(x, pv.index, rotation=30, fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, "Top-1 accuracy per student-test trip", ylabel="Top-1")
    _save(fig, out / "hmm_per_trip.png")

    grid = pd.read_csv(res / "hmm" / "decoder_grid.csv")
    pt = grid.pivot(index="emission_weight", columns="transition_beta_m", values="train_oof_top1_hmm")
    fig, ax = plt.subplots(figsize=(5.5, 3))
    im = ax.imshow(pt.values, cmap="Blues", aspect="auto")
    for i in range(pt.shape[0]):
        for j in range(pt.shape[1]):
            ax.text(j, i, f"{pt.values[i, j]:.4f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(pt.shape[1]), [f"{c:g}" for c in pt.columns])
    ax.set_yticks(range(pt.shape[0]), [f"{c:g}" for c in pt.index])
    ax.set_xlabel("transition beta (m)")
    ax.set_ylabel("emission weight")
    ax.set_title(f"Decoder tuning: Top-1 on training trips (OOF scores)\nclassifier-only = "
                 f"{grid['train_oof_top1_classifier_only'].iloc[0]:.4f}", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, out / "hmm_decoder_grid.png")


def radius_plot(res: Path, out: Path) -> None:
    path = res / "candidates" / "radius_study.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(df["radius_m"], df["cv_top1_mean"], marker="o", color=BLUE, label="CV Top-1 (train, decides)")
    ax.plot(df["radius_m"], df["student_test_top1"], marker="s", color=ORANGE, label="student-test Top-1 (reported)")
    ax.plot(df["radius_m"], 1 - df["train_candidate_miss_rate"], marker="^", color=GREY, linestyle="--",
            label="train candidate coverage (upper bound)")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Candidate radius re-evaluated with a trained model", "radius (m)", "share of observations")
    _save(fig, out / "radius_study.png")


def main() -> None:
    res = get_results_dir()
    out = get_results_dir("figures")
    model_plots(res, out)
    if (res / "optimisation" / "trials.csv").exists():
        optimisation_plots(res, out)
    if (res / "hmm" / "comparison.csv").exists():
        hmm_plots(res, out)
    radius_plot(res, out)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
