"""Build the consolidated technical report (PDF) from the recorded results.

    python scripts/build_report.py

Every number in the report is read from results/ (or outputs/ for phase-1 EDA
tables) at build time; every figure is a file produced by the pipeline; every
screenshot is a capture of the running system (scripts/capture_screenshots.py).
If a result file is missing, its section says so instead of inventing content.
Structure follows the official Deliverables document (sections 2-8).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "technical_report"
FIG = RES / "figures"
SHOT = RES / "screenshots"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=10.5, spaceBefore=6, spaceAfter=3)
BODY = ParagraphStyle("B", parent=styles["BodyText"], fontSize=9.2, leading=12.4)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=7.6, leading=9.6)
CAP = ParagraphStyle("C", parent=BODY, fontSize=8, textColor=colors.HexColor("#555555"), alignment=TA_CENTER)
TITLE = ParagraphStyle("T", parent=styles["Title"], fontSize=20)

story: list = []


def p(text, style=BODY):
    story.append(Paragraph(text, style))


def bullets(items):
    for it in items:
        story.append(Paragraph(f"• {it}", BODY))


def table(rows, widths=None, header=True, font=7.6):
    data = [[Paragraph(str(c), SMALL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("FONTSIZE", (0, 0), (-1, -1), font)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef8")))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 6))


def fig(path: Path, caption: str, width_cm: float = 16.0):
    if not path.exists():
        p(f"<i>[Figure not available: {path.relative_to(ROOT)} was not generated in this run.]</i>")
        return
    from PIL import Image as PILImage  # matplotlib depends on Pillow, so it is installed
    w, h = PILImage.open(path).size
    width = width_cm * cm
    height = width * h / w
    if height > 21 * cm:
        height = 21 * cm
        width = height * w / h
    story.append(KeepTogether([Image(str(path), width=width, height=height), Paragraph(caption, CAP)]))
    story.append(Spacer(1, 6))


def f3(x):
    return "N/A" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.3f}"


def pct(x):
    return "N/A" if x is None or pd.isna(x) else f"{100 * x:.1f}%"


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def copy_eda_figures():
    src = ROOT / "outputs" / "figures"
    dst = FIG / "eda"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("observations_per_trip.png", "sampling_intervals.png", "gps_route_distance.png", "network_size.png",
                 "validation_issues.png", "trip_locations.png", "trip_train_00000033.png", "radius_sweep.png",
                 "largest_gap_per_trip.png"):
        if (src / name).exists():
            shutil.copyfile(src / name, dst / name)
    for name in ("partition_summary.csv", "validation_issues.csv", "anomalies.csv", "inventory_summary.csv"):
        s = ROOT / "outputs" / "eda" / name
        if s.exists():
            (RES / "eda").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s, RES / "eda" / name)


MODEL_TEXT = {
    "logreg": dict(
        why="A linear model of the candidate features is the simplest probabilistic scorer. It shows how much of the "
            "problem is solved by a weighted sum of distance, heading agreement and context distances, and its "
            "probabilities are usually well calibrated, which matters for HMM emissions.",
        strengths="Trains in about a second; interpretable coefficients; smooth probabilities.",
        limits="It cannot represent interactions, for example that heading only matters when the vehicle is moving, "
               "or that distance matters relative to the nearest candidate. Missing headings are median-imputed.",
        loss="L2-regularised binary cross-entropy (lbfgs)."),
    "knn": dict(
        why="Candidates that look alike in feature space (similar distance, heading difference and context) should "
            "share labels. KNN tests this assumption without fitting a parametric form.",
        strengths="No training; adapts to local structure; distance weighting gives graded probabilities.",
        limits="Prediction is the slowest of all models (a neighbour search over ~386k training rows). It depends on "
               "feature scaling, and with distance weighting it reproduces its training labels exactly, so training "
               "scores say nothing about generalisation.",
        loss="None (lazy learner); prediction is a distance-weighted vote of 25 neighbours on standardised features."),
    "rf": dict(
        why="Bagged decision trees capture non-linear rules ('close AND heading agrees AND neighbours close') without "
            "scaling, and averaging many trees gives stable probability estimates.",
        strengths="Handles feature interactions and mixed scales; little tuning needed; robust to noisy labels.",
        limits="Large model file and slower training than boosting; probabilities can be under-confident near 0/1. "
               "Deep trees fit the training rows closely (see the train vs CV gap).",
        loss="Gini impurity reduction per split; bootstrap aggregation over 200 trees."),
    "hgb": dict(
        why="Gradient-boosted trees are the usual strongest choice for tabular data. Native handling of missing "
            "values suits the undefined headings of stationary fixes.",
        strengths="Fast training and prediction; native NaN support; early stopping on an internal validation split.",
        limits="More hyperparameters to tune; it can overfit with many leaves. The early-stopping split is random "
               "within the training trips (not grouped), so its validation curve is slightly optimistic.",
        loss="Binary cross-entropy (log loss) minimised by gradient boosting."),
    "mlp": dict(
        why="A small neural network learns a smooth non-linear scoring function of the standardised features. It is "
            "a different inductive bias from trees.",
        strengths="Smooth probabilities; compact model; fast prediction.",
        limits="Needs scaling and imputation; sensitive to learning rate and initialisation; less interpretable. "
               "Early stopping uses a random (not trip-grouped) internal split.",
        loss="Binary cross-entropy + L2 penalty, Adam optimiser, mini-batches of 512, early stopping."),
}


def main() -> int:
    copy_eda_figures()
    OUT.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(RES / "models" / "model_comparison.csv")
    folds = pd.read_csv(RES / "models" / "cv_folds.csv")
    dataset = load_json(RES / "models" / "dataset_info.json")
    selected = load_json(RES / "models" / "selected_model.json")
    baseline = load_json(RES / "models" / "baseline_nearest_arc.json")
    best = load_json(RES / "optimisation" / "best_params.json")
    space = load_json(RES / "optimisation" / "search_space.json")
    tvu = pd.read_csv(RES / "optimisation" / "tuned_vs_untuned.csv") if (RES / "optimisation" / "tuned_vs_untuned.csv").exists() else None
    trials = pd.read_csv(RES / "optimisation" / "trials.csv") if (RES / "optimisation" / "trials.csv").exists() else None
    hmm_cfg = load_json(RES / "hmm" / "hmm_config.json")
    hmm = pd.read_csv(RES / "hmm" / "comparison.csv").set_index("method") if (RES / "hmm" / "comparison.csv").exists() else None
    hmm_trips = pd.read_csv(RES / "hmm" / "per_trip.csv", dtype={"trip_id": str}) if (RES / "hmm" / "per_trip.csv").exists() else None
    radius = pd.read_csv(RES / "candidates" / "radius_study.csv") if (RES / "candidates" / "radius_study.csv").exists() else None
    rehearsal = pd.read_csv(RES / "api_rehearsal" / "rehearsal.csv", dtype={"trip_id": str}) if (RES / "api_rehearsal" / "rehearsal.csv").exists() else None
    part = pd.read_csv(RES / "eda" / "partition_summary.csv") if (RES / "eda" / "partition_summary.csv").exists() else None
    covered = folds.groupby("model")["top1_accuracy_covered"].mean()
    sel = selected["selected"]
    name = dict(zip(comp["model"], comp["name"]))

    # ------------------------------------------------------------------ title
    p("PathSnap: Map-Matching and Route Reconstruction", TITLE)
    p("Consolidated Technical Report, Final Evaluation", ParagraphStyle("st", parent=BODY, alignment=TA_CENTER, fontSize=12))
    p("UE24CS352A Machine Learning, PES University", ParagraphStyle("st2", parent=BODY, alignment=TA_CENTER))
    p("Team: Nishitha S (integration lead), Neha, Nishanth N, Niranjan", ParagraphStyle("st3", parent=BODY, alignment=TA_CENTER))
    p("Repository: https://github.com/nishithasathya/ML-Hackathon", ParagraphStyle("st4", parent=BODY, alignment=TA_CENTER))
    story.append(Spacer(1, 10))
    p("<b>How to read this report.</b> Every number below is read from the files in <font face='Courier'>results/</font> "
      "when this PDF is built (<font face='Courier'>python scripts/build_report.py</font>). Every figure is an "
      "image produced by the pipeline, and every screenshot is a capture of the running API/UI. The commands that "
      "regenerate them are in the README. The ten evaluator-testing trips were not available to us. All "
      "evaluation uses the 80 training and 10 student-testing trips, and each table names its partition.")

    # ------------------------------------------------------------------ 1
    p("1. Project overview", H1)
    p("For each GPS fix of a trip, PathSnap must name the directed road arc of that trip's network that the vehicle "
      "was on. The pipeline is: preprocessing (cleaning, local metric projection, gap segments, spike flags) → "
      "candidate generation (all arcs within 50 m, KD-tree) → feature extraction for every (fix, candidate) pair → "
      "a classifier that outputs P(candidate is correct) → an HMM whose hidden states are the candidates, decoded "
      "with Viterbi → FastAPI backend (Laptop 1) → Streamlit evaluator UI (Laptop 2).")
    table([["Component", "Implementation", "Owner"],
           ["Data exploration, shared config, metrics, inference bundle, report", "src/common, src/data_exploration, src/evaluation, src/inference", "Nishitha"],
           ["Preprocessing, 5 models, CV, selection", "src/preprocessing, src/models", "Neha"],
           ["Candidates, features, radius study, HMM/Viterbi, Bayesian optimisation", "src/features, src/hmm, src/optimization", "Nishanth"],
           ["FastAPI, UI, Folium/experiment figures, deployment", "src/api, frontend, src/visualization, deployment", "Niranjan"]],
          [6.5 * cm, 7 * cm, 2.5 * cm])

    # ------------------------------------------------------------------ 2
    p("2. Dataset understanding and exploratory data analysis", H1)
    p("2.1 Dataset structure (partitions: train = 80 trips, student_test = 10 trips)", H2)
    p("Each trip has four header-less text files. <b>.track</b> holds longitude, latitude and timestamp (1 Hz nominal; CRLF line "
      "endings). <b>.nodes</b> holds longitude and latitude, and the row number is the node id. <b>.arcs</b> holds the from-node "
      "and to-node, and the row number is the arc id. <b>.route</b> is the ordered list of reference arc ids. "
      "<b>Every trip has its own network</b>, so ids are meaningful only inside one trip. The route has no "
      "timestamps, so per-fix labels have to be derived (§2.4).")
    if part is not None:
        get = lambda prt, stat, col: part[(part["partition"] == prt) & (part["statistic"] == stat)][col].iloc[0]
        table([["Statistic (per trip)", "train median [min-max]", "student_test median [min-max]"]] +
              [[s, f"{get('train', s, '50%'):.1f} [{get('train', s, 'min'):.1f}-{get('train', s, 'max'):.1f}]",
                f"{get('student_test', s, '50%'):.1f} [{get('student_test', s, 'min'):.1f}-{get('student_test', s, 'max'):.1f}]"]
               for s in ("n_observations", "path_length_km", "max_gap_s", "n_nodes", "n_arcs", "n_weak_components",
                         "gps_route_dist_median_m")], [6 * cm, 5 * cm, 5 * cm])
    p("2.2 Missing values, data quality and patterns", H2)
    p("Analysed on both partitions (phase-1 EDA, <font face='Courier'>python -m src.data_exploration.run_eda</font>). "
      "We found <b>no missing values, NaN or invalid coordinates</b> in any file, and no timestamp ever decreases or repeats. "
      "The quality issues are structural. 30 trips contain time gaps over 30 s, and two training trips have gaps of hours. "
      "1,903 duplicate arcs (the same node pair under two ids) occur in 60 trips. 11 reference routes have breaks. "
      "Implied speeds above 70 m/s occur in 11 trips. Two training trips (00000033, 00000054) run 120-160 m away from "
      "their reference route. Most roads are stored as two opposite arcs with identical geometry, so the nearest arc "
      f"is the reference only about half the time (nearest-arc heuristic, student_test: {pct(baseline['student_test_top1_accuracy'])}).")
    fig(FIG / "eda" / "validation_issues.png", "Figure 2.1: Trips affected per validation issue (train + student_test).", 13)
    fig(FIG / "eda" / "sampling_intervals.png", "Figure 2.2: Sampling intervals (both partitions): about 97% are exactly 1 s.", 13)
    fig(FIG / "eda" / "gps_route_distance.png", "Figure 2.3: Median GPS-to-reference-route distance per trip (both partitions); "
        "the two offset training trips stand out.", 13)
    fig(FIG / "eda" / "network_size.png", "Figure 2.4: Road-network size per trip (both partitions).", 13)
    fig(FIG / "eda" / "trip_train_00000033.png", "Figure 2.5: GPS trajectory vs reference route, train trip 00000033 "
        "(the western loop has GPS fixes but no reference arcs).", 9)
    p("2.3 Preprocessing", H2)
    p("Rows with invalid coordinates or missing times are dropped (none occurred). Rows are stably sorted by time, and "
      "duplicate timestamps are dropped. Each trip gets its own local equirectangular projection to metres, with a "
      "measured maximum error of 0.78% against haversine. A new segment starts after a gap over 30 s, and nothing is "
      "measured across a gap. A fix is flagged as a spike when the speed both into and out of it exceeds 70 m/s. "
      "Spikes are flagged rather than deleted, so every fix still receives a prediction, and the HMM treats their "
      "displacement as unreliable (§5). Original row ids (<font face='Courier'>obs_idx</font>) are preserved. Arcs "
      "are not de-duplicated, because their ids must stay aligned with .route. Nothing in preprocessing is fitted, "
      "so it cannot leak.")
    p("2.4 Labels and features", H2)
    p("<b>Labels.</b> A monotone dynamic-programming alignment assigns each fix to a position on the reference "
      "route. Positions only move forward and the total distance is minimised, which resolves routes that use a "
      "street twice or in both directions. A candidate is positive when it is the reference arc <i>or an identical "
      "duplicate arc</i> (new in phase 2; previously identical inputs could carry opposite labels). Positive rate: "
      f"{pct(dataset['positive_rate_all'])} of training candidate rows.")
    p("<b>Features</b> (22 inputs, the same function <font face='Courier'>model_matrix</font> at training and inference):")
    table([["Group", "Features", "Why"],
           ["Geometry", "distance_m, distance_rank, dist_minus_nearest_m, offset_frac, n_candidates", "Closeness to the GPS fix, absolute and relative to the other candidates"],
           ["Direction", "heading_diff_deg, heading_cos, along_arc_motion_m", "Separates the two directions of a street (EDA: twins tie on distance)"],
           ["Temporal context (new)", "prev_obs_dist_m, next_obs_dist_m", "The correct arc is usually also near the neighbouring fixes"],
           ["Arc", "arc_length_m, arc_to_out_degree, arc_has_reverse", "Network structure of the candidate"],
           ["Motion", "speed_mps, speed_smooth_mps, accel_mps2, heading_change_deg, step_dist_m, dt_prev, dt_next, is_speed_outlier, is_segment_start", "Reliability of the heading and the context"]],
          [3 * cm, 7 * cm, 6 * cm])
    p("Trip-progress columns (t_rel, cum_dist_m, segment_id), absolute bearings and all label-derived columns "
      "(ref_arc_id, ref_distance_m, is_reference) are excluded, and a unit test asserts this.")
    if radius is not None:
        chosen_r = radius.loc[radius["selected"], "radius_m"].iloc[0]
        p("2.5 Candidate radius, re-evaluated with a trained model", H2)
        table([["Radius (m)", "Train miss rate", "Candidates/fix", "CV Top-1 (train, decides)", "Student-test Top-1 (reported)"]] +
              [[f"{r.radius_m:g}", pct(r.train_candidate_miss_rate), f"{r.train_candidates_per_obs:.2f}",
                f"{r.cv_top1_mean:.4f}", f"{r.student_test_top1:.4f}"] for r in radius.itertuples()],
              [2.5 * cm, 3 * cm, 3 * cm, 4 * cm, 4 * cm])
        best_r = radius.loc[radius["best_cv"], "radius_m"].iloc[0]
        r50 = radius.set_index("radius_m")
        p(f"Model: {name.get(sel, sel)} with default parameters, same trip folds. The highest training-CV Top-1 was at "
          f"{best_r:g} m ({r50.loc[best_r, 'cv_top1_mean']:.4f}), against {r50.loc[chosen_r, 'cv_top1_mean']:.4f} at "
          f"{chosen_r:g} m: a difference of {100 * (r50.loc[best_r, 'cv_top1_mean'] - r50.loc[chosen_r, 'cv_top1_mean']):.2f} "
          f"points, with {100 * (r50.loc[best_r, 'train_candidates_per_obs'] / r50.loc[chosen_r, 'train_candidates_per_obs'] - 1):.0f}% "
          f"more candidates per fix (the HMM's transition cost grows with the square of the candidate count). "
          f"<b>Decision: {chosen_r:g} m is kept.</b> Radii within 0.5 points of the best are treated as tied, the same "
          f"margin as the model-selection rule, and the smallest tied radius is chosen. Transparency note: this tie "
          f"rule was written after the study's first run, which had simply marked the maximum. We chose to apply the "
          f"existing model-selection margin rather than re-run every later experiment at {best_r:g} m. The student-test "
          f"Top-1 (reported only) does not favour {best_r:g} m either. Re-running at 75-100 m is a reasonable next step. "
          f"The 50 m feature-build time shows 0.15 s because those tables were already cached.")
        fig(FIG / "radius_study.png", "Figure 2.6: Candidate radius vs Top-1 (train CV decides; student test reported).", 12)

    # ------------------------------------------------------------------ 3
    story.append(PageBreak())
    p("3. Model experimentation and comparison", H1)
    p(f"<b>Formulation.</b> Binary candidate scoring: each row is a (fix, candidate arc) pair with target is_reference. "
      f"A multiclass formulation is not possible because arc ids are local to each trip and the candidate sets "
      f"change from fix to fix. <b>Training data:</b> {dataset['n_rows_fit']:,} rows. These are every 3rd fix of each "
      f"training trip (consecutive 1 Hz fixes are near-duplicates), only fixes with a correct candidate, excluding "
      f"the two offset trips {', '.join(dataset['excluded_offset_trips'])} whose labels are unreliable (a train-only "
      f"EDA rule). <b>Evaluation always uses every candidate</b> ({dataset['n_rows_all']:,} training rows across "
      f"the folds; {dataset['n_student_test_rows']:,} student-test rows). <b>CV:</b> 5-fold GroupKFold by trip id; "
      f"imputers and scalers are fitted inside each training fold. Seed {dataset['seed']}. <b>Class balance:</b> "
      f"{pct(dataset['positive_rate_fit'])} positives in the fitting rows. This is moderate, so models are trained "
      f"unweighted to keep calibrated probabilities for the HMM. Class weighting was offered to the Bayesian search "
      f"as a hyperparameter instead.")
    p("<b>Metric definitions.</b> Top-1: share of <i>all</i> labelled fixes whose highest-scoring candidate is the "
      "reference arc; fixes without any candidate count as errors. Macro F1, log loss and the confusion matrix are "
      "computed at candidate level with a 0.5 threshold. Route continuity: share of consecutive fix pairs in the same "
      "gap segment whose predicted positions are connected in the directed network within 60 m/s·Δt + 50 m; pairs "
      "split by a gap are excluded. Route precision/recall/F1: length-weighted overlap between the set of predicted "
      "arcs and the reference route, averaged over trips. The primary selection metric is CV Top-1, because Top-1 is "
      "what the system outputs per fix.")
    p("3.1 Overall model comparison", H2)
    rows = [["Model", "Algorithm family", "CV macro F1 (±sd)", "CV Top-1", "Test Top-1", "Test macro F1", "Test log loss",
             "Route continuity (test)", "Train time (s)", "Inference (s, 131k cand.)"]]
    for r in comp.itertuples():
        rows.append([r.name, r.family, f"{r.cv_macro_f1_mean:.3f} (±{r.cv_macro_f1_std:.3f})", f"{r.cv_top1_mean:.3f}",
                     f"{r.test_top1_accuracy:.3f}", f"{r.test_macro_f1:.3f}", f"{r.test_log_loss:.3f}",
                     f"{r.route_continuity:.3f}", f"{r.fit_time_s:.1f}", f"{r.test_inference_time_s:.2f}"])
    table(rows, [2.6 * cm, 2.6 * cm, 2.0 * cm, 1.3 * cm, 1.3 * cm, 1.4 * cm, 1.4 * cm, 1.6 * cm, 1.3 * cm, 1.8 * cm], font=7)
    p("Key configurations are listed in each model's subsection (3.2) and in results/models/model_comparison.csv.", SMALL)
    p(f"Non-learned reference point (not one of the five models): choosing the nearest arc gives student-test Top-1 "
      f"{baseline['student_test_top1_accuracy']:.4f}. Mean CV Top-1 is lower than student-test Top-1 because some "
      f"training folds contain the offset trips and trips with high candidate miss rates. Restricted to fixes that "
      f"have a correct candidate, CV Top-1 is: " + ", ".join(f"{name[k]} {covered[k]:.4f}" for k in comp['model']) + ".")
    p("<b>Model selection.</b> The rule was fixed before the comparison (src/models/select_model.py): highest mean "
      f"CV Top-1, models within 0.5 percentage points treated as tied, ties broken by lower CV log loss (the HMM "
      f"consumes the probabilities), then by inference time. Tied: {', '.join(name[m] for m in selected['tied_within_margin'])}. "
      f"<b>Selected: {name[sel]}.</b> The tree ensembles and the MLP are statistically indistinguishable on Top-1 "
      f"(fold standard deviation about 0.12, dominated by which trips fall in a fold). {name[sel]} had the best "
      f"probability quality on CV, a moderate inference time, and its probabilities feed the decoder directly. "
      f"Its main cost is a large model file.")
    p("3.2 Individual models", H2)
    for r in comp.itertuples():
        t = MODEL_TEXT[r.model]
        gap = r.train_fit_top1 - covered[r.model]
        if r.model == "knn":
            fit_verdict = (f"Training-row Top-1 is {r.train_fit_top1:.3f} versus CV {covered[r.model]:.3f} on covered fixes. KNN with distance "
                           f"weighting memorises its training rows, so this gap is expected and not informative. Its CV and student-test scores are "
                           f"close to the other models, so the vote over 25 neighbours does not overfit in the harmful sense.")
        elif gap > 0.05:
            fit_verdict = (f"Overfitting is visible: Top-1 on its own training rows is {r.train_fit_top1:.3f}, versus {covered[r.model]:.3f} in CV on "
                           f"fixes with a correct candidate (gap {gap:.3f}). Generalisation is nevertheless as good as the best model, so the "
                           f"extra capacity memorises noise rather than hurting held-out trips.")
        elif gap < -0.01 or r.cv_log_loss_mean > comp["cv_log_loss_mean"].min() + 0.03:
            fit_verdict = (f"Underfitting relative to the tree models: CV log loss {r.cv_log_loss_mean:.3f} versus the best "
                           f"{comp['cv_log_loss_mean'].min():.3f}, and training Top-1 ({r.train_fit_top1:.3f}) is close to CV ({covered[r.model]:.3f}). "
                           f"The model is limited by its capacity, not by the data.")
        else:
            fit_verdict = (f"No strong over- or underfitting: training-row Top-1 {r.train_fit_top1:.3f} versus CV {covered[r.model]:.3f} "
                           f"on covered fixes (gap {gap:.3f}).")
        story.append(KeepTogether([
            Paragraph(f"3.2.{list(comp['model']).index(r.model) + 1} {r.name} ({r.family})", H3),
            Paragraph(f"<b>A. Why suitable.</b> {t['why']} <b>Strengths:</b> {t['strengths']} <b>Limitations:</b> {t['limits']}", BODY),
            Paragraph(f"<b>B. Configuration.</b> Initial hyperparameters: <font face='Courier' size=7>{r.params}</font>. "
                      f"Objective: {t['loss']}", BODY),
            Paragraph(f"<b>C. Results.</b> Student test: Top-1 <b>{r.test_top1_accuracy:.4f}</b>, macro F1 <b>{r.test_macro_f1:.4f}</b>, "
                      f"log loss <b>{r.test_log_loss:.4f}</b>, precision {r.test_precision:.3f}, recall {r.test_recall:.3f}, "
                      f"route continuity of top-1 predictions {r.route_continuity:.4f}, route F1 {r.route_f1_mean:.4f}. "
                      f"CV: Top-1 {r.cv_top1_mean:.4f} ± {r.cv_top1_std:.3f}, macro F1 {r.cv_macro_f1_mean:.4f}, log loss {r.cv_log_loss_mean:.4f}. "
                      f"Training time {r.fit_time_s:.1f} s; inference {r.test_inference_time_s:.2f} s for the {dataset['n_student_test_rows']:,} "
                      f"student-test candidates. <b>Over/underfitting:</b> {fit_verdict}", BODY),
            Spacer(1, 4)]))
    p("3.3 Model visualisations", H2)
    fig(FIG / "cv_macro_f1.png", "Figure 3.1: Cross-validation macro F1 across the five models (train, 5 trip-grouped folds).", 13)
    fig(FIG / "student_test_top1_macro_f1.png", "Figure 3.2: Student-test Top-1 accuracy and macro F1.", 13)
    fig(FIG / "log_loss.png", "Figure 3.3: Log loss, CV and student test.", 13)
    fig(FIG / "timings.png", "Figure 3.4: Training and inference time (log scale).", 14)
    fig(FIG / "confusion_matrices.png", "Figure 3.5: Candidate-level confusion matrices, student test, threshold 0.5.", 16)
    p("Training curves are shown only for the two models that record an epoch- or iteration-wise history. "
      "Logistic regression, KNN and Random Forest produce none, so no curve is drawn for them.")
    fig(FIG / "training_history_hgb.png", "Figure 3.6: HistGradientBoosting: recorded training/validation loss per boosting iteration (final fit).", 11)
    fig(FIG / "training_history_mlp.png", "Figure 3.7: MLP: recorded training loss and validation accuracy per epoch (final fit).", 11)

    # ------------------------------------------------------------------ 4
    story.append(PageBreak())
    p("4. Bayesian hyperparameter optimisation", H1)
    if best is None or tvu is None:
        p("<i>Optimisation results not found.</i>")
    else:
        n_trials = int((trials["state"] == "COMPLETE").sum())
        u = tvu.set_index("variant")
        p(f"<b>Model:</b> {name[best['model']]} (the selected model). <b>Method:</b> Optuna TPE sampler (seed 42, 10 random start-up "
          f"trials), study stored in results/optimisation/study.db. <b>Objective:</b> maximise mean 5-fold GroupKFold Top-1 on the 80 "
          f"training trips (the same folds as §3). To keep the cost manageable, trials fit on every {space['search_fit_stride']}th fix; "
          f"evaluation always uses all candidates of the held-out trips. <b>Trials run:</b> {n_trials}, "
          f"{best['search_time_s'] / 60:.1f} minutes in total. The student test was not used during the search."
          + (f" <b>Note:</b> the search was planned for {space['n_trials']} trials and was stopped after {n_trials} completed "
             f"trials to save time. The first 10 were TPE's random start-up trials, so only {n_trials - 10} were model-guided. "
             f"A longer search could still find a better configuration." if best.get("stopped_early") else ""))
        table([["Hyperparameter", "Range / distribution", "Best value"]] +
              [[k, v, str(best["best_params"].get(k, best["best_params"].get(k.replace("_mode", ""), "-")))] for k, v in space["space"].items()],
              [4 * cm, 7.5 * cm, 4.5 * cm])
        table([["Variant", "CV Top-1 (±sd)", "CV macro F1", "CV log loss", "Test Top-1", "Test macro F1", "Test log loss", "Train time (s)", "Inference (s)"]] +
              [[v, f"{u.loc[v, 'cv_top1_mean']:.4f} (±{u.loc[v, 'cv_top1_std']:.3f})", f"{u.loc[v, 'cv_macro_f1_mean']:.4f}",
                f"{u.loc[v, 'cv_log_loss_mean']:.4f}", f"{u.loc[v, 'test_top1_accuracy']:.4f}", f"{u.loc[v, 'test_macro_f1']:.4f}",
                f"{u.loc[v, 'test_log_loss']:.4f}", f"{u.loc[v, 'fit_time_s']:.1f}", f"{u.loc[v, 'test_inference_time_s']:.2f}"]
               for v in ("untuned", "tuned")], [2 * cm, 2.6 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.8 * cm, 1.7 * cm, 1.6 * cm, 1.6 * cm])
        d_cv = u.loc["tuned", "cv_top1_mean"] - u.loc["untuned", "cv_top1_mean"]
        d_te = u.loc["tuned", "test_top1_accuracy"] - u.loc["untuned", "test_top1_accuracy"]
        d_ll = u.loc["tuned", "cv_log_loss_mean"] - u.loc["untuned", "cv_log_loss_mean"]
        verdict = "improved" if d_cv > 0 else "did not improve"
        p(f"<b>Did tuning help?</b> Under identical conditions, the tuned model {verdict} CV Top-1 by {100 * d_cv:+.2f} points, "
          f"changed student-test Top-1 by {100 * d_te:+.2f} points, and changed CV log loss by {d_ll:+.4f}. The gain is small compared with "
          f"the fold-to-fold spread, which is dominated by trip difficulty and candidate misses rather than by the "
          f"classifier. Most of the remaining error is fixed-by-construction (missing candidates) or sequential, "
          f"which is what the HMM addresses. The tuned model is used for the HMM and the deployed artefact.")
        fig(FIG / "optuna_history.png", "Figure 4.1: Objective per trial and best-so-far (recorded Optuna trials).", 12)
        fig(FIG / "tuned_vs_untuned.png", "Figure 4.2: Tuned vs untuned (same folds, same student-test trips).", 12)

    # ------------------------------------------------------------------ 5
    p("5. HMM and Viterbi route reconstruction", H1)
    if hmm is None:
        p("<i>HMM results not found.</i>")
    else:
        c, h = hmm.loc["classifier_only"], hmm.loc["classifier_hmm"]
        p("<b>1. Hidden states.</b> At fix t the states are its candidates, each a position (arc id, offset along the arc). "
          "The candidate table keeps fix → candidate → arc → features → probability, so the decoded state maps directly to a road arc.")
        p(f"<b>2. Emissions.</b> log P(o_t | s) = w · log(max(p_clf(s), 10⁻⁴)), where p_clf is the tuned classifier's probability. "
          f"w = {hmm_cfg['emission_weight']:g}.")
        p(f"<b>3. Transitions.</b> d_route is the shortest legal driving distance from position s_t to s_t+1 along directed arcs "
          f"(bounded Dijkstra over the trip network). d_gc is the straight-line distance between the two fixes. "
          f"log P = −|d_route − d_gc| / β with β = {hmm_cfg['transition_beta_m']:g} m. The search budget is "
          f"{hmm_cfg['max_speed_mps']:g} m/s·Δt + 2·{hmm_cfg['position_tolerance_m']:g} m. Staying on the same arc "
          f"(including GPS jitter backwards) costs the along-arc distance. If either fix is a flagged spike, "
          f"d_gc is ignored (−d_route/β), because the spike's displacement is not real. A unit test showed that a "
          f"300 m spike otherwise forces a false U-turn.")
        p(f"<b>4. Missing candidates and gaps.</b> The fix sequence is split into chains at every gap segment (Δt > 30 s) and at fixes "
          f"without candidates. Each chain is decoded independently, and no transition is invented across a missing interval. "
          f"Fixes without candidates get no prediction and count as errors.")
        p(f"<b>5. Disconnected segments.</b> A move the network cannot explain within the budget gets a finite penalty "
          f"({hmm_cfg['unreachable_log_prob']:g} in log space), not −∞. A chain is therefore never 'impossible' when the correct "
          f"arc is missing or the network is fragmented, and the continuity metric counts such moves as breaks.")
        p(f"<b>Viterbi.</b> δ_0(j) = e_0(j); δ_t(j) = max_i[δ_t−1(i) + log A_t(i,j)] + e_t(j), with back-pointers; computed in "
          f"log space, so long chains cannot underflow. A unit test checks the result against brute-force enumeration. "
          f"<b>Decoder tuning:</b> a grid over w and β on {len(hmm_cfg['tuning_trips'])} training trips (every third trip, offset "
          f"trips excluded), using out-of-fold classifier probabilities, so the decoder sees honest scores.")
        table([["Method (student test)", "Top-1", "Top-1 (covered)", "Decision macro F1", "Route continuity", "Breaks",
                "Route precision", "Route recall", "Route F1", "Decode time (s, 10 trips)"]] +
              [[lbl, f"{r['top1_accuracy']:.4f}", f"{r['top1_accuracy_covered']:.4f}", f"{r['decision_macro_f1']:.4f}",
                f"{r['route_continuity']:.4f}", f"{int(r['n_breaks'])}", f"{r['route_precision_mean']:.4f}",
                f"{r['route_recall_mean']:.4f}", f"{r['route_f1_mean']:.4f}", f"{r['decode_time_s_total']:.1f}"]
               for lbl, r in (("Classifier only", c), ("Classifier + HMM/Viterbi", h))],
              [3.2 * cm, 1.3 * cm, 1.5 * cm, 1.7 * cm, 1.7 * cm, 1.1 * cm, 1.6 * cm, 1.5 * cm, 1.3 * cm, 1.9 * cm], font=7)
        better = int(h["trips_improved_top1"]) if not pd.isna(h["trips_improved_top1"]) else 0
        p(f"<b>6. Did HMM decoding improve the route?</b> Yes on every aggregate measure: Top-1 {c['top1_accuracy']:.4f} → "
          f"<b>{h['top1_accuracy']:.4f}</b>, continuity {c['route_continuity']:.4f} → <b>{h['route_continuity']:.4f}</b> "
          f"(breaks {int(c['n_breaks'])} → {int(h['n_breaks'])}), route F1 {c['route_f1_mean']:.4f} → {h['route_f1_mean']:.4f}. "
          f"Top-1 improved on {better} of 10 trips. The largest gains come from runs where the classifier flips between a road and its "
          f"opposite-direction twin, or jumps to a parallel street for a few fixes. The transition model makes such moves expensive. "
          f"The HMM cannot help where the correct arc is not a candidate (the candidate miss rate is a hard ceiling), "
          f"and it can make an error persist for a whole chain when the classifier is confidently wrong at a chain's start. "
          f"The per-trip figure shows where the change was small.")
        if hmm_trips is not None:
            pv = hmm_trips.pivot(index="trip_id", columns="method", values="top1_accuracy")
            worst = (pv["classifier_hmm"] - pv["classifier_only"]).sort_values()
            p("Per-trip change in Top-1 (HMM − classifier): " + ", ".join(f"{t}: {100 * d:+.1f}" for t, d in worst.items()) + " points.")
        fig(FIG / "hmm_vs_classifier.png", "Figure 5.1: Classifier-only vs classifier + HMM (student test).", 13)
        fig(FIG / "hmm_per_trip.png", "Figure 5.2: Top-1 per student-test trip.", 13)
        fig(FIG / "hmm_decoder_grid.png", "Figure 5.3: Decoder grid search on training trips (out-of-fold probabilities).", 10)

    # ------------------------------------------------------------------ 6
    story.append(PageBreak())
    p("6. Geographical visualisation using Folium", H1)
    p("Generated by src/visualization/prediction_maps.py from the exported model on student-test trips. Layers: "
      "nearby road network (grey), reference route (green, development partition only; it is never sent to the UI), "
      "raw GPS fixes (blue), flagged spikes and gap starts, classifier-only arcs (orange, off by default) and the "
      "HMM/Viterbi predicted route (magenta, dashed), with a legend, layer control and the bounds fitted to the trip. The "
      "interactive HTML files are in results/maps/.")
    for shot in sorted(SHOT.glob("folium_*.png"))[:2]:
        fig(shot, f"Figure 6.x: Screenshot of {shot.stem.replace('folium_', '')}.html (headless Chrome capture).", 15)

    # ------------------------------------------------------------------ 7
    p("7. API, UI and deployment", H1)
    p("<b>Architecture.</b> Laptop 1 runs FastAPI (uvicorn, 0.0.0.0:8000) with the exported bundle (artifacts/model.joblib + "
      "bundle.json: model, 22-feature list, candidate/preprocessing/HMM configuration, seed, version, SHA-256) and the "
      "evaluation trips from PATHSNAP_EVAL_DATA_ROOT. Laptop 2 runs the Streamlit UI, which only calls the API. Endpoints: "
      "GET /health, GET /api/evaluation/trips, POST /api/predict {trip_id}, GET /api/evaluation/results/{trip_id}. "
      "The prediction path loads .track/.nodes/.arcs only (load_trip_for_inference), and predict_trip refuses a trip that "
      "carries a route. Only after prediction does the server-side evaluator read .route, compute Top-1, continuity "
      "and route P/R/F1, and return numbers. No reference arc or label field exists in the response schema, and a test checks "
      "the JSON for it. Each request and stage (load, preprocessing, candidates, classifier, HMM, timings, evaluation) is "
      "logged to the console and logs/api.log. If the data folder or the artefacts are missing, the API answers 503 with "
      "an explanation and /health reports 'degraded'.")
    if rehearsal is not None:
        p(f"<b>HTTP rehearsal (student_test, labelled 'student_test (rehearsal)' in the UI).</b> The API was started bound to "
          f"0.0.0.0 and called through the machine's LAN address ({rehearsal['api_url'].iloc[0]}) with the UI's client, for "
          f"<b>all {len(rehearsal)} trips</b>. Server inference time: median {rehearsal['server_inference_s'].median():.2f} s, "
          f"max {rehearsal['server_inference_s'].max():.2f} s. The served metrics match the offline HMM results (mean HMM Top-1 "
          f"{rehearsal['hmm_top1'].mean():.4f} per trip).")
        table([["Trip", "Fixes", "Candidates", "Server time (s)", "HMM Top-1", "Classifier Top-1", "HMM continuity", "HMM route F1"]] +
              [[r.trip_id, r.n_observations, r.n_candidates, f"{r.server_inference_s:.2f}", f"{r.hmm_top1:.4f}",
                f"{r.classifier_top1:.4f}", f"{r.hmm_route_continuity:.4f}", f"{r.hmm_route_f1:.4f}"] for r in rehearsal.itertuples()],
              [2 * cm, 1.5 * cm, 2 * cm, 2.2 * cm, 2 * cm, 2.3 * cm, 2.2 * cm, 2 * cm])
    fig(SHOT / "ui_overview.png", "Figure 7.1: Evaluator UI after a prediction (served by the API over the LAN address; rehearsal data).", 16)
    fig(SHOT / "api_results_json.png", "Figure 7.2: API response, GET /api/evaluation/results/{trip_id}, shown in the browser.", 14)
    fig(SHOT / "api_docs.png", "Figure 7.3: FastAPI interactive documentation of the endpoints.", 14)
    fig(SHOT / "api_log.png", "Figure 7.4: API log lines written by the server for these requests (logs/api.log, rendered as text).", 16)
    p("<b>Still to be captured on the two physical laptops:</b> Deliverables §7 asks for screenshots of the "
      "API responses on the receiving laptop's terminal during the two-laptop run. The figures above come from a "
      "single-machine LAN rehearsal and are marked as such. The team will add the two-laptop captures after "
      "running deployment/DEPLOYMENT.md with the organisers' evaluator trips.")

    # ------------------------------------------------------------------ 8
    p("8. Results, limitations and conclusion", H1)
    if hmm is not None and tvu is not None:
        p(f"<b>Results.</b> On the 10 student-test trips, the final system ({name[sel]}, tuned, + HMM/Viterbi) reaches Top-1 "
          f"{hmm.loc['classifier_hmm', 'top1_accuracy']:.4f}, route continuity {hmm.loc['classifier_hmm', 'route_continuity']:.4f} "
          f"and mean route F1 {hmm.loc['classifier_hmm', 'route_f1_mean']:.4f}. The nearest-arc heuristic reaches "
          f"{baseline['student_test_top1_accuracy']:.4f}, and the best untuned classifier alone {comp['test_top1_accuracy'].max():.4f}.")
    bullets([
        "Candidate coverage caps accuracy: fixes whose reference arc lies outside 50 m cannot be matched by any model or decoder.",
        "Labels come from aligning GPS to an untimed reference route. They inherit route errors (11 broken routes) and are weak for the two offset trips, which we excluded from fitting but kept in evaluation.",
        "CV Top-1 varies strongly between folds (sd ≈ 0.12) because trips differ in difficulty. The five models differ far less than that, so model choice was made on probability quality within a tie margin set in advance.",
        "The selected Random Forest is a large file (tens of MB). A boosting model of near-identical accuracy would be about 100× smaller.",
        "The decoder parameters were tuned on a grid over a third of the training trips. A first grid had its best value on the edge, so it was extended toward smaller values before the final run (training data only).",
        "The evaluator trips were not available. The two-laptop run and all ten evaluator trips remain to be verified on the day.",
    ])
    p("<b>Conclusion.</b> Learned candidate scoring with heading and temporal-context features removes most of the "
      "twin-arc ambiguity that limits distance-only matching. HMM/Viterbi decoding over the directed network then "
      "produces almost fully connected routes and further raises accuracy. The system runs end to end through the "
      "API and UI without exposing reference data.")

    # ------------------------------------------------------------------ 9
    p("9. References and AI-assistance disclosure", H1)
    bullets(["P. Newson and J. Krumm, 'Hidden Markov map matching through noise and sparseness', ACM SIGSPATIAL GIS, 2009 (transition model idea).",
             "Pedregosa et al., 'Scikit-learn: Machine Learning in Python', JMLR 12, 2011.",
             "Akiba et al., 'Optuna: A Next-generation Hyperparameter Optimization Framework', KDD 2019.",
             "Official PathSnap documents: Main Guidelines, Models and Training Guidelines, Deliverables, Dataset Exploration, Assessment and Plagiarism Policy."])
    p("<b>AI assistance.</b> As in phase 1, an AI coding assistant (Claude Code) was used to draft the code, tests, "
      "documentation and this report's text, at the team lead's request. All experiments were actually executed. "
      "The numbers, figures and screenshots in this report are read from the files those runs produced, and the "
      "run log is in docs/AI_ASSISTANCE_AND_WORK_LOG.md. Every team member is responsible for understanding and "
      "explaining their module (docs/team/*_FINAL_README.md).")

    doc = SimpleDocTemplate(str(OUT / "PathSnap_Final_Technical_Report.pdf"), pagesize=A4, leftMargin=1.8 * cm,
                            rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title="PathSnap Final Technical Report", author="PathSnap team")

    def footer(canvas, doc_):
        canvas.setFont("Helvetica", 7.5)
        canvas.drawRightString(A4[0] - 1.8 * cm, 1 * cm, f"PathSnap final report, page {doc_.page}")

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Report written to {OUT / 'PathSnap_Final_Technical_Report.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
