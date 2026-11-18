# First-phase scope: what is required vs what we chose

## 1. What the official documents say about the first phase

We read all five supplied PDFs. The **only explicit mention** of the first
phase is the first line of *Deliverables.pdf*:

> "There will be first phase evaluation before lunch break"

No document gives a separate first-phase checklist. Everything else in the
documents describes the **final** submission. Our scope below is therefore
partly **explicit** (from the documents) and partly **inferred** (our own
judgement). The table marks which is which.

### Sources used for the inference

| Source | Statement | How we used it |
|---|---|---|
| Main Guidelines §4 "Expected Work Process" | Stages in order: 1 Understand the problem → 2 Explore the dataset → 3 Develop an independent data-processing and feature-engineering approach → 4 Train/compare models → … → 12 Report | Stages 1-3 are the natural first milestone (inferred) |
| Main Guidelines §2 | Read all documents before implementation | Done; reflected in this file |
| Dataset Exploration §4, §8, §9 | Lists the initial analysis items, the recommended exploration outputs (inventory, summary statistics, plots, **interactive Folium maps**, network connectivity tables, data-quality record) and says that "at the end of dataset exploration" the team should be able to explain the dataset | Treated as the core of phase 1 (explicit list of exploration outputs) |
| Models and Training §5 "Components you'll need" | "Data-processing implementation: raw-data inspection, preprocessing, **candidate generation where appropriate** and feature engineering" | Candidate generation is part of data processing, so we included it |
| Models and Training §5 "Evaluation metrics" | Defines **Candidate Miss Rate** | Measurable as soon as candidates exist; we report it |
| Deliverables §2.2 | Report must cover preprocessing, EDA patterns, extracted features, missing-value analysis, GPS trajectory plots, road-network visualisations, at least one Folium map, partition per analysis | Phase-1 documentation follows these headings |

## 2. Scope table

Legend: **E** = explicitly required by the documents (for the final project) and
started now; **I** = our inferred first-phase choice; **D** = deferred.

| Component | First evaluation (this milestone) | Later evaluation |
|---|---|---|
| Dataset exploration | **Done (E/I).** Inventory, validation, per-trip statistics, connectivity, gaps, anomalies, data-quality notes for all 90 permitted trips | Deeper analysis if needed by model errors |
| Data preprocessing | **Done (I).** Invalid-coordinate removal, time ordering, duplicate-time removal, gap segmentation, speed-spike flagging, per-trip projection | Tune thresholds using model results; optional smoothing |
| Feature engineering | **Done, initial set (I).** Candidate generation (KD-tree), route-to-observation labels, motion, arc and pair features; radius sweep on train | Advanced features (e.g. topology/shortest-path features, multi-observation context); feature selection |
| ML models | **Not started (D).** Feature tables are ready as model input | ≥5 models, group-aware CV, comparison (Models §2, §4, §6) |
| Bayesian optimisation | **Not started (D)** | Search space, objective, trials, tuned vs untuned (Models §7) |
| HMM/Viterbi | **Not started (D).** Candidates and gap segments were designed to become HMM states/breaks | Full route reconstruction and comparison (Models §8-9) |
| FastAPI | **Not started (D)** | Backend on Laptop 1 |
| UI | **Not started (D)** | Evaluator interface on Laptop 2 |
| Deployment | **Not started (D)** | Two-laptop architecture (Main Guidelines §3) |
| Visualisation | **Done (E).** Folium maps with layers + legend; static figures for EDA and the radius sweep | Predicted-route layer (hook already exists) |
| Technical report | **First-phase evidence (I).** `docs/FIRST_PHASE_TECHNICAL_DOC.md` with real numbers | Consolidated final PDF report (Deliverables §2-7) |

## 3. Why not more?

The team asked for an honest first milestone, not a rushed final system.
Training five models before the data, labels and candidate sets are
understood would produce results we could not defend in the viva (Assessment
policy §5-6). The phase-1 code was designed so that the later stages plug in
without rewriting:

* `build_trip_features(pre, with_labels=False)` is already the inference-time feature call.
* `PreprocessedTrip.track.segment_id` marks where an HMM chain must be broken.
* `build_trip_map(..., predicted_arc_ids=...)` already draws a predicted route when one exists.
