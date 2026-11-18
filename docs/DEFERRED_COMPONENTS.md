# Components intentionally deferred to later evaluations

None of the following exist in the repository yet. **Do not claim them during the
first-phase evaluation.** The "hook" column shows what phase 1 already prepares.

| # | Component | Official source | Phase-1 hook already in place |
|---|---|---|---|
| 1 | Group-aware cross-validation over the 80 training trips (folds split by trip) | Models §4, §7 | `trip_id` column in every feature table |
| 2 | Training ≥ 5 models from different families, with configurations, seeds, training/inference times logged | Models §2-4; Assessment §3 | Feature tables in `outputs/features/<partition>/` |
| 3 | Metrics: accuracy, precision, recall, macro-F1, log loss, observation-level Top-1, candidate miss rate, route continuity, sequence-level performance, confusion matrices | Models §5; Deliverables §3 | Candidate miss rate is already computed (`feature_summary.csv`) |
| 4 | Model comparison table and comparison plots (CV macro-F1, student-test Top-1/macro-F1, log loss) | Deliverables §3.1, §3.3 | `src/visualization/plots.py` style helpers |
| 5 | Bayesian hyperparameter optimisation (search space, objective, trials, tuned vs untuned) | Models §7; Deliverables §4 | n/a |
| 6 | HMM/Viterbi route reconstruction: hidden states, emissions from classifier, transitions from network connectivity, handling of gaps and missing candidates, classifier-only vs HMM comparison | Models §8-9; Deliverables §5 | Candidates = future hidden states; `segment_id` = where to break chains; arc table for connectivity |
| 7 | Export of model + preprocessing + HMM configuration artefacts | Models §5 | `PreprocessConfig` / `CandidateConfig` already serialisable |
| 8 | FastAPI backend (Laptop 1): receives trip id, runs pipeline, computes accuracy server-side, logs requests | Main Guidelines §3; Models §6 | `build_trip_features(pre, with_labels=False)` is the inference-time call |
| 9 | Evaluator UI (Laptop 2): list 10 trips, call API, show route and results | Main Guidelines §3; Models §6 | `build_trip_map(..., predicted_arc_ids=...)` already draws a predicted route |
| 10 | Two-laptop deployment, configuration and run instructions | Deliverables §1, §7 | README structure |
| 11 | Consolidated final technical report (PDF) | Deliverables §2-8 | `docs/FIRST_PHASE_TECHNICAL_DOC.md` follows the §2.2 headings |

## Open questions to settle at the start of phase 2

* How should the two offset trips (00000033, 00000054) be treated in training: excluded, down-weighted, or kept? Decide using training data only, and document it.
* Should spike-flagged fixes be predicted normally or smoothed?
* Which primary metric will be used for model selection (Models §5 requires a justified choice)?
* Feature scaling (needed for KNN/SVM/MLP) must be fitted on training folds only.
