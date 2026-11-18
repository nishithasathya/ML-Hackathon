# AI-assistance disclosure and work log

## Disclosure

The Assessment and Plagiarism Policy (§5) and the Deliverables document (§8, rule 5)
require us to disclose AI assistance. **The initial first-phase code skeleton,
tests and documentation drafts were generated with the help of an AI coding
assistant (Claude Code)**, working from our supplied PDFs and dataset. The team
lead requested this.

What that means for us:
* Each member must read, run and be able to explain every line of their module before the evaluation (questions in `DEMO_CHECKLIST.md`).
* All numbers in the documentation come from running this code on our dataset. None were typed in by hand or invented.
* Any later change, experiment or result must be recorded below by the person who did it.
* Follow the event's specific AI-disclosure instructions if they ask for a different format.

## Log of what was actually run

| Date | Who / what | Command | Result |
|---|---|---|---|
| 2026-11-18 | AI-assisted setup run (Linux, Python 3.13, numpy 2.5, pandas 3.0, scipy 1.18, folium 1.0) | `python -m pytest -q` | 27 passed |
| 2026-11-18 | same | `python -m src.data_exploration.run_eda` | 90 trips, 181 validation rows, 0 errors |
| 2026-11-18 | same | `python -m src.preprocessing.run_preprocessing --partition train` / `student_test` | 0 rows dropped; 133 / 2 gaps; 68 / 0 spike flags |
| 2026-11-18 | same | `python -m src.features.run_features --partition train --sweep 25 50 75 100` | see `radius_sweep.csv`; 50 m chosen |
| 2026-11-18 | same | `python -m src.features.run_features --partition train` / `student_test` | miss rate 0.1174 / 0.0026 |
| 2026-11-18 | same | `python -m src.visualization.run_visualizations figures` and `maps` | 8 dataset figures, 4 maps + 4 static trip plots |
| 2026-11-18 | same | `python scripts/run_first_phase_demo.py` (train 00000010, student_test 00000052) | completed |
| | *(team: add your own runs here, on your own laptops)* | | |

## Changes made by team members after the initial version

| Date | Who | File(s) | What and why |
|---|---|---|---|
| | | | |

## Final evaluation (phase 2)

The phase-2 code, tests, documentation and report text were again drafted with an AI coding assistant
(Claude Code), at the integration lead's request, starting from our phase-1 code and the five official PDFs.
Every experiment listed below was actually executed. The report reads its numbers from the files these runs wrote.

| Date | Who / where | Command | Result (file) |
|---|---|---|---|
| 2026-11-18 | AI-assisted run on Nishitha's MacBook (macOS, Python 3.11.9, numpy 2.4.6, pandas 3.1.1, scikit-learn 1.9.1, optuna 5.1.0, 10 cores) | `python -m pytest -q` | all tests passed (see final count in the demo guide) |
| 2026-11-18 | same | `python -m src.models.train_compare` | 5 models × 5 trip-grouped folds + student test, about 25 min (`results/models/`) |
| 2026-11-18 | same | `python -m src.models.select_model` | Random Forest selected by the pre-set rule (`selected_model.json`) |
| 2026-11-18 | same | `python -m src.features.radius_study --model rf` | 25/50/75/100 m, about 20 min; 50 m kept by the tie rule (`results/candidates/`) |
| 2026-11-18 | same | `python -m src.optimization.bayes_opt --model rf --trials 25` | Optuna TPE study (`results/optimisation/`) |
| 2026-11-18 | same | `python -m src.hmm.run_hmm --model rf --probs tuned` | decoder grid + classifier vs HMM (`results/hmm/`) |
| 2026-11-18 | same | `scripts/build_final_artifacts.py`, `scripts/validate_artifacts.py` | `artifacts/` bundle loads and predicts |
| 2026-11-18 | same | `scripts/run_api_rehearsal.py` | API over HTTP via LAN IP, all 10 student-test trips (`results/api_rehearsal/`) |
| 2026-11-18 | same | `scripts/capture_screenshots.py`, `scripts/build_report.py` | real UI/API screenshots, PDF report |
| | *(team: add your own runs, especially the two-laptop rehearsal and the evaluator trips)* | | |

Known decisions made during the run (also stated in the report):
* The radius tie rule (0.5 points, the same margin as model selection) was added after the radius study's first run.
* A GPS-spike rule was added to the HMM after a unit test showed that a 300 m spike forced a false U-turn.
