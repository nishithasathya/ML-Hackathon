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
