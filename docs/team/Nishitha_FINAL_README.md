# Nishitha S: Integration lead, data pipeline, evaluation, inference, report (final evaluation)

**Branches:** `integration/final-shared` (round 1), `feature/inference-pipeline` (round 5),
`results/final-evidence` (round 7). You review and merge every PR.

## Your files
| File | Round | What it does |
|---|---|---|
| `src/common/config.py` (changed) | 1 | `get_eval_data_root` (`PATHSNAP_EVAL_DATA_ROOT`), results/artefact/processed dirs; partition guard unchanged |
| `src/common/io.py` (changed) | 1 | `load_trip_for_inference` (no `.route`), `load_reference_route`, `list_trip_dirs` |
| `src/common/graph.py` | 1 | `RoadGraph`: directed network distance between positions on arcs (bounded Dijkstra with cache) |
| `src/evaluation/metrics.py`, `route_eval.py` | 1 | Every metric definition: Top-1, macro F1, log loss, confusion, miss rate, route continuity, route precision/recall/F1 |
| `tests/conftest.py` (changed), `tests/test_evaluation.py` | 1 | Synthetic multi-trip fixtures and a tiny real bundle; metric/graph/loader tests |
| `requirements.txt`, `.gitignore`, `.env.example`, `README.md`, `docs/*` | 1 | Shared project files and documentation |
| `src/inference/artifacts.py`, `predictor.py` | 5 | Bundle export/load (hash and feature-list checks); label-free end-to-end `predict_trip` |
| `scripts/*` | 5 | Artefact export/validation, full pipeline, HTTP rehearsal, screenshots, report builder |
| `tests/test_inference.py` | 5 | Bundle round trip, prediction without labels, integrity checks |
| `results/`, `artifacts/`, `docs/technical_report/` | 7 | Evidence from the full run |

## Be ready to explain
* The definition of each metric and its denominator (Top-1 counts fixes without candidates as errors; continuity excludes pairs split by a gap).
* How the reference route is kept out of the prediction path (loader, `predict_trip` refuses a route, evaluator runs afterwards).
* How the pieces fit together (README architecture) and the merge order.
