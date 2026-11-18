# Nishitha S: Data Exploration (+ Integration lead)

**Branch:** `feature/data-exploration` (the shared skeleton goes first on `setup/project-skeleton`, from the separate *Integration package*)

## Your responsibility in phase 1
Dataset inspection, file inventory, validation, per-trip statistics, data-quality notes; integrating and merging everyone's work.

## Files in this ZIP (copy them to the same paths in the repo)
| File | What it does |
|---|---|
| `src/data_exploration/inventory.py` | Reads raw text: file existence, size, line count, column counts, CRLF detection |
| `src/data_exploration/validation.py` | Rule-based checks (`ValidationIssue`: severity, code, count) for track/nodes/arcs/route |
| `src/data_exploration/statistics.py` | Track stats (gaps, speeds, extent, projection error), network stats (components, reciprocity, duplicates), route stats, GPS-to-route distance |
| `src/data_exploration/run_eda.py` | CLI: writes all EDA CSVs and the generated `data_quality_notes.md` |
| `tests/test_data_exploration.py` | 4 tests on the synthetic trip |
| `docs/team/Nishitha_README.md` | this file |

`src/data_exploration/__init__.py` comes from the integration package, not from this ZIP.

## Install
```bash
pip install -r requirements.txt
```
Dependencies used: numpy, pandas, scipy (`scipy.sparse.csgraph.connected_components`).

## Run
```bash
python -m pytest -q tests/test_data_exploration.py
```
```bash
python -m src.data_exploration.run_eda
```
Options: `--partitions train`, `--data-root <folder>`, `--output-dir <folder>`.

## Inputs → outputs
Input: `data/raw/{train,student_test}/<id>/<id>.{track,nodes,arcs,route}`.
Output `outputs/eda/`: `inventory.csv`, `inventory_summary.csv`, `validation_issues.csv`,
`trip_statistics.csv`, `partition_summary.csv`, `anomalies.csv`, `data_quality_notes.md`. Run time ≈ 25 s.

## Shared interfaces
* **Uses:** `src.common.io.iter_trips/list_trips/trip_file/RawTrip`, `src.common.geo.LocalProjection/haversine_m/point_segment_distance`.
* **Provides:** `trip_statistics.csv` and `validation_issues.csv` column names, read by Niranjan's `plots.py` (`docs/INTERFACES.md` §4). Don't rename those columns.
* Must **not** import preprocessing, features or visualisation (keeps your branch independent).

## Do not modify
`src/preprocessing/*`, `src/features/*`, `src/visualization/*` and their tests. As integration lead you own `src/common/*`, but changing a signature there needs the team's agreement.

## Git
Rounds 0 and 1b in `docs/GIT_WORKFLOW.md`. You also merge everyone's PRs in the documented order. Ask a teammate to review your own PR.

## Understand before the evaluation
* Weak vs strong connected components, and why one-way streets create many strong components.
* Why `gps_to_route_distance` is chunked (memory = chunk × route length).
* The anomaly rules in `run_eda.py` are reporting thresholds, not model parameters.
