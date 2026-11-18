# Nishanth N: Feature Engineering

**Branch:** `feature/feature-engineering` (needs `feature/preprocessing` merged into `main` first)

## Your responsibility in phase 1
Candidate road arcs per GPS fix, per-observation reference labels from the route, displacement/temporal/geographic/road-network features, the radius experiment, feature documentation.

## Files in this ZIP
| File | What it does |
|---|---|
| `src/features/spatial_index.py` | `ArcIndex`: samples arcs every ≤ 20 m into a KD-tree; exact arcs within a radius |
| `src/features/candidates.py` | `CandidateConfig`, `generate_candidates()` → `obs_idx, arc_id, distance_m, offset_frac, distance_rank` |
| `src/features/labels.py` | `align_track_to_route()`: monotonic dynamic-programming alignment of fixes to route positions |
| `src/features/features.py` | `observation_features`, `arc_attributes`, `candidate_features`, `build_trip_features()` + summary (miss rate, nearest-arc diagnostics) |
| `src/features/run_features.py` | CLI: feature tables per trip; `--sweep` radius experiment (train only) |
| `tests/test_features.py` | 6 tests (index vs brute force, candidates, alignment, motion features, full table, CLI) |
| `docs/team/Nishanth_README.md` | this file |

## Install / run
```bash
pip install -r requirements.txt
```
```bash
python -m pytest -q tests/test_features.py
```
```bash
python -m src.features.run_features --partition train --sweep 25 50 75 100
```
```bash
python -m src.features.run_features --partition train
```
```bash
python -m src.features.run_features --partition student_test
```
Dependencies: numpy, pandas, scipy (`cKDTree`).

## Inputs → outputs
Input: raw trips (through `preprocess_trip`). Output `outputs/features/<partition>/`:
`<trip_id>_features.csv.gz` (one row per fix × candidate), `feature_summary.csv`; with `--sweep`:
`radius_sweep.csv`, `radius_sweep_per_trip.csv`. Train ≈ 1.23 M rows, ~56 MB, ≈ 60 s; sweep ≈ 80 s.

## Shared interfaces
* **Uses:** `src.preprocessing.pipeline.preprocess_trip/PreprocessedTrip` (track columns `obs_idx, x, y, t, t_rel, dt_prev, segment_id, is_segment_start, is_speed_outlier`), `src.common.geo`.
* **Provides:** feature-table columns and `build_trip_features(pre, config, with_labels)` (`docs/INTERFACES.md` §3). The future model code will call `with_labels=False` at inference.
* `radius_sweep.csv` columns are read by Niranjan's `plot_radius_sweep`.

## Do not modify
`src/common/*`, `src/preprocessing/*`, `src/data_exploration/*`, `src/visualization/*`, `tests/conftest.py`.

## Git
Round 2a in `docs/GIT_WORKFLOW.md`.

## Understand before the evaluation
* Why querying the KD-tree with r + spacing/2 guarantees no arc is missed.
* The recurrence `cost[i, j] = d(i, j) + min_{k ≤ j} cost[i-1, k]` and the back-pointer trick.
* Why labels (`is_reference`, `ref_distance_m`) must never be features.
* Findings: 50 m radius → 1.7 % miss rate (excluding 2 offset trips) on train, 0.26 % on student_test. The nearest arc is the opposite-direction twin of the reference arc for 29 % of train fixes. Median heading difference is 13° for the reference arc vs 101° for the others.
