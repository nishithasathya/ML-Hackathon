# Neha: Preprocessing

**Branch:** `feature/preprocessing`. Merge this one early: Nishanth and Niranjan import your code.

## Your responsibility in phase 1
GPS data loading into a clean, time-ordered, projected track; missing/invalid value handling; timestamp processing; coordinate validation; gap segmentation; spike flagging.

Note: the low-level file readers (`src/common/io.py`) are in the shared integration package so that all four modules can load data from day one. Your module builds the **cleaned** loading path (`preprocess_trip`) on top of them.

## Files in this ZIP
| File | What it does |
|---|---|
| `src/preprocessing/cleaning.py` | `track_to_frame`, `drop_invalid_coordinates`, `enforce_time_order`, `add_projected_coordinates`, `add_time_deltas`, `assign_gap_segments`, `flag_speed_outliers` |
| `src/preprocessing/pipeline.py` | `PreprocessConfig`, `PreprocessedTrip`, `preprocess_trip()` |
| `src/preprocessing/run_preprocessing.py` | CLI: cleans a partition and writes CSVs + report + config |
| `tests/test_preprocessing.py` | 5 tests (invalid coords, ordering/dedup, full pipeline on synthetic trip, drop option, CLI) |
| `docs/team/Neha_README.md` | this file |

## Install / run
```bash
pip install -r requirements.txt
```
```bash
python -m pytest -q tests/test_preprocessing.py
```
```bash
python -m src.preprocessing.run_preprocessing --partition train
```
```bash
python -m src.preprocessing.run_preprocessing --partition student_test
```
Options: `--gap-threshold 30`, `--max-speed 70`, `--drop-speed-outliers`, `--trips 00000010 ...`.
Dependencies: numpy, pandas.

## Inputs → outputs
Input: raw trip files. Output `outputs/preprocessed/<partition>/`: `<trip_id>.csv` (clean track),
`preprocessing_report.csv`, `preprocess_config.json`. Run time ≈ 10 s for both partitions.

## Shared interfaces
* **Provides (contract, `docs/INTERFACES.md` §2):** `preprocess_trip(raw, config) -> PreprocessedTrip` and the column names of `PreprocessedTrip.track`. Features and visualisation depend on them, so don't rename columns.
* **Uses:** `src.common.io.RawTrip/iter_trips`, `src.common.geo.LocalProjection`.

## Do not modify
`src/common/*`, `src/data_exploration/*`, `src/features/*`, `src/visualization/*`, `tests/conftest.py`.

## Git
Round 1a in `docs/GIT_WORKFLOW.md`.

## Understand before the evaluation
* Why `obs_idx` is never renumbered.
* Why a stable (mergesort) sort is used.
* Why a spike needs high speed **both** in and out (a single real jump would otherwise flag two points).
* Why arcs are left untouched even though some are duplicates.
* Findings on our data: 0 rows dropped; 133 gaps > 30 s in train; 68 spikes (67 in trip 00000013).
