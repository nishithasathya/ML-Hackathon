# Niranjan: Visualisation & Documentation

**Branch:** `feature/visualization-documentation`, then `docs/phase1-evidence` (after everything is merged)

## Your responsibility in phase 1
Folium maps (mandatory per Deliverables §6), GPS trajectory and road-network plots, EDA figures, environment setup instructions, first-phase technical documentation, demo checklist, and committing the generated evidence.

## Files in this ZIP
| File | What it does |
|---|---|
| `src/visualization/folium_maps.py` | `build_trip_map()` with layers (network near track, reference route, GPS fixes, spike flags, gap starts, candidates, predicted-route hook), legend, layer control; `nearby_arc_ids()` |
| `src/visualization/plots.py` | Static figures: trip GPS-vs-route, trip locations, distributions, network size, sampling intervals, validation issues, radius sweep |
| `src/visualization/run_visualizations.py` | CLI: `maps` and `figures` |
| `tests/test_visualization.py` | 5 tests |
| `docs/FIRST_PHASE_TECHNICAL_DOC.md` | first-phase technical documentation (real numbers from our runs) |
| `docs/SETUP.md` | environment setup |
| `docs/DEMO_CHECKLIST.md` | evaluation-day checklist and viva questions |
| `docs/team/Niranjan_README.md` | this file |

## Install / run
```bash
pip install -r requirements.txt
```
```bash
python -m pytest -q tests/test_visualization.py
```
```bash
python -m src.visualization.run_visualizations maps --partition train --trips 00000010 00000033
```
```bash
python -m src.visualization.run_visualizations figures
```
`figures` needs `outputs/eda/` (run Nishitha's `run_eda` first) and `outputs/features/train/radius_sweep.csv`
(Nishanth's `--sweep`). If either is missing, that figure is **skipped with a message, never faked**.
Dependencies: folium (pulls in branca), matplotlib, numpy, pandas.

## Inputs → outputs
Output: `outputs/maps/<partition>_<trip>.html`, `outputs/figures/*.png`, `outputs/figures/sampling_interval_counts.csv`.
Maps need internet in the browser (Leaflet and tiles load from CDNs).

## Shared interfaces
* **Uses:** `src.common.io`, `src.preprocessing.pipeline.preprocess_trip` (flags `is_speed_outlier`, `is_segment_start`), CSV columns from `docs/INTERFACES.md` §4.
* Must **not** import `src.features`. Candidate tables arrive as plain DataFrames (`obs_idx, arc_id, distance_m, distance_rank`).
* **Provides:** `build_trip_map(..., predicted_arc_ids=...)`, which the future API/UI phase will use.
* Always read project CSVs with `dtype={"trip_id": str}` (`plots.read_csv_keep_ids`).

## Do not modify
`src/common/*`, `src/data_exploration/*`, `src/preprocessing/*`, `src/features/*`, `README.md`, the integration docs.
If the technical document needs a number, take it from an output file. Never type in an estimate.

## Git
Rounds 2b and 3 in `docs/GIT_WORKFLOW.md`.

## Understand before the evaluation
* Folium coordinates are `[lat, lon]` (the reverse of our arrays).
* Why only arcs near the track are drawn (networks up to 307,392 arcs).
* Which partition every figure in the technical doc uses.
