# Shared interfaces

These are the contracts between the four modules. **Changing any of them needs a
heads-up to the whole team and to the integration lead**, because other people's
code depends on them.

## Import rule

Always import with the full package path from the repository root, and run
code with `python -m ...` from the repository root:

```python
from src.common.io import load_trip            # correct
from common.io import load_trip                # wrong - breaks for everyone else
```

Dependency direction (no cycles allowed):

```
src.common  <-  src.data_exploration
src.common  <-  src.preprocessing  <-  src.features
                                   <-  src.visualization
scripts/run_first_phase_demo.py imports all of them
```

`data_exploration` must not import from `preprocessing`/`features`/`visualization`.
`visualization` must not import from `features` (it reads candidate tables as plain DataFrames).

## 1. `src.common` (integration lead; do not edit on feature branches)

| Name | Signature / meaning |
|---|---|
| `config.ALLOWED_PARTITIONS` | `("train", "student_test")`; anything else raises `ValueError` |
| `config.get_data_root(override=None) -> Path` | `--data-root` > `$PATHSNAP_DATA_ROOT` > `data/raw` |
| `config.get_output_dir(*parts, root=None, create=True) -> Path` | `outputs/<parts>` |
| `io.RawTrip` | dataclass: `trip_id: str`, `partition: str`, `track (n,3) float [lon, lat, t]`, `nodes (m,2) float [lon, lat]`, `arcs (k,2) int [from, to]`, `route (r,) int`, `directory` |
| `io.list_trips(partition, data_root=None) -> list[str]` | sorted trip ids such as `"00000010"` |
| `io.load_trip(partition, trip_id, data_root=None) -> RawTrip` | |
| `io.iter_trips(partition, data_root=None, trip_ids=None)` | generator of `RawTrip` |
| `io.read_track/read_nodes/read_arcs/read_route(path)` | raw readers; arcs/route raise `DataFormatError` if not integer |
| `geo.haversine_m(lon1, lat1, lon2, lat2)` | metres |
| `geo.LocalProjection(lon0, lat0)` / `.from_points(lon, lat)` | `.forward(lon, lat) -> (x, y)` metres; `.inverse(x, y)` |
| `geo.point_segment_distance(px, py, ax, ay, bx, by) -> (dist, t)` | broadcasting; `t` in [0,1] |
| `geo.bearing_from_xy(dx, dy)`, `geo.angle_diff_deg(a, b)` | compass degrees; smallest difference in [0,180] |

**Trip ids are strings with leading zeros.** When reading any CSV produced by this
project use `pd.read_csv(path, dtype={"trip_id": str})`.

## 2. `src.preprocessing` (Neha)

```python
from src.preprocessing.pipeline import PreprocessConfig, PreprocessedTrip, preprocess_trip
pre = preprocess_trip(raw_trip, PreprocessConfig())   # gap_threshold_s=30, max_speed_mps=70, drop_speed_outliers=False
```

`PreprocessedTrip` fields: `trip_id, partition, track (DataFrame), nodes_lonlat, nodes_xy, arcs, route, projection, report`.

`pre.track` columns (contract):

| column | meaning |
|---|---|
| `obs_idx` | original row number in the .track file (never renumbered) |
| `lon`, `lat`, `t` | raw values |
| `x`, `y` | metres in `pre.projection` (same projection as `nodes_xy`) |
| `t_rel` | seconds since the first kept fix |
| `dt_prev` | seconds since previous kept fix (NaN on first row) |
| `is_segment_start`, `segment_id` | segment changes where `dt_prev > gap_threshold_s` |
| `speed_in_mps`, `is_speed_outlier` | spike flag (speed in AND out above threshold) |

## 3. `src.features` (Nishanth)

```python
from src.features.candidates import CandidateConfig          # radius_m=50, max_candidates=10, sample_spacing_m=20
from src.features.features import build_trip_features
table, summary = build_trip_features(pre, CandidateConfig(), with_labels=True)
```

Candidate table columns: `obs_idx, arc_id, distance_m, offset_frac, distance_rank`.
Feature table adds `trip_id`, observation features (`t_rel, dt_prev, dt_next, step_dist_m, speed_mps,
speed_smooth_mps, accel_mps2, heading_deg, heading_change_deg, cum_dist_m, segment_id, is_segment_start,
is_speed_outlier`), arc attributes (`arc_length_m, arc_bearing_deg, arc_to_out_degree, arc_has_reverse`), pair
features (`n_candidates, dist_minus_nearest_m, heading_diff_deg`) and, with labels, `ref_distance_m, is_reference`.

**`is_reference` and `ref_distance_m` are targets and diagnostics. Never use them as model inputs.**

`labels.align_track_to_route(track, nodes_xy, arcs, route)` returns `obs_idx, ref_route_pos, ref_arc_id, ref_distance_m`.

## 4. `src.data_exploration` output files (Nishitha), read by visualisation

`outputs/eda/trip_statistics.csv`: one row per trip. Columns used by `src/visualization/plots.py`:
`partition, trip_id, center_lon, center_lat, n_observations, n_nodes, n_arcs, gps_route_dist_median_m, max_gap_s`.

`outputs/eda/validation_issues.csv`: `partition, trip_id, file, severity, code, count, message`.

`outputs/features/train/radius_sweep.csv` (from features): `radius_m, n_trips, n_observations,
candidate_miss_rate, frac_obs_without_candidates, mean_candidates_per_obs, subset`.

## 5. `src.visualization` (Niranjan)

```python
from src.visualization.folium_maps import build_trip_map, save_trip_map
fmap = build_trip_map(raw, track=pre.track, candidates=None, predicted_arc_ids=None)
```
`predicted_arc_ids` is the hook for later phases (classifier/HMM output).

## Files nobody except the owner should modify

| Path | Owner |
|---|---|
| `src/common/*`, `scripts/*`, `tests/conftest.py`, `tests/test_common.py`, `requirements.txt`, `README.md`, `.gitignore`, `docs/INTERFACES.md`, `docs/INTEGRATION.md`, `docs/GIT_WORKFLOW.md`, `docs/FIRST_PHASE_SCOPE.md`, `docs/DEFERRED_COMPONENTS.md`, `docs/AI_ASSISTANCE_AND_WORK_LOG.md` | Nishitha (integration) |
| `src/data_exploration/*`, `tests/test_data_exploration.py` | Nishitha |
| `src/preprocessing/*`, `tests/test_preprocessing.py` | Neha |
| `src/features/*`, `tests/test_features.py` | Nishanth |
| `src/visualization/*`, `tests/test_visualization.py`, `docs/FIRST_PHASE_TECHNICAL_DOC.md`, `docs/SETUP.md`, `docs/DEMO_CHECKLIST.md`, `docs/evidence/*` | Niranjan |

Need a new dependency? Ask the integration lead to add it to `requirements.txt`; don't add it on your branch.
