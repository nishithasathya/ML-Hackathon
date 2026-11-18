# PathSnap: First-Phase Technical Documentation

UE24CS352A Machine Learning, PES University. Team: Nishitha S, Neha, Nishanth N, Niranjan.

> Every number in this document was produced by the code in this repository
> (commands are given next to each result). Partition used is stated for each
> analysis. Re-running the commands must reproduce the numbers. If it doesn't,
> the document is wrong and must be corrected. Nothing here is a model result:
> **no model has been trained yet.**

Status labels used below: **Implemented & tested** (code + unit tests + run on real data),
**Implemented** (run on real data, no dedicated unit test), **Partial**, **Planned**.

---

## 1. Project overview

PathSnap reconstructs the road route a vehicle drove from noisy GPS fixes. The
final system will (a) generate candidate road segments for each GPS fix,
(b) score them with a trained classifier, (c) decode a coherent route with an
HMM/Viterbi decoder, and (d) serve it through a FastAPI backend to an evaluator UI
on a second laptop. **This milestone covers the foundation: understanding,
validating and preparing the data, and producing model-ready candidate/feature
tables.**

## 2. Problem definition (as we currently frame it)

For every GPS observation *i* of a trip, choose the road arc (directed segment)
of that trip's network that the vehicle was on. We plan to treat this as
**candidate classification**: for every pair (observation *i*, candidate arc *a*)
predict whether *a* is the reference arc, then let an HMM enforce connectivity.
Phase 1 produced the candidate pairs, their features and their labels.

## 3. Dataset description (verified against the files)

Command: `python -m src.data_exploration.run_eda` → `outputs/eda/`

| Partition | Trips | GPS observations | Nodes (sum) | Arcs (sum) | Route arcs (sum) | Size |
|---|---|---|---|---|---|---|
| train | 80 | 208,417 | 1,933,485 | 3,876,904 | 31,336 | 89.7 MB |
| student_test | 10 | 19,160 | 90,689 | 185,041 | 3,477 | 4.4 MB |

Evaluator-testing trips were **not** supplied to us and our loader refuses any partition other than `train`/`student_test`.

File formats as documented were confirmed for all 90 trips (column counts are
uniform: track 3, nodes 2, arcs 2, route 1). Additional facts we found:

* **Every trip has its own road network.** No two trips share an identical `.nodes` file, so node and arc ids are local to a trip.
* **The trips are spread worldwide.** Track centres range from longitude −100.35 to 137.22 and latitude −34.62 to 61.67 (Europe, Russia, the Americas, Asia, South Africa). Figure: `trip_locations.png`.
* **All `.track` files use Windows CRLF line endings**; the other files use LF. Our reader treats `\r` as whitespace.
* **Timestamps are relative seconds**, starting at 1 for 74 trips and at a later value for 16 trips (maximum start 693 s, trip 00000006).
* **Fractional timestamps** occur in 3 train trips (00000002, 00000071, 00000097; 9,659 fixes in total). Inspection shows small offsets (e.g. 2.003, 7.2), not a different sampling rate.

## 4. Dataset exploration findings

### 4.1 Missing and invalid values (both partitions)

`validation.py` checks NaN/out-of-range/(0,0) coordinates, missing timestamps,
non-increasing and duplicate timestamps, arcs pointing to missing nodes, and
route ids outside the arc table.

* **No missing values, NaNs or invalid coordinates in any file. No timestamp ever decreases or repeats.**
* **No `error`-severity issue** (no arc refers to a missing node and no route id is out of range).
* Informational/warning issues found (181 rows, 78 trips; figure `validation_issues.png`):

| Issue | Trips | Occurrences |
|---|---|---|
| Duplicate arcs (same from,to under several arc ids) | 60 | 1,903 |
| Distinct nodes with identical coordinates | 37 | 92 |
| Time gaps > 30 s | 30 | 135 |
| Track does not start at t = 1 | 16 | 16 |
| **Reference route discontinuities** (`to_node` ≠ next `from_node`) | 11 | 14 |
| Route repeats an arc | 10 | 322 |
| Self-loop arcs | 9 | 24 |
| Consecutive identical positions | 5 | 6 |
| Fractional timestamps | 3 | 9,659 |

### 4.2 Sampling and gaps

Figure `sampling_intervals.png`; counts in `outputs/figures/sampling_interval_counts.csv`.

| Interval | train | student_test |
|---|---|---|
| 1 s | 201,064 | 18,886 |
| 2 s | 4,336 | 126 |
| 3-5 s | 1,721 | 94 |
| 6-30 s | 1,083 | 42 |
| 31-300 s | 117 | 2 |
| > 300 s | 16 | 0 |

The data is close to 1 Hz (96.5 % of train intervals and 98.6 % of student-test
intervals are exactly 1 s). Eleven train trips contain a gap over 5 minutes;
the extreme cases are 00000090 (153,064 s ≈ 42.5 h) and 00000078 (52,810 s).
**student_test has no gap above 210 s.** So the train set contains harder
gap situations than the student-test set.

### 4.3 Trip size and speed (per-trip ranges, train / student_test)

| Statistic | train min / median / max | student_test min / median / max |
|---|---|---|
| GPS observations | 361 / 1,529 / 19,313 | 262 / 1,420 / 3,958 |
| GPS path length (km) | 6.3 / 24.1 / 169.3 | 5.5 / 20.9 / 71.7 |
| 95th-percentile speed (m/s) | 14.4 / 25.8 / 65.8 | 18.9 / 28.5 / 35.9 |
| Route length (arcs) | 59 / 280 / 2,266 | 95 / 250 / 1,323 |

Implied speeds above 70 m/s (252 km/h) between consecutive fixes occur in 11
trips. Trip 00000013 alone has 95 such steps, which indicates repeated position jumps.

### 4.4 Road-network structure

Figure `network_size.png`.

| Statistic | train min / median / max | student_test min / median / max |
|---|---|---|
| Nodes | 299 / 9,608 / 155,003 | 1,770 / 6,597 / 26,635 |
| Arcs | 649 / 19,527 / 307,392 | 3,577 / 13,145 / 56,027 |
| Weakly connected components | 1 / 20 / 421 | 5 / 11 / 100 |
| Strongly connected components | 2 / 127 / 1,942 | 6 / 54 / 415 |
| Largest weak component (share of nodes) | 0.69 / 0.98 / 1.00 | 0.87 / 0.97 / 0.99 |
| Arcs whose reverse arc exists | 0.35 / 0.95 / 1.00 | 0.86 / 0.93 / 1.00 |
| Median arc length (m) | 6.7 / 24.6 / 84.9 | 15.8 / 31.9 / 53.2 |

* **No isolated nodes** in any network (every node is used by an arc).
* Networks are **not fully connected**: small disconnected fragments exist, and
  the many strongly connected components (one-way streets, dead ends) mean that
  some arc-to-arc transitions are impossible. This will matter for HMM transitions.
* **Most roads are stored twice** (one arc per direction, identical geometry).
  This creates an ambiguity that pure distance cannot resolve (§6.4).
* Some arcs are long (max 9.3 km in trip 00000028), so indexing arcs by their
  endpoints alone would miss them (§6.1).

### 4.5 GPS noise relative to the reference route

For every fix we computed the distance to the nearest reference-route arc
(`gps_to_route_distance`). Figure `gps_route_distance.png`.

* Typical trip: **median 4.5 m**, 95th percentile 14.2 m (train medians across trips).
* student_test: per-trip medians 4.2-6.8 m and p95 11.7-26.1 m, so all 10 trips are "clean".
* **Two train trips are outliers: 00000054 (median 138.5 m) and 00000033 (122.4 m).**
  In both, the first part of the trip lies within a few metres of the route,
  then the track runs ~120-160 m away from the route for long stretches, mainly
  displaced north-south. The route and the track still cover the same area. We
  have **not** determined the cause (shifted GPS vs. erroneous reference route).
  We flag these trips rather than "fix" them. Map: `train_00000033.html`.

### 4.6 Reference-route limitations

* 11 routes (14 places) are not continuous: consecutive route arcs do not share a node.
* 10 routes traverse the same arc more than once (loops, returns).
* The route is a sequence of arcs **without timestamps**, so which GPS fix belongs to which arc must be inferred (§6.3).

Following the Dataset Exploration guide (§6), we treat routes as reference information, not verified truth.

### 4.7 Choice of coordinate system

Degrees are not distances (at 55° N a degree of longitude is only ~57 % of a
degree of latitude). We use the haversine formula for reported lengths and a
per-trip **local equirectangular projection** (metres) for geometry. We measured
its error instead of assuming it: the maximum relative error of a projected step
length versus haversine is 0.78 % over all trips (median trip ≈ 0.1 %), which is
acceptable for metre-level matching.

## 5. Data preprocessing (Implemented & tested)

Command: `python -m src.preprocessing.run_preprocessing --partition train`.
Configuration is saved next to the outputs.

| Step | Rule | Result on train / student_test |
|---|---|---|
| Invalid coordinates | drop NaN, out of range, (0,0) | 0 / 0 dropped |
| Time order | drop missing t, stable sort, drop duplicate t | 0 / 0 changed |
| Projection | per-trip local metres; nodes use the same projection | n/a |
| Gap segmentation | new segment when Δt > 30 s | 133 gaps → 213 segments / 2 gaps → 12 segments |
| Speed spikes | flag a fix when speed in AND out > 70 m/s (same segment) | 68 flagged (67 in trip 00000013, 1 in 00000092) / 0 |

Design decisions:
* **Original row ids are preserved (`obs_idx`)**, because later phases must output one prediction per observation.
* **Spikes are flagged, not deleted** (optional `--drop-speed-outliers`), so later phases can decide.
* **Arcs are not de-duplicated** even though duplicates exist: arc ids must stay aligned with `.route`.
* Nothing is fitted to data in preprocessing, so it cannot leak test information.

## 6. Feature engineering (Implemented & tested, initial set)

Commands: `python -m src.features.run_features --partition train|student_test` and `--sweep 25 50 75 100`.

### 6.1 Candidate generation

Each arc is sampled every ≤ 20 m and the samples are stored in a KD-tree. A
query radius of *r* + 10 m on the samples guarantees that no arc within *r* is
missed, and the exact point-to-segment distance then filters the result. A unit
test checks this against brute force. For each fix we keep up to 10 nearest arcs within *r*.

### 6.2 Choosing the radius (train only)

Figure `radius_sweep.png`, file `outputs/features/train/radius_sweep.csv`. Miss rate = share of fixes whose reference arc is not in the candidate set.

| Radius | Miss rate, all 80 train trips | Miss rate, 78 trips (excl. 00000033, 00000054) | Candidates per fix (78 trips) |
|---|---|---|---|
| 25 m | 13.4 % | 3.4 % | 4.3 |
| **50 m** | 11.7 % | **1.7 %** | 6.4 |
| 75 m | 11.3 % | 1.2 % | 7.4 |
| 100 m | 10.9 % | 1.0 % | 8.0 |

The two offset trips cause about 88 % of the misses at 50 m. We chose **50 m**:
going from 25 m to 50 m halves the miss rate, but further increases give little
(−0.5 points) for one extra candidate per fix. Applied unchanged to student_test,
50 m gives a **0.26 % miss rate** with 6.9 candidates per fix.

### 6.3 Per-observation reference labels

Because the route has no timestamps, we align fixes to route positions with
dynamic programming. Route positions may only move forward in time, and the total
fix-to-arc distance is minimised. This resolves places where the route uses the
same street twice or in both directions. Output: `ref_arc_id`, `ref_distance_m`
per fix, giving `is_reference` per candidate. **These columns are labels and
diagnostics only, never model inputs.**

### 6.4 Features implemented

| Group | Features |
|---|---|
| Observation (motion) | `t_rel`, `dt_prev`, `dt_next`, `step_dist_m`, `speed_mps`, `speed_smooth_mps` (5-fix rolling median), `accel_mps2`, `heading_deg` (prev→next fix, NaN if < 2 m movement), `heading_change_deg`, `cum_dist_m`, `segment_id`, `is_segment_start`, `is_speed_outlier` |
| Arc (network) | `arc_length_m`, `arc_bearing_deg`, `arc_to_out_degree`, `arc_has_reverse` |
| Pair | `distance_m`, `distance_rank`, `dist_minus_nearest_m`, `offset_frac`, `n_candidates`, `heading_diff_deg` |

All motion features are computed **within** gap segments, so nothing is measured across a long gap.

Observations that motivated the features (train, 1,228,707 candidate rows):
* **Distance alone is not enough.** Choosing the nearest arc matches the reference for 49.3 % of fixes (train) and 53.6 % (student_test).
  In 28.9 % / 35.4 % of fixes the nearest arc is the **opposite-direction twin** of the reference arc, because the twin has identical geometry and ties on distance.
  (This "nearest arc" figure is a geometric heuristic, not a model.)
* **Heading separates the twins.** Median `heading_diff_deg` is 13.1° for reference candidates and 101.0° for the others (heading is undefined for 1.7 % of rows).
  Median distance is 4.9 m for reference candidates and 19.4 m for the others.

## 7. Visualisations generated (Implemented & tested)

Commands: `python -m src.visualization.run_visualizations figures` and `... maps --partition <p> --trips <ids>`.

| Output | Partition | Content |
|---|---|---|
| `maps/<partition>_<trip>.html` (Folium) | train 00000010, 00000033, 00000002; student_test 00000052 | Layers: nearby road network, reference route, raw GPS fixes, flagged spikes, gap starts, start/end markers; legend; layer control; fitted bounds |
| `demo/<partition>_<trip>_map.html` | any | Same, plus candidate arcs for 5 sampled fixes |
| `trip_<partition>_<trip>.png` | as above | Static GPS-vs-route plot in metres |
| `trip_locations.png` | both | Trip centres on lon/lat axes |
| `observations_per_trip.png`, `gps_route_distance.png`, `largest_gap_per_trip.png`, `network_size.png` | both | Per-trip distributions |
| `sampling_intervals.png` | both | Sampling-interval shares |
| `validation_issues.png` | both | Trips affected per issue type |
| `radius_sweep.png` | train | Miss rate vs radius |

The map has a "Predicted route" layer hook, which stays empty until a model exists.

## 8. Implementation architecture

```
raw files ──► src.common.io.load_trip ──► RawTrip
                  │
                  ├─► src.data_exploration (inventory, validate_trip, trip_statistics) ─► outputs/eda/*.csv
                  │
                  └─► src.preprocessing.preprocess_trip ─► PreprocessedTrip (clean track in metres + network)
                              │
                              ├─► src.features.build_trip_features ─► candidate/feature table (+ labels)
                              │       └ ArcIndex (KD-tree) → generate_candidates → features → align_track_to_route
                              │
                              └─► src.visualization (Folium map, static plots)
```

Shared code lives in `src/common` (paths, readers, geodesy). Module contracts are
in `docs/INTERFACES.md`. The test suite (27 tests) uses a synthetic 3×3 street
grid with a known spike, gap and duplicate timestamp, so it runs without the dataset.

## 9. Challenges encountered

1. **Per-trip networks of very different size** (649 to 307,392 arcs). This needed a spatial index; brute-force distance to every arc is infeasible.
2. **Long arcs** (up to 9.3 km), which rule out endpoint/midpoint indexing. Solved by sampling along arcs with a provable search margin.
3. **Route has no timestamps.** Labels need an alignment step; plain nearest-route-arc is ambiguous on two-way streets and repeated passes.
4. **Two-way streets** make distance-only matching ≈ 50 % correct; heading is required.
5. **Data-quality outliers**: two trips with a 120-160 m GPS-to-route offset, 11 discontinuous routes, multi-hour time gaps. They are flagged and documented, not silently removed.
6. **Global coordinates**: one global projection is impossible, so we use a per-trip local projection with measured error.

## 10. Current implementation status

| Component | Status |
|---|---|
| Raw-file readers + partition guard | Implemented & tested |
| Inventory, validation, statistics, data-quality notes | Implemented & tested |
| Preprocessing pipeline | Implemented & tested |
| Candidate generation (KD-tree) | Implemented & tested |
| Route-to-observation labels | Implemented & tested (synthetic); run on all 90 real trips |
| Initial feature set | Implemented & tested |
| Radius sweep | Implemented & tested; run on train |
| Folium maps + static figures | Implemented & tested |
| Group-aware CV split | Planned |
| Feature scaling / selection | Planned |
| Handling of offset trips (exclude/down-weight?) | Partial: identified, decision pending |
| Models, Bayesian optimisation, HMM/Viterbi, API, UI, deployment | Planned (see `DEFERRED_COMPONENTS.md`) |

## 11. Components deferred to subsequent evaluations

See [`DEFERRED_COMPONENTS.md`](DEFERRED_COMPONENTS.md).
