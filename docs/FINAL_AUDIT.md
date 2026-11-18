# Final-evaluation audit of the first-phase implementation

Date: 18 Nov 2026. Input: `PathSnap_First_Evaluation_Integrated.zip` (identical, file for file, to the
four member ZIPs + integration package), the five official PDFs, and `Student folder.zip`.

## 1. What already exists and is tested (phase 1)

| Module | Files | Tests | Verdict |
|---|---|---|---|
| Shared config / IO / geodesy | `src/common/{config,io,geo}.py` | `test_common.py` (6) | Correct. File formats verified on 90 trips. Partition guard works. |
| EDA | `src/data_exploration/*` | `test_data_exploration.py` (4) | Correct; outputs reproduced on macOS. |
| Preprocessing | `src/preprocessing/*` | `test_preprocessing.py` (6) | Correct. Nothing fitted, so no leakage. |
| Candidates + features + labels | `src/features/*` | `test_features.py` (7) | Correct. KD-tree search proven complete by a brute-force test. |
| Visualisation | `src/visualization/*` | `test_visualization.py` (4) | Correct; "predicted route" layer hook already exists. |

All 27 tests pass (macOS, Python 3.11.9).

## 2. Problems found in the phase-1 code

| # | Problem | Evidence | Fix in the final version |
|---|---|---|---|
| P1 | `trip_id` is written to CSV and read back as an integer (`00000052` → `52`). | `pd.read_csv(..._features.csv.gz).dtypes` shows `trip_id int64`. | The training dataset is built in-process from `build_trip_features` (string ids). Every CSV reader uses `dtype={"trip_id": str}`. |
| P2 | Duplicate arcs (same from/to node pair under two arc ids, 1,903 cases in 60 trips) are geometrically identical, but only the id that appears in `.route` was labelled positive. That trains the model to separate two identical inputs. | `validation_issues.csv` DUPLICATE_ARC. | `is_reference` now treats arcs with the same (from, to) node pair as equivalent. Top-1 and all metrics use the same equivalence. |
| P3 | `load_trip` requires the `.route` file, so inference could not run on a trip without labels, and nothing kept labels away from the model. | `src/common/io.py`. | New `load_trip_for_inference` reads only `.track/.nodes/.arcs`. Labels are loaded separately by the server-side evaluator. |
| P4 | The partition guard blocks every non-development partition, including the evaluator trips the API must serve. | `config.check_partition`. | Guard kept for training/analysis code. The API reads evaluator trips through a separate, explicitly configured directory (`PATHSNAP_EVAL_DATA_ROOT`). |
| P5 | Features carry no temporal context: a candidate is scored only from its own fix. | Feature list in `features.py`. | Added inference-safe context features: distance of the previous/next fix to the same arc, and motion along the arc direction. |
| P6 | Radius 50 m was chosen from train only, by eye. | `radius_sweep.csv`. | Re-evaluated with the trained model: student-test Top-1 vs radius (§ results/candidates). |
| P7 | `t_rel`, `cum_dist_m`, `segment_id` are trip-progress columns. They cannot generalise and would let a model memorise trips. | `features.py`. | Excluded from the model feature list (kept in the table for diagnostics). |
| P8 | `ref_distance_m` is a label-derived column inside the feature table. | `features.py`. | Explicitly listed in `LABEL_COLUMNS` and excluded from model input; a test asserts it. |

## 3. Official requirements still missing (Main Guidelines, Models, Deliverables, Assessment)

| Requirement | Source | Status before | Where it is now |
|---|---|---|---|
| ≥ 5 trained models, different families | Models §2, §4 | missing | `src/models/` |
| Group-aware CV on the 80 training trips | Models §4, §7 | missing | `src/models/cv.py` (GroupKFold by trip) |
| Metrics: accuracy, precision, recall, F1, macro F1, confusion matrix, log loss, Top-1, miss rate, route continuity, sequence-level, timings | Models §5 | only miss rate | `src/evaluation/metrics.py` |
| Comparison table + plots | Deliverables §3 | missing | `results/models/`, `results/figures/` |
| Bayesian optimisation (search space, trials, tuned vs untuned) | Models §7, Deliverables §4 | missing | `src/optimization/` (Optuna TPE) |
| HMM + Viterbi, classifier-only vs HMM | Models §8-9, Deliverables §5 | missing | `src/hmm/` |
| Exported artefacts loadable without retraining | Models §5 table | missing | `artifacts/`, `src/inference/` |
| FastAPI backend (Laptop 1), request logs | Main §3, Models §6 | missing | `src/api/` |
| Evaluator UI (Laptop 2) | Main §3 | missing | `frontend/` (Streamlit) |
| Two-laptop deployment instructions | Deliverables §1, §7 | missing | `deployment/` |
| Folium map with predicted route | Deliverables §6 | hook only | `src/visualization/folium_maps.py` + UI map |
| Consolidated PDF report | Deliverables §2-8 | missing | `docs/technical_report/` |

## 4. Interfaces that must be preserved

`load_trip`, `iter_trips`, `list_trips`, `RawTrip`, `preprocess_trip`, `PreprocessedTrip`, `PreprocessConfig`,
`ArcIndex`, `CandidateConfig`, `generate_candidates`, `build_trip_features(pre, config, with_labels, index, labels)`,
`build_trip_map(...)`, the column names in `docs/INTERFACES.md`. All are kept; only additions are made.

## 5. Files modified vs created

Modified (owner in brackets): `src/common/config.py`, `src/common/io.py` (Nishitha); `src/features/features.py`,
`src/features/candidates.py` (Nishanth); `src/visualization/folium_maps.py`, `src/visualization/plots.py` (Niranjan);
`src/preprocessing/pipeline.py` (Neha); `requirements.txt`, `.gitignore`, `README.md` (Nishitha).

Created: `src/common/graph.py`, `src/evaluation/`, `src/inference/` (Nishitha); `src/models/` (Neha);
`src/hmm/`, `src/optimization/` (Nishanth); `src/api/`, `frontend/`, `deployment/` (Niranjan); tests for each.

## 6. Things that cannot be verified by us

* The ten evaluator trips are not in the data we received. The API supports them through
  `PATHSNAP_EVAL_DATA_ROOT`. The rehearsal used the student-test trips, clearly labelled as such.
  **Running on all ten evaluator trips remains pending until the organisers supply them.**
* The two-laptop network test was run on one machine through its LAN IP address. It must be repeated on two physical
  laptops before the evaluation (`deployment/DEPLOYMENT.md`, checklist).
