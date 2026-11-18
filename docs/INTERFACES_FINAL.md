# Shared interfaces: final version (additions to `INTERFACES.md`)

All phase-1 interfaces are unchanged. Additions are listed with their owner.

## Data flow

```
TRAINING / EXPERIMENTS (train, student_test only)
  load_trip ─► preprocess_trip ─► build_trip_features(with_labels=True) ─► build_partition
           ─► training_subset ─► make_model / cross_validate ─► bayes_opt ─► run_hmm ─► export_bundle

INFERENCE (API, any partition, NO labels)
  load_trip_for_inference ─► predict_trip(raw, bundle):
        preprocess_trip ─► build_trip_features(with_labels=False) ─► model_matrix ─► model.predict_proba
        ─► decode_trip (HMM/Viterbi) ─► TripPrediction
  then, server-side only: load_reference_route ─► evaluate_prediction ─► metrics (numbers only)
```

## Contracts

| Function / object | Owner | Contract |
|---|---|---|
| `config.get_eval_data_root()` | Nishitha | `Path` from `PATHSNAP_EVAL_DATA_ROOT`, or `None` |
| `io.load_trip_for_inference(dir, trip_id)` | Nishitha | `RawTrip` with `route` empty; reads `.track/.nodes/.arcs` only |
| `io.load_reference_route(dir, trip_id)` | Nishitha | `np.ndarray` or `None`; used only by evaluators |
| `graph.RoadGraph(nodes_xy, arcs)` | Nishitha | `.route_distance(a, fa, b, fb, budget_m)` → metres or `inf`; `.arc_key` (duplicate-arc equivalence) |
| `metrics.*` | Nishitha | Definitions in the `metrics.py` docstring; the same everywhere |
| `features.FEATURE_COLUMNS` | Nishanth | The 22 model inputs, in this order; `LABEL_COLUMNS` must never be inputs |
| `features.model_matrix(table)` | Nishanth | `table[FEATURE_COLUMNS].astype(float)`, used by training AND inference |
| `models.make_model(key, params)` | Neha | sklearn `Pipeline`; `key ∈ {logreg, knn, rf, hgb, mlp}` |
| `models.cross_validate(key, params, fit_rows, all_rows, obs, folds)` | Neha | (per-fold metrics DataFrame, out-of-fold probabilities aligned with `all_rows`) |
| `hmm.decode_trip(cands, track, graph, config)` | Nishanth | `cands`: obs_idx, arc_id, offset_frac, prob → (obs_idx, arc_id, offset_frac, prob) per decoded fix + stats |
| `inference.load_bundle(dir)` / `export_bundle(...)` | Nishitha | `artifacts/model.joblib` + `bundle.json` (configs, features, seed, version, sha256) |
| `inference.predict_trip(raw, bundle)` | Nishitha | Raises if `raw.route` is non-empty; returns `TripPrediction` |
| API JSON (`src/api/schemas.py`) | Niranjan | `PredictResponse` contains geometry, predicted arcs and metrics; **no reference route or labels** |

## Table columns added in phase 2

Candidate table: `prev_obs_dist_m`, `next_obs_dist_m`, `along_arc_motion_m`, `heading_cos`.
`ref_arc_id` is now kept as a label column.
Observation table (`build_partition`): `trip_id, obs_idx, ref_arc_id, ref_distance_m, has_reference_candidate, segment_id`.
`trip_id` is always a string with its 8 digits.
