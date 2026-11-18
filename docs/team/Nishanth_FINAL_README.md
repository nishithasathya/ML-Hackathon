# Nishanth N: Features, HMM/Viterbi, Bayesian optimisation (final evaluation)

**Branches:** `feature/features-hmm` (round 2), then `feature/bayesian-optimisation` (round 4).
The ZIP has two folders, one per branch: `round2_features_hmm/` and `round4_optimisation/`.

## Your files
| File | What it does |
|---|---|
| `src/features/features.py` (changed) | `FEATURE_COLUMNS` / `LABEL_COLUMNS` / `model_matrix()`, new context features (previous/next fix distance to the arc, motion along the arc, heading cosine), duplicate-arc label equivalence, string trip ids |
| `src/features/radius_study.py` | Re-evaluates the 25/50/75/100 m radius with a trained model (decision on train CV only) |
| `src/hmm/decoder.py` | `HMMConfig`, emissions, transitions (directed network distance vs GPS displacement), log-space `viterbi`, `decode_trip` (chains split at gaps / missing candidates, spike handling) |
| `src/hmm/run_hmm.py` | Decoder tuning on training trips (out-of-fold scores) and classifier-only vs HMM on student test |
| `src/optimization/bayes_opt.py` | Optuna TPE search, search spaces, trials in SQLite, tuned vs untuned |
| `tests/test_features.py`, `tests/test_hmm.py`, `tests/test_optimization.py` | Feature/label separation, context features, duplicate labels; Viterbi vs brute force, disconnected roads, missing candidates, gaps, spikes; search-space validity, trial recording, best-parameter persistence |

## Install, test, run
```bash
python -m pytest -q tests/test_features.py tests/test_hmm.py tests/test_optimization.py
```
```bash
python -m src.features.radius_study --model rf
```
```bash
python -m src.optimization.bayes_opt --model rf --trials 25
```
```bash
python -m src.hmm.run_hmm --model rf --probs tuned
```

## Interfaces
* **Uses:** `RoadGraph.route_distance` (`src/common/graph.py`), `make_model` / `cross_validate` (Neha).
* **Provides:** `FEATURE_COLUMNS`, `model_matrix`, `build_trip_features(..., with_labels=False)` (inference), `decode_trip`, `HMMConfig` (inference + API).

## Git
For round 2 (copy from `round2_features_hmm/`):
```bash
git add src/features tests/test_features.py src/hmm tests/test_hmm.py docs/team/Nishanth_FINAL_README.md
```
For round 4 (copy from `round4_optimisation/`):
```bash
git add src/optimization tests/test_optimization.py
```
The full commands are in `docs/PathSnap_Final_GitHub_Workflow.md`.

## Be ready to explain
* Hidden states, the emission formula `w·log p`, and the transition formula `-|d_route - d_gc| / beta`, plus why an unreachable move gets a finite penalty.
* The Viterbi recurrence and backtracking, why it runs in log space, and why chains are cut at gaps.
* Why the decoder is tuned on out-of-fold probabilities, not in-sample ones.
* The search space, the objective (CV Top-1), why only every 6th fix is used during the search, and the tuned vs untuned results.
