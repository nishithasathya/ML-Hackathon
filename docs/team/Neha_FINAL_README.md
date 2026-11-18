# Neha: Preprocessing + Model Experimentation (final evaluation)

**Branch:** `feature/model-training` (round 3, after `integration/final-shared` and `feature/features-hmm`)

## Your files
| File | What it does |
|---|---|
| `src/models/dataset.py` | Builds the candidate table per partition (`build_partition`), the per-observation label table, the training subset (offset-trip rule, every 3rd fix, observations with a correct candidate) and `xy()` = the only place model inputs are selected |
| `src/models/model_zoo.py` | The 5 models as sklearn Pipelines (imputer/scaler inside), default hyperparameters, objectives, recorded training histories |
| `src/models/cv.py` | `trip_folds` (GroupKFold by trip), `cross_validate` (fit on training trips, score ALL candidates of held-out trips), `evaluate_scores` |
| `src/models/train_compare.py` | CLI: CV + final fit + student-test evaluation of all 5 models; writes `results/models/*` |
| `src/models/select_model.py` | Selection rule fixed before looking at results (CV Top-1, tie margin 0.5 pp, then CV log loss) |
| `tests/test_models.py` | 11 tests: string ids, label separation, subset rules, folds never share trips, every model trains/outputs probabilities/serialises, histories, CV, selection rule |

Preprocessing (`src/preprocessing/`) was re-audited for the final version. Duplicates, ordering, gaps and spikes are
already handled and tested, so no code change was needed. The HMM now uses your `is_speed_outlier` flag
(see the HMM section of the report).

## Install, test, run
```bash
pip install -r requirements.txt
```
```bash
python -m pytest -q tests/test_preprocessing.py tests/test_models.py
```
```bash
python -m src.models.train_compare
```
```bash
python -m src.models.select_model
```
`train_compare` takes about 25 minutes, mostly KNN prediction and Random Forest fitting.

## Interfaces
* **Uses:** `build_trip_features`, `model_matrix`, `FEATURE_COLUMNS` (features), `classification_metrics`, `observation_accuracy`, `select_top1` (evaluation), `evaluate_routes` (route metrics).
* **Provides:** `make_model(key, params)`, `cross_validate(...)`, `trip_folds(...)`, `build_partition(...)`, `training_subset(...)`. These are used by optimisation (Nishanth) and the artefact export (Nishitha).

## Git (exact commands are in `docs/PathSnap_Final_GitHub_Workflow.md`, round 3)
```bash
git add src/models tests/test_models.py docs/team/Neha_FINAL_README.md
```
```bash
git commit -m "Five candidate-scoring models with trip-grouped CV, comparison and selection rule"
```
```bash
git push -u origin feature/model-training
```
Then open a PR to `main`.

## Be ready to explain
* Why binary candidate scoring instead of multiclass: arc ids are local to each trip, and the candidate sets change from fix to fix.
* Why GroupKFold by trip: neighbouring 1 Hz fixes are near-duplicates, so a random split would leak.
* Why the training subset is not used for evaluation, and why offset trips 00000033 and 00000054 are excluded from fitting only.
* What each model minimises, why KNN scores 1.0 on its own training rows, and why that is not evidence that it is good.
* The selection rule, and why Random Forest won: tied on CV Top-1 with HGB and MLP, lowest CV log loss.
