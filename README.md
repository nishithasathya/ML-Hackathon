# PathSnap: Map-Matching & Route Reconstruction (UE24CS352A, PES University)

PathSnap matches noisy GPS observations to a road network and reconstructs the route a vehicle drove.

For every GPS fix, nearby road arcs are generated as **candidates**. A trained classifier scores each
(fix, candidate) pair. A **Hidden Markov Model with Viterbi decoding** then picks a connected route through
those candidates. A **FastAPI** backend (Laptop 1) serves predictions to a **Streamlit** evaluator UI (Laptop 2).

**Status: FINAL EVALUATION.** All components are implemented and were run on the 80 training and 10
student-test trips. Numbers are in `results/` and in the technical report
(`docs/technical_report/PathSnap_Final_Technical_Report.pdf`). The 10 evaluator trips have not been
supplied to us; the API serves them from `PATHSNAP_EVAL_DATA_ROOT` once they are
(see `deployment/DEPLOYMENT.md`).

## Team and module ownership

| Member | Area | Phase-2 branch | Owns |
|---|---|---|---|
| Nishitha S (integration lead) | Integration, data pipeline, evaluation, inference, report | `integration/final-shared`, `results/final-evidence` | `src/common/`, `src/data_exploration/`, `src/evaluation/`, `src/inference/`, `scripts/`, `tests/conftest.py`, `tests/test_common.py`, `tests/test_evaluation.py`, docs |
| Neha | Preprocessing + model experimentation | `feature/model-training` | `src/preprocessing/`, `src/models/`, `tests/test_preprocessing.py`, `tests/test_models.py` |
| Nishanth N | Features, Bayesian optimisation, HMM/Viterbi | `feature/features-hmm-optimisation` | `src/features/`, `src/hmm/`, `src/optimization/`, `tests/test_features.py`, `tests/test_hmm.py`, `tests/test_optimization.py` |
| Niranjan | API, UI, visualisation, deployment | `feature/api-ui-deployment` | `src/api/`, `frontend/`, `src/visualization/`, `deployment/`, `tests/test_api.py`, `tests/test_ui.py`, `tests/test_visualization.py` |

## Repository layout

```
src/common/            config (paths, env vars, partition guard), raw readers, geodesy, road graph
src/data_exploration/  phase-1 EDA
src/preprocessing/     cleaning, gap segments, spike flags
src/features/          KD-tree candidates, features (FEATURE_COLUMNS), labels, radius study
src/models/            dataset builder, 5-model zoo, GroupKFold CV, comparison, selection rule
src/optimization/      Optuna TPE Bayesian optimisation
src/hmm/               HMM emissions/transitions, Viterbi, classifier-vs-HMM experiments
src/evaluation/        metric definitions (Top-1, macro F1, log loss, continuity, route P/R/F1)
src/inference/         artefact bundle (export/load) + label-free end-to-end predictor
src/api/               FastAPI app, service, server-side evaluator, schemas, settings
src/visualization/     Folium maps (incl. predicted route), EDA + experiment figures
frontend/              Streamlit evaluator UI + API client
deployment/            start scripts (macOS/Linux/Windows), connection check, DEPLOYMENT.md
artifacts/             final model bundle (model.joblib + bundle.json)
results/               committed experiment evidence (CSV/JSON/PNG/HTML)
scripts/               full pipeline, artefact export/validation, API rehearsal, report builder
tests/                 pytest suite (synthetic trips; no dataset needed)
docs/                  audit, interfaces, workflow, team READMEs, technical report
```

## Install

```bash
git clone https://github.com/nishithasathya/ML-Hackathon.git
```
```bash
cd ML-Hackathon
```
```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

To run the dataset steps, copy the dataset (git-ignored):

```bash
cp -R "/path/to/Student folder/train" "/path/to/Student folder/student_test" data/raw/
```

## Run

Everything runs from the repository root.

```bash
python -m pytest -q
```
```bash
python scripts/validate_artifacts.py --trip-dir data/raw/student_test --trip 00000052
```
```bash
./scripts/run_full_pipeline.sh
```

The full pipeline reproduces every result (about 1.5–2 hours). The individual steps are:

| Step | Command |
|---|---|
| 5 models, GroupKFold CV, student test | `python -m src.models.train_compare` |
| Selection rule | `python -m src.models.select_model` |
| Radius re-evaluation | `python -m src.features.radius_study --model rf` |
| Bayesian optimisation | `python -m src.optimization.bayes_opt --model rf --trials 25` |
| HMM tuning + comparison | `python -m src.hmm.run_hmm --model rf --probs tuned` |
| Export artefacts | `python scripts/build_final_artifacts.py` |
| Figures / maps | `python -m src.visualization.experiment_plots`, `python -m src.visualization.prediction_maps --trips 00000052` |
| HTTP rehearsal (all 10 student-test trips) | `python scripts/run_api_rehearsal.py` |
| Report PDF | `python scripts/build_report.py` |

## Deploy (two laptops)

Run this on Laptop 1:

```bash
PATHSNAP_EVAL_DATA_ROOT=/path/to/evaluator_test ./deployment/start_api.sh
```

Run this on Laptop 2:

```bash
PATHSNAP_API_URL=http://<laptop-1-ip>:8000 ./deployment/start_ui.sh
```

Firewall, Windows and troubleshooting notes are in `deployment/DEPLOYMENT.md`.

## Data-leakage rules followed

* Evaluator-testing data is never read by training or analysis code (`check_partition`). The API reads it only
  from an explicitly configured folder.
* Models see only `FEATURE_COLUMNS`. Labels (`is_reference`, `ref_arc_id`, `ref_distance_m`) are separate, and a
  test asserts this.
* Prediction loads `.track/.nodes/.arcs` only (`load_trip_for_inference`). The `.route` file is read afterwards
  by the server-side evaluator, and only metrics are returned.
* CV folds are split by trip (GroupKFold). Imputers and scalers are fitted inside each fold's pipeline.
* Model selection, the radius, hyperparameters and decoder parameters are chosen on training trips. The student
  test is used for reporting and comparison only.
