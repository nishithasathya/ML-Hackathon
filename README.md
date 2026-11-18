# PathSnap: Map-Matching & Route Reconstruction (UE24CS352A, PES University)

PathSnap matches noisy GPS observations to a road network and reconstructs the
route a vehicle actually drove.

**Current status: FIRST-PHASE EVALUATION milestone.** This repository contains
dataset exploration, validation, preprocessing, candidate generation, feature
engineering and visualisation. **No machine-learning model, Bayesian optimisation,
HMM/Viterbi decoder, API or UI is implemented yet.** Those are planned for later
evaluations (see [`docs/DEFERRED_COMPONENTS.md`](docs/DEFERRED_COMPONENTS.md)).

## Team

| Member | Area | Branch | Folder |
|---|---|---|---|
| Nishitha S (integration lead) | Data exploration + integration | `feature/data-exploration` | `src/data_exploration/` |
| Neha | Preprocessing | `feature/preprocessing` | `src/preprocessing/` |
| Nishanth N | Feature engineering | `feature/feature-engineering` | `src/features/` |
| Niranjan | Visualisation + documentation | `feature/visualization-documentation` | `src/visualization/`, `docs/` |

## Repository layout

```
.
├── README.md                 this file
├── requirements.txt
├── data/raw/                 put the dataset here (git-ignored)
│   ├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
│   └── student_test/<trip_id>/...
├── src/
│   ├── common/               shared config, raw-file readers, geodesy (integration lead)
│   ├── data_exploration/     inventory, validation, statistics, run_eda.py
│   ├── preprocessing/        cleaning, pipeline, run_preprocessing.py
│   ├── features/             spatial index, candidates, labels, features, run_features.py
│   └── visualization/        Folium maps, matplotlib figures, run_visualizations.py
├── scripts/run_first_phase_demo.py   end-to-end demo on one trip
├── tests/                    pytest suite (uses a synthetic mini-trip, no dataset needed)
├── docs/                     scope, interfaces, Git workflow, technical documentation
└── outputs/                  generated results (git-ignored)
```

## Installation

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

On Windows, activate with `.venv\Scripts\activate`.

### Dataset

Copy the two partition folders from `Student folder/` into `data/raw/`:

```bash
cp -R "/path/to/Student folder/train" "/path/to/Student folder/student_test" data/raw/
```

Or leave the data where it is and point to it:

```bash
export PATHSNAP_DATA_ROOT="/path/to/Student folder"
```

Only `train` (80 trips) and `student_test` (10 trips) are readable by the code.
Any other partition name, including evaluator-testing data, is refused by
`src/common/config.py`.

## Running (always from the repository root)

```bash
python -m pytest -q
```
```bash
python -m src.data_exploration.run_eda
```
```bash
python -m src.preprocessing.run_preprocessing --partition train
```
```bash
python -m src.preprocessing.run_preprocessing --partition student_test
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
```bash
python -m src.visualization.run_visualizations figures
```
```bash
python -m src.visualization.run_visualizations maps --partition train --trips 00000010 00000033 00000002
```
```bash
python scripts/run_first_phase_demo.py --partition student_test --trip 00000052
```

Approximate run times measured on a cloud Linux container: EDA ≈ 25 s, preprocessing ≈ 10 s,
radius sweep ≈ 80 s, feature extraction for train ≈ 60 s. The feature files for train take ~56 MB.

## Documentation

* [`docs/FIRST_PHASE_SCOPE.md`](docs/FIRST_PHASE_SCOPE.md): what the official documents require and what we chose for phase 1
* [`docs/FIRST_PHASE_TECHNICAL_DOC.md`](docs/FIRST_PHASE_TECHNICAL_DOC.md): dataset findings, methods, status
* [`docs/INTERFACES.md`](docs/INTERFACES.md): shared functions, data classes and CSV columns
* [`docs/GIT_WORKFLOW.md`](docs/GIT_WORKFLOW.md): exact Git commands per teammate
* [`docs/INTEGRATION.md`](docs/INTEGRATION.md): merge order and integration testing
* [`docs/DEMO_CHECKLIST.md`](docs/DEMO_CHECKLIST.md): what to show at the evaluation
* [`docs/DEFERRED_COMPONENTS.md`](docs/DEFERRED_COMPONENTS.md): what is intentionally not built yet
* [`docs/AI_ASSISTANCE_AND_WORK_LOG.md`](docs/AI_ASSISTANCE_AND_WORK_LOG.md): AI-use disclosure and record of what was run

## Data-leakage rules followed

* Evaluator-testing data is not in the repository and cannot be loaded by the code.
* The reference route is used only to build **labels** (`is_reference`) and data-quality diagnostics, never as a feature.
* The candidate radius was chosen with a sweep over **training trips only**.
* Nothing is fitted (no scalers or models) in this phase, so no statistics leak from student-test data.
