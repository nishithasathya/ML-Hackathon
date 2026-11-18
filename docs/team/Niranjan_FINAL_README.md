# Niranjan: API, UI, Visualisation, Deployment (final evaluation)

**Branch:** `feature/api-ui-deployment` (round 6, after `feature/inference-pipeline`)

## Your files
| File | What it does |
|---|---|
| `src/api/main.py` | FastAPI app: `/health`, `/api/evaluation/trips`, `POST /api/predict`, `/api/evaluation/results/{id}`; request-log middleware; CORS |
| `src/api/service.py` | Loads the bundle once, lists trips from `PATHSNAP_EVAL_DATA_ROOT`, runs the label-free prediction, builds map geometry |
| `src/api/evaluator.py` | Server-side metrics after prediction (reads `.route` here only, returns numbers only) |
| `src/api/schemas.py`, `src/api/settings.py` | Pydantic schemas (no label fields) and environment-variable configuration |
| `frontend/app.py`, `frontend/api_client.py` | Streamlit evaluator UI (trip buttons, status, metrics, Folium map, timings) and its HTTP client |
| `src/visualization/folium_maps.py` (changed), `prediction_maps.py`, `experiment_plots.py` | HMM and classifier-only map layers, prediction maps, all experiment figures from result CSVs |
| `deployment/*` | `start_api.sh/.bat`, `start_ui.sh/.bat`, `check_connection.py`, `DEPLOYMENT.md` |
| `tests/test_api.py`, `tests/test_ui.py`, `tests/test_visualization.py` | Health, listing, prediction, no label leakage, 404/422/503 errors, logging; client and Streamlit page; map layers |

## Install, test, run
```bash
python -m pytest -q tests/test_api.py tests/test_ui.py tests/test_visualization.py
```
Laptop 1 (API):
```bash
PATHSNAP_EVAL_DATA_ROOT=data/raw/student_test PATHSNAP_DATASET_LABEL="student_test (rehearsal)" ./deployment/start_api.sh
```
Laptop 2 (UI):
```bash
PATHSNAP_API_URL=http://<laptop-1-ip>:8000 ./deployment/start_ui.sh
```

## Git
```bash
git add src/api frontend src/visualization deployment tests/test_api.py tests/test_ui.py tests/test_visualization.py docs/team/Niranjan_FINAL_README.md
```
The full commands are in `docs/PathSnap_Final_GitHub_Workflow.md`, round 6.

## Be ready to explain / show
* Where the reference route is read (only `evaluator.py`, after prediction) and why the UI never sees it.
* What appears in the Laptop 1 log for one request, and how the UI reaches Laptop 1 (IP address, port, firewall, CORS).
* What happens if the evaluator data or the model is missing (503 with a message; health shows `degraded`).
* The map layers, and why only the nearby network is drawn.
