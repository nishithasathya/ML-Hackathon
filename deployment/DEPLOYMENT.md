# Two-laptop deployment

```
Laptop 2 (evaluator UI)                       Laptop 1 (model + API)
Streamlit  frontend/app.py   ── HTTP ──►      FastAPI  src/api/main.py  (port 8000)
port 8501                    ◄── JSON ──      artifacts/ (model, configs, HMM)
                                              PATHSNAP_EVAL_DATA_ROOT (10 evaluator trips, incl. .route)
```

* The UI holds **no data and no model**. It only shows what the API returns.
* The reference routes stay on Laptop 1. The API returns predicted arcs, geometry and summary metrics only.
* Every request and processing stage is logged in Laptop 1's terminal and in `logs/api.log`.
  That terminal is what you show the evaluator.

## 0. Both laptops: install

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

On Windows, activate with `.venv\Scripts\activate`. Laptop 2 needs only `streamlit`, `httpx`, `folium` and `pandas`, but installing everything is simplest.

## 1. Same network

Connect both laptops to the same Wi-Fi or to one phone's hotspot. Campus Wi-Fi often blocks laptop-to-laptop
traffic (client isolation), so a phone hotspot is the safest choice.

Find Laptop 1's IP address:
* macOS: `ipconfig getifaddr en0`
* Windows: `ipconfig`, then read the "IPv4 Address" line
* Linux: `hostname -I`

## 2. Laptop 1: API

1. Check that `artifacts/model.joblib` and `artifacts/bundle.json` exist, then run `python scripts/validate_artifacts.py`.
2. Put the organisers' evaluator trips in a folder, for example `~/evaluator_test/<trip_id>/<trip_id>.track|.nodes|.arcs|.route`.
3. Copy `.env.example` to `.env` and set:
   ```
   PATHSNAP_EVAL_DATA_ROOT=/Users/<you>/evaluator_test
   PATHSNAP_DATASET_LABEL=evaluator_test
   ```
4. Start the API:
   ```bash
   ./deployment/start_api.sh
   ```
   On Windows, run `deployment\start_api.bat`.
5. **Firewall.** Allow incoming connections on port 8000.
   * macOS: System Settings → Network → Firewall → Options → allow "python"/"uvicorn" (or click
     "Allow" on the pop-up the first time).
   * Windows: in an admin PowerShell, run
     `New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`.
6. Check from Laptop 1 itself: open `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/docs`.

**Rehearsal without the evaluator data:** set `PATHSNAP_EVAL_DATA_ROOT=data/raw/student_test` and
`PATHSNAP_DATASET_LABEL=student_test (rehearsal)`. The UI then shows that label, so it is never mistaken for
the evaluator set. If no data folder is configured, the API starts but reports `degraded`, and prediction
returns HTTP 503 with an explanation.

## 3. Laptop 2: UI

```bash
python deployment/check_connection.py http://<laptop-1-ip>:8000
```
```bash
PATHSNAP_API_URL=http://<laptop-1-ip>:8000 ./deployment/start_ui.sh
```

On Windows, run `deployment\start_ui.bat http://<laptop-1-ip>:8000`.

Open `http://localhost:8501`. The sidebar shows the connection status, and the API URL can also be changed there.

## 4. Demo flow

1. The UI lists the trips the API serves (10 expected; a warning appears if the count differs).
2. The evaluator clicks any trip. The UI sends `POST /api/predict {"trip_id": ...}` to Laptop 1.
3. Laptop 1's terminal logs the request, the loading, preprocessing, candidates, classifier, HMM, timing
   and the server-side evaluation.
4. The UI shows the metrics, inference time and the Folium map: GPS fixes, nearby road network and the predicted route.

## 5. Shutdown

Press `Ctrl+C` in both terminals. Nothing keeps running in the background. Logs remain in `logs/api.log`.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `check_connection` FAIL / timeout | Same network? API started with `--host 0.0.0.0`? Firewall rule? Try `ping <ip>`. Use a phone hotspot. |
| UI: "Evaluation data not available" | `PATHSNAP_EVAL_DATA_ROOT` is unset or wrong on Laptop 1. Restart the API after fixing it. |
| Health shows `model_loaded: false` | `artifacts/` is missing. Pull the repository again or run `python scripts/build_final_artifacts.py`. |
| Map tiles are grey | Laptop 2 has no internet access. The route and points still draw; only the background tiles are missing. |

## 7. What was verified, and what was not

* **Verified (18 Nov 2026, one MacBook):** the API bound to `0.0.0.0` and was called through the machine's LAN IP
  address. The UI was pointed at that LAN URL. The full prediction ran for all 10 student-test trips (rehearsal
  dataset). The test suite passed.
* **Still to do on the two physical laptops before the evaluation:** repeat steps 1-4 on the real
  network, and run all 10 evaluator trips once the organisers supply them. Record the screenshots required by
  Deliverables §7 (UI, API response, Laptop 1 terminal log).
