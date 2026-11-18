#!/usr/bin/env bash
# Laptop 2: start the Streamlit evaluator UI, pointing at Laptop 1's API.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
[ -d .venv ] && . .venv/bin/activate
: "${PATHSNAP_API_URL:?Set PATHSNAP_API_URL=http://<laptop-1-ip>:8000 (in .env or the shell)}"
: "${PATHSNAP_UI_PORT:=8501}"
export PATHSNAP_API_URL
python deployment/check_connection.py "$PATHSNAP_API_URL" || echo "WARNING: API not reachable yet - the UI will show the error."
exec streamlit run frontend/app.py --server.address 0.0.0.0 --server.port "$PATHSNAP_UI_PORT" --server.headless true
