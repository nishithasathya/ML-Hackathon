#!/usr/bin/env bash
# Laptop 1: start the FastAPI backend, listening on the local network.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
[ -d .venv ] && . .venv/bin/activate
: "${PATHSNAP_API_HOST:=0.0.0.0}" "${PATHSNAP_API_PORT:=8000}"
export PATHSNAP_EVAL_DATA_ROOT PATHSNAP_DATASET_LABEL PATHSNAP_ARTIFACT_DIR PATHSNAP_CORS_ORIGINS
python scripts/validate_artifacts.py
echo "Laptop 1 IP addresses (give one of these to Laptop 2):"
(ipconfig getifaddr en0 2>/dev/null; hostname -I 2>/dev/null) | tr ' ' '\n' | grep -E '^[0-9]' || true
echo "API logs are printed here and written to logs/api.log. Stop with Ctrl+C."
exec python -m uvicorn src.api.main:app --host "$PATHSNAP_API_HOST" --port "$PATHSNAP_API_PORT"
