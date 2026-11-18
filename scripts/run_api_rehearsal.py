"""Rehearse the evaluation over real HTTP: start the API, call it for every trip, record the results.

    python scripts/run_api_rehearsal.py [--data data/raw/student_test] [--host <lan-ip>]

Starts uvicorn as a separate process bound to 0.0.0.0, then acts like Laptop 2:
health check, trip listing, POST /api/predict for EVERY listed trip, using the
same client class as the UI and the machine's LAN address (not localhost) when
available. Writes results/api_rehearsal/{rehearsal.csv, api_log_excerpt.txt,
predict_response_<trip>.json}. The dataset label makes clear that this is the
student-test rehearsal, not the evaluator set.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "frontend"))
from api_client import PathSnapClient  # noqa: E402

from src.common.config import get_results_dir  # noqa: E402


def lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "data" / "raw" / "student_test"))
    parser.add_argument("--label", default="student_test (rehearsal)")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    out = get_results_dir("api_rehearsal")
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "api.log"
    start_offset = log_file.stat().st_size if log_file.exists() else 0
    env = {**os.environ, "PATHSNAP_EVAL_DATA_ROOT": args.data, "PATHSNAP_DATASET_LABEL": args.label}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0",
                             "--port", str(args.port)], cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    host = args.host or lan_ip()
    url = f"http://{host}:{args.port}"
    rows = []
    try:
        client = PathSnapClient(url)
        for _ in range(60):
            try:
                health = client.health()
                break
            except Exception:  # noqa: BLE001 - server still starting
                time.sleep(1)
        else:
            raise SystemExit("API did not start")
        print(f"API at {url}: {health}")
        trips = client.trips()
        for trip in trips:
            t0 = time.perf_counter()
            res = client.predict(trip)
            wall = time.perf_counter() - t0
            ev = res["evaluation"]
            rows.append({"trip_id": trip, "api_url": url, "n_observations": res["n_observations"],
                         "n_candidates": res["n_candidates"], "server_inference_s": res["timings_s"]["total"],
                         "round_trip_s": wall, "hmm_top1": ev["hmm"]["top1_accuracy"],
                         "classifier_top1": ev["classifier_only"]["top1_accuracy"],
                         "hmm_route_continuity": ev["hmm"]["route_continuity"],
                         "classifier_route_continuity": ev["classifier_only"]["route_continuity"],
                         "hmm_route_f1": ev["hmm"]["route_f1"]})
            print(rows[-1], flush=True)
            if trip == trips[0]:
                slim = {k: (v[:3] if isinstance(v, list) else v) for k, v in res.items()}
                (out / f"predict_response_{trip}.json").write_text(json.dumps(slim, indent=2))
    finally:
        proc.terminate()
        proc.wait(timeout=20)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "rehearsal.csv", index=False)
    with open(log_file) as fh:
        fh.seek(start_offset)
        (out / "api_log_excerpt.txt").write_text(fh.read())
    print(f"{len(rows)} trips served over HTTP at {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
