"""Network check from Laptop 2 to Laptop 1: is the API reachable and ready?

    python deployment/check_connection.py http://<laptop-1-ip>:8000
"""
import sys

import httpx


def main(url: str) -> int:
    url = url.rstrip("/")
    try:
        health = httpx.get(f"{url}/health", timeout=5).json()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: cannot reach {url}/health ({exc}).\n"
              "  - Are both laptops on the same Wi-Fi/hotspot?  - Is the API started with --host 0.0.0.0?\n"
              "  - Is port 8000 allowed through Laptop 1's firewall?  - Try: ping <laptop-1-ip>")
        return 1
    print(f"OK: {url} status={health['status']} model={health.get('model_key')} "
          f"dataset={health['dataset_label']} trips={health['n_trips_available']}")
    if health["status"] != "ok":
        print(f"  detail: {health.get('detail')}")
    return 0 if health["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"))
