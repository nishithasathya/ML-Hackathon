"""Capture REAL screenshots of the running system for the report (optional tool).

    pip install playwright        # uses the installed Google Chrome (channel="chrome")
    python scripts/capture_screenshots.py

Starts the API (student-test rehearsal data, bound to 0.0.0.0) and the Streamlit
UI as separate processes, drives the UI with headless Chrome through the LAN
address, clicks a trip, and saves screenshots of:
    ui_overview.png        UI after the prediction (metrics + map)
    ui_trip_list.png       UI trip selection (top of page)
    ui_map.png             the Folium map inside the UI
    api_docs.png           FastAPI interactive docs
    api_results_json.png   GET /api/evaluation/results/<trip> in the browser
    api_log.png            the API log lines written for these requests (rendered from logs/api.log)
    folium_<trip>.png      the development Folium map (results/maps/...)
into results/screenshots/. Nothing is drawn by hand: each image is the page as served.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_api_rehearsal import lan_ip  # noqa: E402

TRIP = os.environ.get("PATHSNAP_SCREENSHOT_TRIP", "00000052")


def wait_http(url: str, seconds: int = 90) -> None:
    import httpx
    for _ in range(seconds):
        try:
            httpx.get(url, timeout=2)
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise SystemExit(f"{url} did not come up")


def render_log(text: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = [ln[:150] for ln in text.strip().splitlines()][-38:]
    fig = plt.figure(figsize=(13, 0.22 * len(lines) + 0.6), facecolor="#1e1e1e")
    fig.text(0.01, 0.99, "\n".join(lines), family="monospace", fontsize=8, color="#d4d4d4", va="top")
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    from playwright.sync_api import sync_playwright

    out = ROOT / "results" / "screenshots"
    out.mkdir(parents=True, exist_ok=True)
    ip = lan_ip()
    api_url, ui_url = f"http://{ip}:8766", f"http://{ip}:8502"
    log_file = ROOT / "logs" / "api.log"
    offset = log_file.stat().st_size if log_file.exists() else 0
    env = {**os.environ, "PATHSNAP_EVAL_DATA_ROOT": str(ROOT / "data" / "raw" / "student_test"),
           "PATHSNAP_DATASET_LABEL": "student_test (rehearsal)", "PATHSNAP_API_URL": api_url}
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8766"],
                           cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.address", "0.0.0.0",
                           "--server.port", "8502", "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                          cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_http(f"{api_url}/health")
        wait_http(ui_url)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 1000}, device_scale_factor=1.5)
            page.goto(ui_url)
            page.get_by_role("button", name=TRIP).wait_for(timeout=60000)
            page.screenshot(path=str(out / "ui_trip_list.png"))
            page.get_by_role("button", name=TRIP).click()
            page.get_by_text("predicted route", exact=False).first.wait_for(timeout=180000)
            page.wait_for_timeout(8000)  # map tiles
            page.screenshot(path=str(out / "ui_overview.png"), full_page=True)
            frame = page.locator("iframe").last
            frame.scroll_into_view_if_needed()
            page.wait_for_timeout(3000)
            frame.screenshot(path=str(out / "ui_map.png"))
            page.goto(f"{api_url}/docs")
            page.wait_for_timeout(3000)
            page.screenshot(path=str(out / "api_docs.png"))
            page.goto(f"{api_url}/api/evaluation/results/{TRIP}")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(out / "api_results_json.png"))
            for html in sorted((ROOT / "results" / "maps").glob("*_prediction.html"))[:3]:
                page.goto(html.as_uri())
                page.wait_for_timeout(6000)
                page.screenshot(path=str(out / f"folium_{html.stem}.png"))
            browser.close()
    finally:
        for proc in (ui, api):
            proc.terminate()
            proc.wait(timeout=20)
    with open(log_file) as fh:
        fh.seek(offset)
        excerpt = fh.read()
    (out / "api_log_excerpt.txt").write_text(excerpt)
    render_log(excerpt, out / "api_log.png")
    print(textwrap.dedent(f"Screenshots written to {out} (API {api_url}, UI {ui_url})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
