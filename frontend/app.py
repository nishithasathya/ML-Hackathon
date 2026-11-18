"""PathSnap evaluator UI (Streamlit, runs on Laptop 2).

    PATHSNAP_API_URL=http://<laptop-1-ip>:8000 streamlit run frontend/app.py --server.address 0.0.0.0

Shows the trips the API offers, sends the selected trip id to the API, and
displays what the API returns: the predicted route on an interactive Folium
map, the server-side metrics and the inference time. It holds no data, no
model and no reference routes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_client import ApiError, PathSnapClient, metrics_rows  # noqa: E402

COLOURS = {"network": "#9e9e9e", "gps": "#1565c0", "route": "#d81b60"}

st.set_page_config(page_title="PathSnap evaluator", layout="wide")
st.title("PathSnap: map-matching evaluator")

api_url = st.sidebar.text_input("API base URL (Laptop 1)", os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000"))
client = PathSnapClient(api_url)

with st.sidebar:
    st.subheader("Connection")
    try:
        health = client.health()
        state = "connected" if health["status"] == "ok" else "connected (degraded)"
        st.success(f"API {state}")
        st.caption(f"model: {health.get('model_key')} | bundle {health.get('bundle_version')} | "
                   f"dataset: {health['dataset_label']} | trips: {health['n_trips_available']}")
        if health.get("detail"):
            st.warning(health["detail"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Cannot reach the API at {api_url}: {exc}")
        st.stop()

try:
    listing = client.trip_listing()
except ApiError as exc:
    st.error(str(exc))
    st.stop()

st.write(f"**Dataset served by the API:** `{listing['dataset_label']}`, {listing['count']} trips "
         f"(expected {listing['expected_count']})")
if listing.get("warning"):
    st.warning(listing["warning"])
trip_ids = [t["trip_id"] for t in listing["trips"]]
cols = st.columns(min(len(trip_ids), 5) or 1)
for i, tid in enumerate(trip_ids):
    if cols[i % len(cols)].button(tid, key=f"btn_{tid}", use_container_width=True):
        st.session_state["trip"] = tid
selected = st.session_state.get("trip")
st.caption("Click a trip to run the full pipeline on the API (preprocessing → candidates → classifier → HMM/Viterbi).")

if selected:
    with st.status(f"Trip {selected}: request sent to {api_url}/api/predict ...", expanded=False) as status:
        try:
            res = client.predict(selected)
            status.update(label=f"Trip {selected}: prediction received", state="complete")
        except Exception as exc:  # noqa: BLE001
            status.update(label=f"Trip {selected}: failed", state="error")
            st.error(str(exc))
            st.stop()

    t = res["timings_s"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GPS observations", f"{res['n_observations']:,}")
    c2.metric("Candidate pairs scored", f"{res['n_candidates']:,}")
    c3.metric("Predicted road arcs", f"{len(res['predicted_arc_sequence']):,}")
    c4.metric("Inference time (server)", f"{t['total']:.2f} s")

    ev = res["evaluation"]
    if ev.get("available"):
        hm = ev["hmm"]
        d1, d2, d3 = st.columns(3)
        d1.metric("Top-1 accuracy (HMM)", f"{hm['top1_accuracy']:.2%}",
                  f"{100 * (hm['top1_accuracy'] - ev['classifier_only']['top1_accuracy']):+.2f} pp vs classifier only")
        d2.metric("Route continuity (HMM)", f"{hm['route_continuity']:.2%}")
        d3.metric("Route F1 (HMM)", f"{hm['route_f1']:.2%}")
        table = pd.DataFrame(metrics_rows(ev)).set_index("method")
        fmt = {c: "{:.4f}" for c in table.columns if c != "Continuity breaks"}
        fmt["Continuity breaks"] = "{:.0f}"
        st.dataframe(table.style.format(fmt, na_rep="-"), use_container_width=True)
        st.caption(f"Metrics computed on the server against its internal reference data "
                   f"({ev['n_labelled_observations']} labelled observations). No reference route is sent to this UI.")
    else:
        st.info(ev.get("detail", "No evaluation available"))

    fmap = folium.Map(tiles="OpenStreetMap", control_scale=True)
    net = folium.FeatureGroup(name="Road network (nearby)")
    for line in res["network_lines"]:
        folium.PolyLine(line, color=COLOURS["network"], weight=1.5, opacity=0.7).add_to(net)
    net.add_to(fmap)
    gps = folium.FeatureGroup(name="GPS observations")
    pts = res["observations"]
    step = max(1, len(pts) // 2500)
    for p in pts[::step]:
        folium.CircleMarker([p["lat"], p["lon"]], radius=2, color=COLOURS["gps"], fill=True, fill_opacity=0.8,
                            tooltip=f"obs {p['obs_idx']} → arc {p['arc_id_hmm']}").add_to(gps)
    gps.add_to(fmap)
    route = folium.FeatureGroup(name="Predicted route (HMM/Viterbi)")
    for line in res["predicted_route_lines"]:
        folium.PolyLine(line, color=COLOURS["route"], weight=5, opacity=0.85).add_to(route)
    route.add_to(fmap)
    folium.Marker([pts[0]["lat"], pts[0]["lon"]], tooltip="start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker([pts[-1]["lat"], pts[-1]["lon"]], tooltip="end", icon=folium.Icon(color="red")).add_to(fmap)
    lats = [p["lat"] for p in pts]
    lons = [p["lon"] for p in pts]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    folium.LayerControl(collapsed=False).add_to(fmap)
    legend = ('<div style="position:fixed;bottom:30px;left:10px;z-index:9999;background:white;padding:6px 8px;'
              'border:1px solid #888;font:12px sans-serif">'
              f'<div><span style="color:{COLOURS["route"]}">━━</span> predicted route</div>'
              f'<div><span style="color:{COLOURS["gps"]}">●</span> GPS observations</div>'
              f'<div><span style="color:{COLOURS["network"]}">━━</span> road network</div></div>')
    fmap.get_root().html.add_child(folium.Element(legend))
    st.subheader(f"Trip {selected}: predicted route")
    components.html(fmap.get_root().render(), height=620)

    with st.expander("Timing breakdown and raw API response"):
        st.json({"timings_s": t, "n_segments": res["n_segments"], "model_key": res["model_key"],
                 "predicted_arc_sequence_head": res["predicted_arc_sequence"][:50]})
