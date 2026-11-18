"""Interactive Folium maps of a trip.

Layers (each can be toggled in the layer control):
    * Road network      - arcs of the trip network near the GPS track (grey)
    * Reference route   - the supplied .route arcs (green)
    * Raw GPS points    - every observation (blue dots)
    * Speed outliers    - fixes flagged by preprocessing (red)
    * Gap starts        - first fix after a long time gap (orange)
    * Candidates        - optional: candidate arcs of a few selected observations (purple)
    * Classifier-only   - optional (phase 2): top-1 arc per observation, no HMM (orange)
    * Predicted route   - optional (phase 2): HMM/Viterbi route (magenta)

Only the part of the network around the track is drawn: big trips have
>300,000 arcs, which would make the HTML file unusable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import folium
import numpy as np
import pandas as pd
from branca.element import Element

from src.common.io import RawTrip

COLOURS = {"network": "#9e9e9e", "route": "#2e7d32", "gps": "#1565c0", "outlier": "#d32f2f",
           "gap": "#ef6c00", "candidate": "#7b1fa2", "predicted": "#d81b60", "classifier": "#ff8f00"}
METRES_PER_DEG_LAT = 111_320.0


def _arc_lines(nodes: np.ndarray, arcs: np.ndarray, arc_ids: Iterable[int]):
    """Folium wants [[lat, lon], [lat, lon]] per line (note: lat first)."""
    return [[[nodes[a, 1], nodes[a, 0]], [nodes[b, 1], nodes[b, 0]]] for a, b in arcs[list(arc_ids)]]


def nearby_arc_ids(raw: RawTrip, buffer_m: float = 150.0, max_arcs: int = 15000) -> np.ndarray:
    """Arcs with at least one end node inside the track's bounding box expanded by buffer_m."""
    lon, lat = raw.track[:, 0], raw.track[:, 1]
    dlat = buffer_m / METRES_PER_DEG_LAT
    dlon = buffer_m / (METRES_PER_DEG_LAT * max(np.cos(np.radians(np.nanmean(lat))), 1e-6))
    nodes = raw.nodes
    inside = ((nodes[:, 0] >= np.nanmin(lon) - dlon) & (nodes[:, 0] <= np.nanmax(lon) + dlon)
              & (nodes[:, 1] >= np.nanmin(lat) - dlat) & (nodes[:, 1] <= np.nanmax(lat) + dlat))
    ids = np.flatnonzero(inside[raw.arcs[:, 0]] | inside[raw.arcs[:, 1]])
    if ids.size > max_arcs:  # deterministic thinning for very dense areas
        ids = ids[np.linspace(0, ids.size - 1, max_arcs).astype(int)]
    return ids


def _add_legend(fmap: folium.Map, entries) -> None:
    items = "".join(f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
                    f'margin-right:6px;vertical-align:middle"></span>{label}</div>' for label, c in entries)
    html = ('<div style="position:fixed;bottom:56px;left:12px;z-index:9999;background:white;padding:8px 10px;'
            'border:1px solid #888;border-radius:4px;font:12px sans-serif">' + items + "</div>")
    fmap.get_root().html.add_child(Element(html))


def build_trip_map(raw: RawTrip, track: Optional[pd.DataFrame] = None, candidates: Optional[pd.DataFrame] = None,
                   predicted_arc_ids: Optional[Iterable[int]] = None, show_network: bool = True,
                   show_route: bool = True, network_buffer_m: float = 150.0,
                   classifier_arc_ids: Optional[Iterable[int]] = None) -> folium.Map:
    """Build the map. ``track`` is PreprocessedTrip.track (for outlier/gap flags);
    ``candidates`` is a candidate table restricted to the observations to show."""
    lon, lat = raw.track[:, 0], raw.track[:, 1]
    fmap = folium.Map(location=[float(np.nanmean(lat)), float(np.nanmean(lon))], tiles="OpenStreetMap",
                      control_scale=True)
    legend = []

    if show_network:
        layer = folium.FeatureGroup(name="Road network (near track)", show=True)
        folium.PolyLine(_arc_lines(raw.nodes, raw.arcs, nearby_arc_ids(raw, network_buffer_m)),
                        color=COLOURS["network"], weight=1, opacity=0.6).add_to(layer)
        layer.add_to(fmap)
        legend.append(("Road network", COLOURS["network"]))

    if show_route and raw.route.size:
        layer = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(_arc_lines(raw.nodes, raw.arcs, raw.route), color=COLOURS["route"], weight=5,
                        opacity=0.8, tooltip="Reference route").add_to(layer)
        layer.add_to(fmap)
        legend.append(("Reference route", COLOURS["route"]))

    layer = folium.FeatureGroup(name="Raw GPS observations", show=True)
    for i, (x, y, t) in enumerate(raw.track):
        if np.isnan(x) or np.isnan(y):
            continue
        folium.CircleMarker([y, x], radius=2, color=COLOURS["gps"], fill=True, fill_opacity=0.8, weight=0,
                            tooltip=f"obs {i}, t={t:g}s").add_to(layer)
    layer.add_to(fmap)
    legend.append(("Raw GPS observations", COLOURS["gps"]))

    if track is not None:
        for column, name, colour in (("is_speed_outlier", "Speed outliers (flagged)", COLOURS["outlier"]),
                                     ("is_segment_start", "Segment starts after time gap", COLOURS["gap"])):
            rows = track[track[column].astype(bool)]
            if column == "is_segment_start":
                rows = rows.iloc[1:]  # the first fix is always a segment start; not a gap
            if rows.empty:
                continue
            layer = folium.FeatureGroup(name=name, show=True)
            for r in rows.itertuples():
                folium.CircleMarker([r.lat, r.lon], radius=6, color=colour, fill=False, weight=2,
                                    tooltip=f"{name}: obs {r.obs_idx}").add_to(layer)
            layer.add_to(fmap)
            legend.append((name, colour))

    if candidates is not None and len(candidates):
        layer = folium.FeatureGroup(name="Candidate arcs (selected observations)", show=False)
        for r in candidates.itertuples():
            (a_lat, a_lon), (b_lat, b_lon) = _arc_lines(raw.nodes, raw.arcs, [r.arc_id])[0]
            folium.PolyLine([[a_lat, a_lon], [b_lat, b_lon]], color=COLOURS["candidate"], weight=3,
                            tooltip=f"obs {r.obs_idx}: arc {r.arc_id}, {r.distance_m:.1f} m, rank {r.distance_rank}"
                            ).add_to(layer)
        layer.add_to(fmap)
        legend.append(("Candidate arcs", COLOURS["candidate"]))

    if classifier_arc_ids is not None:
        layer = folium.FeatureGroup(name="Classifier-only prediction (no HMM)", show=False)
        folium.PolyLine(_arc_lines(raw.nodes, raw.arcs, sorted(set(classifier_arc_ids))), color=COLOURS["classifier"],
                        weight=3, opacity=0.9, tooltip="Classifier-only top-1 arc").add_to(layer)
        layer.add_to(fmap)
        legend.append(("Classifier-only arcs", COLOURS["classifier"]))

    if predicted_arc_ids is not None:  # HMM/Viterbi output
        layer = folium.FeatureGroup(name="Predicted route (HMM/Viterbi)", show=True)
        folium.PolyLine(_arc_lines(raw.nodes, raw.arcs, list(dict.fromkeys(predicted_arc_ids))),
                        color=COLOURS["predicted"], weight=3, dash_array="6", tooltip="Predicted route").add_to(layer)
        layer.add_to(fmap)
        legend.append(("Predicted route (HMM)", COLOURS["predicted"]))

    folium.Marker([lat[0], lon[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker([lat[-1], lon[-1]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
    fmap.fit_bounds([[float(np.nanmin(lat)), float(np.nanmin(lon))], [float(np.nanmax(lat)), float(np.nanmax(lon))]])
    folium.LayerControl(collapsed=False).add_to(fmap)
    _add_legend(fmap, legend)
    return fmap


def save_trip_map(fmap: folium.Map, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(path))
    return path
