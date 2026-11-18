"""PathSnap evaluator UI (runs on Laptop 2).

Everything goes through the API: this laptop holds no data and no model.

    export PATHSNAP_API_URL=http://<laptop-1-ip>:8000
    streamlit run ui/app.py
"""
from __future__ import annotations

import os
import time

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

API_URL = os.environ.get("PATHSNAP_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: map-matching evaluation")

with st.sidebar:
    st.caption(f"API: {API_URL}")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API reachable (v{health['version']}, model loaded: {health['model_loaded']})")
    except requests.RequestException as exc:
        st.error(f"Cannot reach the API: {exc}")
        st.stop()

trips = requests.get(f"{API_URL}/trips", timeout=10).json()
if not trips:
    st.warning("The API reports no evaluator trips.")
    st.stop()

labels = {f"{t['trip_id']}  ({t['n_observations']} points)": t["trip_id"] for t in trips}
choice = st.radio("Select an evaluation trip", list(labels), horizontal=False)

if st.button("Run map-matching", type="primary"):
    start = time.perf_counter()
    resp = requests.post(f"{API_URL}/predict", json={"trip_id": labels[choice]}, timeout=300)
    st.session_state["result"] = (resp.status_code, resp.json(), time.perf_counter() - start)

if "result" in st.session_state:
    status, body, secs = st.session_state["result"]
    st.caption(f"Round trip: {secs:.2f} s, HTTP {status}")
    if status != 200:
        st.error(body.get("detail", body))
    else:
        metrics = body.get("metrics", {})
        if metrics:
            cols = st.columns(len(metrics))
            for col, (k, v) in zip(cols, metrics.items()):
                col.metric(k, f"{v:.3f}" if isinstance(v, float) else v)
        gps = body.get("gps", [])          # [[lon, lat], ...]
        route = body.get("route_lonlat", [])  # [[lon, lat], ...]
        if gps:
            m = folium.Map(location=[gps[0][1], gps[0][0]], zoom_start=15, tiles="cartodbpositron")
            fg_gps = folium.FeatureGroup(name="GPS observations")
            for lon, lat in gps:
                folium.CircleMarker([lat, lon], radius=2, color="#1f77b4", fill=True).add_to(fg_gps)
            fg_gps.add_to(m)
            if route:
                fg_pred = folium.FeatureGroup(name="Predicted route")
                folium.PolyLine([(lat, lon) for lon, lat in route], color="#d62728", weight=5).add_to(fg_pred)
                fg_pred.add_to(m)
            folium.LayerControl().add_to(m)
            m.fit_bounds([[min(p[1] for p in gps), min(p[0] for p in gps)],
                          [max(p[1] for p in gps), max(p[0] for p in gps)]])
            st_folium(m, height=600, use_container_width=True)
