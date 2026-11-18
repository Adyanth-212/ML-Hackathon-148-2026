"""PathSnap evaluator UI (Laptop 2).

The UI is a thin API client: it does not read dataset files, reference routes,
or model artifacts. Configure PATHSNAP_API_URL for the API laptop.
"""
from __future__ import annotations

import os
import time

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

API_URL = os.environ.get("PATHSNAP_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = (5, 300)

st.set_page_config(page_title="PathSnap | Route reconstruction", page_icon="🗺️", layout="wide")
st.title("PathSnap: GPS map-matching")
st.caption("Select an evaluator trip, run the server-side prediction, and inspect the reconstructed route.")

with st.sidebar:
    st.subheader("API connection")
    st.code(API_URL, language=None)
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        health_response.raise_for_status()
        health = health_response.json()
        st.success(f"Connected · API v{health['version']}")
        st.caption(f"Prediction mode: {health.get('model') or 'not loaded'}")
    except (requests.RequestException, ValueError) as exc:
        st.error(f"Cannot reach the API: {exc}")
        st.info("Start the API on Laptop 1 and set PATHSNAP_API_URL to its LAN IP.")
        st.stop()

try:
    trips_response = requests.get(f"{API_URL}/trips", timeout=15)
    trips_response.raise_for_status()
    trips = trips_response.json()
except (requests.RequestException, ValueError) as exc:
    st.error(f"Could not load evaluator trips: {exc}")
    st.stop()

if not trips:
    st.warning("The API currently reports no evaluator trips.")
    st.stop()

trip_labels = {
    f"{item['trip_id']} · {item['n_observations']:,} GPS points · {item.get('duration_s', 0):.0f}s": item["trip_id"]
    for item in trips
}
selected_label = st.selectbox("Evaluator trip", list(trip_labels))
selected_trip = trip_labels[selected_label]

if st.button("Run map-matching", type="primary", use_container_width=True):
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{API_URL}/predict", json={"trip_id": selected_trip}, timeout=REQUEST_TIMEOUT
        )
        elapsed = time.perf_counter() - started
        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text[:1000]}
        st.session_state["prediction_result"] = {
            "status": response.status_code,
            "payload": payload,
            "elapsed_s": elapsed,
            "trip_id": selected_trip,
        }
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")

result = st.session_state.get("prediction_result")
if result and result.get("trip_id") == selected_trip:
    st.divider()
    st.subheader(f"Prediction · {selected_trip}")
    st.caption(f"API round trip: {result['elapsed_s']:.2f}s · HTTP {result['status']}")
    body = result["payload"]
    if result["status"] != 200:
        st.error(body.get("detail", body))
    else:
        timings = body.get("timings_ms", {})
        metric_values = body.get("metrics", {})
        metric_order = [
            ("route_f1", "Route F1"),
            ("route_precision", "Route precision"),
            ("route_recall", "Route recall"),
            ("route_continuity", "Continuity"),
            ("route_edit_similarity", "Edit similarity"),
        ]
        visible_metrics = [(key, label) for key, label in metric_order if metric_values.get(key) is not None]
        if visible_metrics:
            columns = st.columns(len(visible_metrics))
            for column, (key, label) in zip(columns, visible_metrics):
                column.metric(label, f"{metric_values[key]:.3f}")
        st.caption(
            " · ".join(f"{stage}: {value:.1f} ms" for stage, value in timings.items())
            + f" · Model: {body.get('model', 'unknown')}"
        )

        gps = body.get("gps", [])
        predicted_route = body.get("route_lonlat", [])
        network = body.get("network_lonlat", [])
        if gps:
            center = [sum(point[1] for point in gps) / len(gps), sum(point[0] for point in gps) / len(gps)]
            map_view = folium.Map(location=center, zoom_start=15, tiles=None, control_scale=True)
            folium.TileLayer("CartoDB positron", name="Light map", control=True).add_to(map_view)

            gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
            for index, (lon, lat) in enumerate(gps):
                folium.CircleMarker(
                    location=[lat, lon], radius=3, color="#1769aa", weight=1,
                    fill=True, fill_opacity=0.9,
                    tooltip=f"GPS observation {index + 1}",
                ).add_to(gps_layer)
            gps_layer.add_to(map_view)

            network_layer = folium.FeatureGroup(name="Nearby road network", show=False)
            for segment in network:
                if len(segment) == 2:
                    folium.PolyLine(
                        [(segment[0][1], segment[0][0]), (segment[1][1], segment[1][0])],
                        color="#929292", weight=2, opacity=0.55,
                    ).add_to(network_layer)
            network_layer.add_to(map_view)

            route_layer = folium.FeatureGroup(name="Predicted route", show=True)
            if predicted_route:
                folium.PolyLine(
                    [(lat, lon) for lon, lat in predicted_route],
                    color="#d62728", weight=5, opacity=0.9,
                    tooltip="Predicted route (reference route is never sent to the UI)",
                ).add_to(route_layer)
            route_layer.add_to(map_view)
            folium.LayerControl(collapsed=False).add_to(map_view)
            map_view.fit_bounds(
                [[min(p[1] for p in gps), min(p[0] for p in gps)],
                 [max(p[1] for p in gps), max(p[0] for p in gps)]]
            )
            st_folium(map_view, height=650, use_container_width=True, returned_objects=[])
        else:
            st.warning("The API returned no GPS coordinates for this trip.")

        with st.expander("Prediction diagnostics"):
            st.write(f"Predicted arcs per observation: {len(body.get('pred_arcs', []))}")
            st.write(f"Unique consecutive route arcs: {len(body.get('route_arc_ids', []))}")
            st.write(f"Nearby network segments displayed: {len(network)}")
            st.json(metric_values)
