"""Folium and matplotlib helpers (map layers: GPS, network, reference, predicted)."""
from __future__ import annotations

import folium


def base_map(lat: float, lon: float, zoom: int = 15) -> folium.Map:
    return folium.Map(location=[lat, lon], zoom_start=zoom, tiles="cartodbpositron")


def add_gps_layer(m: folium.Map, track, name: str = "GPS observations", color: str = "#1f77b4") -> folium.FeatureGroup:
    fg = folium.FeatureGroup(name=name)
    for lon, lat in track[["lon", "lat"]].to_numpy():
        folium.CircleMarker([lat, lon], radius=2, color=color, fill=True, fill_opacity=0.8).add_to(fg)
    fg.add_to(m)
    return fg


def add_polyline_layer(m: folium.Map, lonlat_coords, name: str, color: str, weight: int = 4) -> folium.FeatureGroup:
    fg = folium.FeatureGroup(name=name)
    folium.PolyLine([(lat, lon) for lon, lat in lonlat_coords], color=color, weight=weight).add_to(fg)
    fg.add_to(m)
    return fg
