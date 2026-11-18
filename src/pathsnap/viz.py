"""Folium and matplotlib helpers (map layers: GPS, network, reference, predicted).

All map builders take lon/lat arrays; folium wants (lat, lon) so the swap
happens here and nowhere else.
"""
from __future__ import annotations

import colorsys

import folium
import numpy as np
from branca.element import MacroElement, Template

GPS_COLOR = "#1f77b4"      # blue
NETWORK_COLOR = "#8c8c8c"  # grey
ROUTE_COLOR = "#2ca02c"    # green
PRED_COLOR = "#d62728"     # red
OUTLIER_COLOR = "#ff7f0e"  # orange


def base_map(lat: float, lon: float, zoom: int = 15) -> folium.Map:
    # Esri Light Gray Canvas needs no API key and works from file:// (CARTO now wants a key and the
    # OSM tile server rejects requests without a referer).
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors", name="Basemap (Esri light grey)",
        max_zoom=19, max_native_zoom=16).add_to(m)
    return m


def fit_to(m: folium.Map, lon: np.ndarray, lat: np.ndarray) -> None:
    m.fit_bounds([[float(np.min(lat)), float(np.min(lon))], [float(np.max(lat)), float(np.max(lon))]])


def add_gps_layer(m: folium.Map, track, name: str = "GPS observations", color: str = GPS_COLOR,
                  max_points: int = 6000, colors: list[str] | None = None, tooltips: list[str] | None = None,
                  radius: float = 2.5, show: bool = True) -> folium.FeatureGroup:
    """Raw GPS points as small circles. Long tracks are strided to ``max_points`` (noted in the layer name)."""
    lonlat = track[["lon", "lat"]].to_numpy() if hasattr(track, "columns") else np.asarray(track)
    idx = np.arange(len(lonlat))
    if len(idx) > max_points:
        idx = idx[:: int(np.ceil(len(idx) / max_points))]
        name = f"{name} (every {int(np.ceil(len(lonlat) / max_points))}th of {len(lonlat)})"
    fg = folium.FeatureGroup(name=name, show=show)
    for i in idx:
        lon, lat = lonlat[i]
        c = colors[i] if colors is not None else color
        folium.CircleMarker([lat, lon], radius=radius, color=c, weight=1, fill=True, fill_color=c, fill_opacity=0.85,
                            tooltip=tooltips[i] if tooltips is not None else None).add_to(fg)
    fg.add_to(m)
    return fg


def add_polyline_layer(m: folium.Map, lonlat_coords, name: str, color: str, weight: int = 4,
                       opacity: float = 0.9, show: bool = True) -> folium.FeatureGroup:
    fg = folium.FeatureGroup(name=name, show=show)
    folium.PolyLine([(lat, lon) for lon, lat in lonlat_coords], color=color, weight=weight, opacity=opacity).add_to(fg)
    fg.add_to(m)
    return fg


def arc_segments_lonlat(nodes, arcs, arc_ids) -> list[list[tuple[float, float]]]:
    """[(lat, lon), (lat, lon)] per arc, ready for a multi-PolyLine."""
    f = arcs["from_node"].to_numpy()[arc_ids]
    t = arcs["to_node"].to_numpy()[arc_ids]
    lon, lat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    return [[(lat[a], lon[a]), (lat[b], lon[b])] for a, b in zip(f, t)]


def arcs_in_bbox(nodes, arcs, lon: np.ndarray, lat: np.ndarray, margin_m: float = 300.0) -> np.ndarray:
    """Arc ids with at least one end inside the track bounding box grown by ``margin_m``.

    Only these are drawn, otherwise maps of 300k-arc networks become hundreds of MB.
    """
    mlat = margin_m / 111_320.0
    mlon = margin_m / (111_320.0 * max(np.cos(np.radians(np.mean(lat))), 0.05))
    lo_lon, hi_lon = np.min(lon) - mlon, np.max(lon) + mlon
    lo_lat, hi_lat = np.min(lat) - mlat, np.max(lat) + mlat
    nl, nt = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    inside = (nl >= lo_lon) & (nl <= hi_lon) & (nt >= lo_lat) & (nt <= hi_lat)
    f, t = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    return np.flatnonzero(inside[f] | inside[t])


def arcs_near_track(nodes, arcs, lon: np.ndarray, lat: np.ndarray, corridor_m: float = 200.0) -> np.ndarray:
    """Arc ids with an end node within ``corridor_m`` metres of the GPS track (per-trip UTM metres).

    A corridor rather than the bounding box: long diagonal trips have huge
    boxes, and drawing 300k arcs makes the HTML tens of MB.
    """
    from scipy.spatial import cKDTree

    from .geo import LocalProjector

    proj = LocalProjector.for_points(lon, lat)
    tx, ty = proj.forward(lon, lat)
    nx_, ny_ = proj.forward(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    near = cKDTree(np.c_[tx, ty]).query(np.c_[nx_, ny_], distance_upper_bound=corridor_m)[0] <= corridor_m
    f, t = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    return np.flatnonzero(near[f] | near[t])


def add_network_layer(m: folium.Map, nodes, arcs, arc_ids, name: str = "Road network", color: str = NETWORK_COLOR,
                      weight: float = 1.5, show: bool = True) -> folium.FeatureGroup:
    fg = folium.FeatureGroup(name=f"{name} ({len(arc_ids)} arcs)", show=show)
    if len(arc_ids):
        folium.PolyLine(arc_segments_lonlat(nodes, arcs, arc_ids), color=color, weight=weight, opacity=0.7).add_to(fg)
    fg.add_to(m)
    return fg


def add_route_layer(m: folium.Map, nodes, arcs, route_arc_ids, name: str = "Reference route",
                    color: str = ROUTE_COLOR, weight: float = 5, show: bool = True) -> folium.FeatureGroup:
    fg = folium.FeatureGroup(name=name, show=show)
    folium.PolyLine(arc_segments_lonlat(nodes, arcs, np.asarray(route_arc_ids)), color=color, weight=weight,
                    opacity=0.75).add_to(fg)
    fg.add_to(m)
    return fg


def add_legend(m: folium.Map, items: list[tuple[str, str]], title: str = "Legend") -> None:
    rows = "".join(f'<div><span style="display:inline-block;width:14px;height:14px;background:{c};'
                   f'margin-right:6px;border-radius:2px;vertical-align:middle"></span>{label}</div>'
                   for label, c in items)
    html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white; padding: 8px 10px;
                border: 1px solid #999; border-radius: 4px; font: 12px sans-serif; box-shadow: 0 1px 4px rgba(0,0,0,.3)">
      <b>{title}</b>{rows}
    </div>
    {{% endmacro %}}"""
    el = MacroElement()
    el._template = Template(html)
    m.get_root().add_child(el)


def distinct_colors(n: int, seed: int = 148) -> list[str]:
    """n visually distinct hex colours (golden-ratio hue walk), deterministic."""
    out, h = [], (seed % 100) / 100
    for i in range(n):
        h = (h + 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75 if i % 2 else 0.95, 0.85 if i % 3 else 0.65)
        out.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return out
