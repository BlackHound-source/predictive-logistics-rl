"""Interactive corridor and node network visualization using Folium."""

import folium
from config import THEATERS


def build_tactical_gis_map(theater_key, route_health, chosen_action, sector_stock_ratio):
    theater = THEATERS.get(theater_key, THEATERS["LADAKH_NORTHERN_AXIS"])

    map_object = folium.Map(
        location=theater["center"],
        zoom_start=theater["zoom"],
        tiles="OpenStreetMap",
        control_scale=True,
    )

    primary_color = "#6f8f45" if route_health >= 0.55 else "#b23b3b"
    folium.PolyLine(
        theater["primary_route"],
        color=primary_color,
        weight=5,
        opacity=0.90,
        tooltip=f"PRIMARY SUPPLY CORRIDOR // INTEGRITY {route_health * 100:.1f}%",
    ).add_to(map_object)

    if route_health < 0.55 or chosen_action == "ALTERNATE ROUTE":
        folium.PolyLine(
            theater["alternate_route"],
            color="#c69a3a",
            weight=5,
            dash_array="8, 8",
            opacity=0.95,
            tooltip="ALTERNATE CORRIDOR // REROUTE AVAILABLE",
        ).add_to(map_object)

    for code, info in theater["nodes"].items():
        node_color = "#607d4d"
        if code == "POST_SIERRA" and sector_stock_ratio < 0.55:
            node_color = "#b43c3c"
        elif code == "NODE_TRANSIT" and route_health < 0.55:
            node_color = "#c69a3a"

        folium.CircleMarker(
            location=[info["lat"], info["lon"]],
            radius=8,
            color=node_color,
            fill=True,
            fill_color=node_color,
            fill_opacity=0.95,
            weight=2,
            popup=f"<b>{info['name']}</b><br>ROLE: {info['role']}<br>NODE: {code}",
        ).add_to(map_object)

    all_points = theater["primary_route"] + theater["alternate_route"]
    map_object.fit_bounds([
        [min(p[0] for p in all_points), min(p[1] for p in all_points)],
        [max(p[0] for p in all_points), max(p[1] for p in all_points)],
    ])

    return map_object._repr_html_()
