
import streamlit as st
import requests
import folium
import math

from pathlib import Path
from streamlit_folium import st_folium
from folium.plugins import FastMarkerCluster


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="PathSnap - GPS Map Matching",
    page_icon="📍",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "evaluator_test"

REQUIRED_EXTENSIONS = [
    "track",
    "nodes",
    "arcs",
    "route"
]


# ==========================================================
# 2. PAGE DESIGN
# ==========================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)


st.title("📍 PathSnap")

st.subheader(
    "GPS Trajectory Map Matching System"
)

st.write(
    "Interactive GPS trajectory visualization "
    "and road-network map matching."
)


# ==========================================================
# 3. HELPER FUNCTIONS
# ==========================================================

def haversine(point1, point2):

    lat1, lon1 = point1
    lat2, lon2 = point2

    radius = 6371000

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * radius * math.asin(math.sqrt(a))


def get_trip_files(trip_id):

    folder = DATASET_DIR / trip_id

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Trip folder not found: {trip_id}"
        )

    files = {
        ext: folder / f"{trip_id}.{ext}"
        for ext in REQUIRED_EXTENSIONS
    }

    missing = [
        ext for ext, path in files.items()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing dataset files: {missing}"
        )

    return files


# ==========================================================
# 4. LOAD GPS TRACK
# ==========================================================

def load_track(file_path):

    points = []

    with open(file_path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, 1):

            values = line.split()

            if len(values) != 3:
                continue

            try:

                lon = float(values[0])
                lat = float(values[1])
                index = float(values[2])

                if not all(map(
                    math.isfinite,
                    [lon, lat, index]
                )):
                    continue

                if not (
                    -180 <= lon <= 180
                    and -90 <= lat <= 90
                ):
                    continue

                points.append({
                    "longitude": lon,
                    "latitude": lat,
                    "point_index": index
                })

            except ValueError:
                continue

    points.sort(
        key=lambda point: point["point_index"]
    )

    if not points:
        raise ValueError(
            "No valid GPS observations found."
        )

    return points


# ==========================================================
# 5. LOAD NODES
# ==========================================================

def load_nodes(file_path):

    nodes = {}

    with open(file_path, "r", encoding="utf-8") as file:

        for node_id, line in enumerate(file):

            values = line.split()

            if len(values) < 2:
                continue

            try:

                lon = float(values[0])
                lat = float(values[1])

                if not all(map(
                    math.isfinite,
                    [lon, lat]
                )):
                    continue

                if not (
                    -180 <= lon <= 180
                    and -90 <= lat <= 90
                ):
                    continue

                # Node ID is the zero-based row position.
                nodes[node_id] = (lat, lon)

            except ValueError:
                continue

    if not nodes:
        raise ValueError(
            "No valid road-network nodes found."
        )

    return nodes


# ==========================================================
# 6. LOAD ARCS
# ==========================================================

def load_arcs(file_path):

    arcs = []

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            values = line.split()

            if len(values) != 2:
                continue

            try:

                source = int(values[0])
                destination = int(values[1])

                arcs.append(
                    (source, destination)
                )

            except ValueError:
                continue

    return arcs


# ==========================================================
# 7. LOAD REFERENCE ROUTE
# ==========================================================

def load_route(file_path):

    route = []

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            value = line.strip()

            if not value:
                continue

            try:
                route.append(int(value))

            except ValueError:
                continue

    return route


# ==========================================================
# 8. LOAD COMPLETE DATASET
# ==========================================================

def load_complete_trip(trip_id):

    files = get_trip_files(trip_id)

    return {
        "gps": load_track(files["track"]),
        "nodes": load_nodes(files["nodes"]),
        "arcs": load_arcs(files["arcs"]),
        "route": load_route(files["route"])
    }


# ==========================================================
# 9. REFERENCE ROUTE INTERPRETATION
# ==========================================================

def build_node_route(route, nodes):

    coordinates = []

    for node_id in route:

        if node_id not in nodes:
            return None

        coordinates.append(
            nodes[node_id]
        )

    return coordinates


def build_arc_route(route, arcs, nodes):

    coordinates = []

    for arc_id in route:

        if not (0 <= arc_id < len(arcs)):
            return None

        source, destination = arcs[arc_id]

        if source not in nodes or destination not in nodes:
            return None

        start = nodes[source]
        end = nodes[destination]

        if not coordinates:
            coordinates.extend([start, end])
            continue

        previous = coordinates[-1]

        # Choose the endpoint that connects most closely
        # to the previous route segment.
        if haversine(previous, end) < haversine(previous, start):
            start, end = end, start

        coordinates.append(start)
        coordinates.append(end)

    return coordinates


def route_continuity_score(coordinates):

    if not coordinates or len(coordinates) < 2:
        return float("inf")

    distances = [
        haversine(coordinates[i], coordinates[i + 1])
        for i in range(len(coordinates) - 1)
    ]

    distances.sort()

    return distances[len(distances) // 2]


def resolve_reference_route(route, arcs, nodes, mode):

    node_coordinates = None
    arc_coordinates = None

    if mode in ["Auto-detect", "Node IDs"]:

        node_coordinates = build_node_route(
            route, nodes
        )

    if mode in ["Auto-detect", "Arc row indices"]:

        arc_coordinates = build_arc_route(
            route, arcs, nodes
        )

    if mode == "Node IDs":

        return node_coordinates, "Node IDs"

    if mode == "Arc row indices":

        return arc_coordinates, "Arc row indices"

    candidates = []

    if node_coordinates and len(node_coordinates) >= 2:

        candidates.append((
            route_continuity_score(node_coordinates),
            node_coordinates,
            "Node IDs"
        ))

    if arc_coordinates and len(arc_coordinates) >= 2:

        candidates.append((
            route_continuity_score(arc_coordinates),
            arc_coordinates,
            "Arc row indices"
        ))

    if not candidates:
        return None, "Unresolved"

    candidates.sort(key=lambda item: item[0])

    best_score, best_coordinates, best_mode = candidates[0]

    return best_coordinates, best_mode


# ==========================================================
# 10. GET AVAILABLE TRIPS FROM API
# ==========================================================

@st.cache_data(ttl=10)
def get_available_trips():

    response = requests.get(
        f"{API_URL}/trips",
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# ==========================================================
# 11. SIDEBAR - TRIP SELECTION
# ==========================================================

st.sidebar.title("PathSnap Navigation")

st.sidebar.markdown("### Evaluation Trips")

try:

    api_data = get_available_trips()

    available_trips = api_data.get("trips", [])

    trip_ids = [
        trip["trip_id"]
        for trip in available_trips
        if trip.get("status") == "available"
    ]

    if not trip_ids:

        st.error("No available evaluation trips.")
        st.stop()

    selected_trip = st.sidebar.selectbox(
        "Select Evaluation Trip",
        trip_ids
    )

except requests.RequestException as error:

    st.error(
        "Cannot connect to FastAPI. "
        "Start the backend first."
    )

    st.code(str(error))
    st.stop()


# ==========================================================
# 12. LOAD SELECTED TRIP
# ==========================================================

try:

    trip_data = load_complete_trip(
        selected_trip
    )

    gps_points = trip_data["gps"]
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]
    reference_route = trip_data["route"]

except Exception as error:

    st.error(
        f"Dataset loading failed: {error}"
    )

    st.stop()


# ==========================================================
# 13. API TRIP DETAILS
# ==========================================================

try:

    api_response = requests.get(
        f"{API_URL}/trips/{selected_trip}",
        timeout=60
    )

    api_response.raise_for_status()

    api_trip = api_response.json()

except requests.RequestException as error:

    st.error(
        f"API trip request failed: {error}"
    )

    st.stop()


# ==========================================================
# 14. DASHBOARD
# ==========================================================

st.markdown("---")

st.header("Evaluation Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Selected Trip",
    selected_trip
)

col2.metric(
    "GPS Observations",
    len(gps_points)
)

col3.metric(
    "Road Network Nodes",
    len(nodes)
)

col4.metric(
    "Reference Route Entries",
    len(reference_route)
)

st.markdown("---")


# ==========================================================
# 15. ROUTE MAPPING SETTINGS
# ==========================================================

st.header("Reference Route Configuration")

route_mode = st.selectbox(
    "Interpret route IDs as:",
    [
        "Auto-detect",
        "Node IDs",
        "Arc row indices"
    ],
    help=(
        "The route file must be mapped to either node IDs "
        "or zero-based row positions in the arcs file. "
        "Verify the selected interpretation against "
        "the dataset specification."
    )
)

reference_coordinates, resolved_mode = resolve_reference_route(
    reference_route,
    arcs,
    nodes,
    route_mode
)

if reference_coordinates:

    st.success(
        f"Reference route interpreted as: {resolved_mode}"
    )

else:

    st.warning(
        "Reference route could not be mapped. "
        "Check the route ID format."
    )


# ==========================================================
# 16. GPS COORDINATES
# ==========================================================

gps_coordinates = [
    (
        point["latitude"],
        point["longitude"]
    )
    for point in gps_points
]

center_lat = sum(
    point[0] for point in gps_coordinates
) / len(gps_coordinates)

center_lon = sum(
    point[1] for point in gps_coordinates
) / len(gps_coordinates)


# ==========================================================
# 17. MAP CREATION
# ==========================================================

st.markdown("---")

st.header("Geographical Visualization")

st.write(
    "Interactive geographical map displaying "
    "original GPS observations, road-network geometry "
    "and reference route."
)

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=15,
    tiles=None,
    control_scale=True
)


# ==========================================================
# 18. BASEMAP
# ==========================================================

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/World_Street_Map/"
        "MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri World Street Map",
    name="Street Map",
    overlay=False,
    control=True
).add_to(m)


# ==========================================================
# 19. ORIGINAL GPS TRAJECTORY
# ==========================================================

gps_layer = folium.FeatureGroup(
    name="Original GPS Trajectory",
    show=True
)

folium.PolyLine(
    gps_coordinates,
    color="blue",
    weight=4,
    opacity=0.85,
    tooltip="Original GPS Trajectory"
).add_to(gps_layer)

# Limit marker count for performance.
marker_step = max(
    1,
    len(gps_coordinates) // 300
)

sampled_gps = gps_coordinates[::marker_step]

FastMarkerCluster(
    sampled_gps,
    name="GPS Observation Markers"
).add_to(gps_layer)

folium.Marker(
    gps_coordinates[0],
    tooltip="GPS Starting Point",
    icon=folium.Icon(
        color="green",
        icon="play"
    )
).add_to(gps_layer)

folium.Marker(
    gps_coordinates[-1],
    tooltip="GPS Ending Point",
    icon=folium.Icon(
        color="red",
        icon="stop"
    )
).add_to(gps_layer)

gps_layer.add_to(m)


# ==========================================================
# 20. ROAD NETWORK GEOMETRY
# ==========================================================

road_layer = folium.FeatureGroup(
    name="Road Network Geometry",
    show=True
)

lat_values = [
    point[0] for point in gps_coordinates
]

lon_values = [
    point[1] for point in gps_coordinates
]

lat_padding = max(
    (max(lat_values) - min(lat_values)) * 0.20,
    0.001
)

lon_padding = max(
    (max(lon_values) - min(lon_values)) * 0.20,
    0.001
)

min_lat = min(lat_values) - lat_padding
max_lat = max(lat_values) + lat_padding

min_lon = min(lon_values) - lon_padding
max_lon = max(lon_values) + lon_padding

visible_edges = 0

MAX_EDGES = 15000

for source, destination in arcs:

    if visible_edges >= MAX_EDGES:
        break

    if source not in nodes or destination not in nodes:
        continue

    start = nodes[source]
    end = nodes[destination]

    # Keep only edges inside the GPS viewing region.
    if not (
        min_lat <= start[0] <= max_lat
        and min_lon <= start[1] <= max_lon
        and min_lat <= end[0] <= max_lat
        and min_lon <= end[1] <= max_lon
    ):
        continue

    folium.PolyLine(
        [start, end],
        color="green",
        weight=2,
        opacity=0.55
    ).add_to(road_layer)

    visible_edges += 1

road_layer.add_to(m)


# ==========================================================
# 21. REFERENCE ROUTE
# ==========================================================

reference_layer = folium.FeatureGroup(
    name="Reference Route",
    show=True
)

if reference_coordinates and len(reference_coordinates) >= 2:

    folium.PolyLine(
        reference_coordinates,
        color="orange",
        weight=5,
        opacity=0.95,
        tooltip="Reference Route"
    ).add_to(reference_layer)

    folium.Marker(
        reference_coordinates[0],
        tooltip="Reference Route Start",
        icon=folium.Icon(
            color="orange",
            icon="flag"
        )
    ).add_to(reference_layer)

    folium.Marker(
        reference_coordinates[-1],
        tooltip="Reference Route End",
        icon=folium.Icon(
            color="darkred",
            icon="flag"
        )
    ).add_to(reference_layer)

reference_layer.add_to(m)


# ==========================================================
# 22. MAP LAYER CONTROL
# ==========================================================

folium.LayerControl(
    collapsed=False
).add_to(m)


# Fit the map around the GPS trajectory.
m.fit_bounds([
    [
        min(lat_values),
        min(lon_values)
    ],
    [
        max(lat_values),
        max(lon_values)
    ]
])


# ==========================================================
# 23. DISPLAY MAP
# ==========================================================

st_folium(
    m,
    width=1200,
    height=650,
    returned_objects=[],
    key=f"pathsnap_map_{selected_trip}_{resolved_mode}"
)


# ==========================================================
# 24. MAP LEGEND
# ==========================================================

st.markdown("""
### Map Legend

🔵 Original GPS observations

🟢 Road-network geometry

🟠 Reference route

🟢 GPS starting point

🔴 GPS ending point

Use the layer control on the map to enable or
disable individual geographical layers.
""")


# ==========================================================
# 25. ROUTE PREDICTION
# ==========================================================

st.markdown("---")

st.header("Route Prediction")

st.info(
    "The reference route is loaded from the evaluation "
    "dataset. The trained ML prediction model has not "
    "yet been connected to this frontend."
)

st.metric(
    "Prediction Status",
    "Awaiting Model Integration"
)


# ==========================================================
# 26. DATASET VALIDATION
# ==========================================================

st.markdown("---")

st.header("Dataset Validation")

valid_arc_connections = sum(
    1 for source, destination in arcs
    if source in nodes and destination in nodes
)

st.write(
    "Total GPS observations:",
    len(gps_points)
)

st.write(
    "Total road-network nodes:",
    len(nodes)
)

st.write(
    "Total arc connections:",
    len(arcs)
)

st.write(
    "Valid arc connections:",
    valid_arc_connections
)

st.write(
    "Reference route entries:",
    len(reference_route)
)

st.write(
    "Road edges displayed:",
    visible_edges
)

st.write(
    "Reference route interpretation:",
    resolved_mode
)

if reference_coordinates:

    st.write(
        "Mapped reference route coordinates:",
        len(reference_coordinates)
    )

    st.write(
        "Median consecutive route distance (metres):",
        round(
            route_continuity_score(reference_coordinates),
            2
        )
    )


# ==========================================================
# 27. GPS OBSERVATION TABLE
# ==========================================================

with st.expander("View GPS Observations"):

    st.dataframe(
        gps_points,
        use_container_width=True
    )


# ==========================================================
# 28. REFERENCE ROUTE TABLE
# ==========================================================

with st.expander("View Reference Route IDs"):

    st.dataframe(
        {
            "Route ID": reference_route
        },
        use_container_width=True
    )


# ==========================================================
# 29. ROAD NETWORK SAMPLE
# ==========================================================

with st.expander("View Road Network Connections"):

    st.dataframe(
        [
            {
                "Source Node": source,
                "Destination Node": destination
            }
            for source, destination in arcs[:500]
        ],
        use_container_width=True
    )


# ==========================================================
# 30. FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "PathSnap | GPS Trajectory Map Matching System"
)

st.caption(
    "Interactive geographical visualization using Folium."
)
