
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\Krishitha\OneDrive\Desktop\Student folder"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = DATA_ROOT / "train"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DIAGNOSTIC_DIR = PROCESSED_DIR / "alignment_diagnostics"

DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

TRIP_IDS = ["00000033", "00000054"]


# ============================================================
# 2. LOAD TRIP DATA
# ============================================================

def load_trip_data(trip_id):

    trip_dir = TRAIN_DIR / trip_id

    track = pd.read_csv(
        trip_dir / f"{trip_id}.track",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    nodes = pd.read_csv(
        trip_dir / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        trip_dir / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"]
    )

    route = pd.read_csv(
        trip_dir / f"{trip_id}.route",
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )

    return track, nodes, arcs, route


# ============================================================
# 3. PROJECT GPS POINT ONTO ROAD SEGMENT
# ============================================================

def project_point_onto_segment(
    gps_lon,
    gps_lat,
    start_lon,
    start_lat,
    end_lon,
    end_lat
):

    longitude_scale = (
        111320 * np.cos(np.radians(gps_lat))
    )

    latitude_scale = 110574

    start_x = (start_lon - gps_lon) * longitude_scale
    start_y = (start_lat - gps_lat) * latitude_scale

    end_x = (end_lon - gps_lon) * longitude_scale
    end_y = (end_lat - gps_lat) * latitude_scale

    dx = end_x - start_x
    dy = end_y - start_y

    length_squared = dx**2 + dy**2

    if length_squared == 0:

        distance = np.sqrt(
            start_x**2 + start_y**2
        )

        return float(distance), 0.0

    fraction = -(
        start_x * dx + start_y * dy
    ) / length_squared

    fraction = float(np.clip(fraction, 0, 1))

    projected_x = start_x + fraction * dx
    projected_y = start_y + fraction * dy

    distance = np.sqrt(
        projected_x**2 + projected_y**2
    )

    return float(distance), fraction


# ============================================================
# 4. BUILD REFERENCE ROUTE GEOMETRY
# ============================================================

def build_reference_route(nodes, arcs, route):

    node_coordinates = nodes[
        ["longitude", "latitude"]
    ].to_numpy(dtype=float)

    arc_array = arcs[
        ["from_node_id", "to_node_id"]
    ].to_numpy(dtype=int)

    route_arc_ids = route["arc_id"].to_numpy(dtype=int)

    route_segments = []

    cumulative_distance = 0.0

    for route_position, arc_id in enumerate(route_arc_ids):

        if arc_id < 0 or arc_id >= len(arc_array):
            raise ValueError(
                f"Invalid route arc ID: {arc_id}"
            )

        from_node, to_node = arc_array[arc_id]

        if (
            from_node < 0
            or to_node < 0
            or from_node >= len(node_coordinates)
            or to_node >= len(node_coordinates)
        ):
            raise ValueError(
                f"Invalid node index in arc {arc_id}"
            )

        start = node_coordinates[from_node]
        end = node_coordinates[to_node]

        mean_latitude = (
            start[1] + end[1]
        ) / 2

        dx = (
            (end[0] - start[0])
            * 111320
            * np.cos(np.radians(mean_latitude))
        )

        dy = (
            (end[1] - start[1])
            * 110574
        )

        segment_length = np.sqrt(
            dx**2 + dy**2
        )

        route_segments.append({

            "route_position": route_position,

            "arc_id": int(arc_id),

            "from_node": int(from_node),

            "to_node": int(to_node),

            "start_lon": start[0],

            "start_lat": start[1],

            "end_lon": end[0],

            "end_lat": end[1],

            "segment_length_m": segment_length,

            "cumulative_start_m": cumulative_distance

        })

        cumulative_distance += segment_length

    return pd.DataFrame(route_segments)


# ============================================================
# 5. CHECK ROUTE CONNECTIVITY
# ============================================================

def inspect_route_connectivity(route_geometry):

    from_nodes = route_geometry[
        "from_node"
    ].to_numpy()

    to_nodes = route_geometry[
        "to_node"
    ].to_numpy()

    connected = (
        to_nodes[:-1] == from_nodes[1:]
    )

    disconnected_count = int(
        (~connected).sum()
    )

    print("\n" + "=" * 65)
    print("REFERENCE ROUTE CONNECTIVITY")
    print("=" * 65)

    print(
        "Route arcs:",
        len(route_geometry)
    )

    print(
        "Adjacent arc pairs:",
        len(connected)
    )

    print(
        "Connected adjacent pairs:",
        int(connected.sum())
    )

    print(
        "Disconnected adjacent pairs:",
        disconnected_count
    )

    if disconnected_count > 0:

        positions = np.where(~connected)[0]

        print("\nFirst disconnected positions:")

        for position in positions[:10]:

            print(
                f"Position {position}: "
                f"arc {route_geometry.iloc[position]['arc_id']} "
                f"ends at node {to_nodes[position]}, "
                f"but next arc starts at node "
                f"{from_nodes[position + 1]}"
            )

    return connected


# ============================================================
# 6. FIND NEAREST REFERENCE ROUTE ARC
# ============================================================

def find_nearest_route_arc(track, route_geometry):

    results = []

    print("\nFinding nearest reference route positions...")

    for gps_index, gps in track.iterrows():

        best_distance = float("inf")

        best_segment = None

        best_fraction = None

        for _, segment in route_geometry.iterrows():

            distance, fraction = project_point_onto_segment(

                gps["longitude"],

                gps["latitude"],

                segment["start_lon"],

                segment["start_lat"],

                segment["end_lon"],

                segment["end_lat"]

            )

            if distance < best_distance:

                best_distance = distance

                best_segment = segment

                best_fraction = fraction

        results.append({

            "gps_index": gps_index,

            "timestamp": gps["timestamp"],

            "longitude": gps["longitude"],

            "latitude": gps["latitude"],

            "nearest_route_position": int(
                best_segment["route_position"]
            ),

            "nearest_route_arc_id": int(
                best_segment["arc_id"]
            ),

            "distance_to_route_m": best_distance,

            "projection_fraction": best_fraction

        })

        if (gps_index + 1) % 500 == 0:

            print(
                f"Processed {gps_index + 1}/"
                f"{len(track)} GPS observations"
            )

    return pd.DataFrame(results)


# ============================================================
# 7. ANALYSE ROUTE PROGRESSION
# ============================================================

def analyse_progression(alignment, position_column, distance_column):

    positions = alignment[
        position_column
    ].to_numpy()

    differences = np.diff(positions)

    backward_steps = int(
        (differences < 0).sum()
    )

    forward_steps = int(
        (differences > 0).sum()
    )

    unchanged_steps = int(
        (differences == 0).sum()
    )

    print("\nRoute progression:")

    print(
        "Backward transitions:",
        backward_steps
    )

    print(
        "Forward transitions:",
        forward_steps
    )

    print(
        "Unchanged transitions:",
        unchanged_steps
    )

    print("\nRoute position statistics:")

    print(
        alignment[position_column]
        .describe()
        .round(3)
    )

    print("\nDistance statistics:")

    print(
        alignment[distance_column]
        .describe()
        .round(3)
    )


# ============================================================
# 8. COMPARE WITH EXISTING MONOTONIC ALIGNMENT
# ============================================================

def compare_with_monotonic_alignment(
    trip_id,
    nearest_alignment
):

    existing_file = (
        PROCESSED_DIR
        / f"trip_{trip_id}_monotonic_alignment.csv"
    )

    if not existing_file.exists():

        print(
            "\nExisting monotonic alignment not found."
        )

        return None

    monotonic = pd.read_csv(existing_file)

    comparison = nearest_alignment.merge(

        monotonic[[
            "gps_index",
            "aligned_route_position",
            "aligned_arc_id",
            "distance_to_aligned_arc_m"
        ]],

        on="gps_index",

        how="inner"

    )

    comparison["position_difference"] = (

        comparison["nearest_route_position"]

        - comparison["aligned_route_position"]

    )

    comparison["distance_difference_m"] = (

        comparison["distance_to_route_m"]

        - comparison["distance_to_aligned_arc_m"]

    )

    print("\n" + "=" * 65)
    print("NEAREST ROUTE VS MONOTONIC DP")
    print("=" * 65)

    print(
        "Compared observations:",
        len(comparison)
    )

    print("\nNearest-route alignment:")

    analyse_progression(
        comparison,
        "nearest_route_position",
        "distance_to_route_m"
    )

    print("\nExisting monotonic DP alignment:")

    analyse_progression(
        comparison,
        "aligned_route_position",
        "distance_to_aligned_arc_m"
    )

    print("\nRoute-position difference:")

    print(
        comparison["position_difference"]
        .describe()
        .round(3)
    )

    print("\nDistance comparison:")

    print(
        comparison[[
            "distance_to_route_m",
            "distance_to_aligned_arc_m"
        ]].describe().round(3)
    )

    print("\nLargest differences between alignments:")

    print(
        comparison.reindex(
            comparison["position_difference"]
            .abs()
            .sort_values(ascending=False)
            .index
        )[[
            "gps_index",
            "nearest_route_position",
            "aligned_route_position",
            "nearest_route_arc_id",
            "aligned_arc_id",
            "distance_to_route_m",
            "distance_to_aligned_arc_m"
        ]].head(15).to_string(index=False)
    )

    return comparison


# ============================================================
# 9. SAVE DIAGNOSTIC RESULTS
# ============================================================

def save_results(
    trip_id,
    route_geometry,
    nearest_alignment,
    comparison
):

    route_file = (
        DIAGNOSTIC_DIR
        / f"trip_{trip_id}_reference_route_geometry.csv"
    )

    nearest_file = (
        DIAGNOSTIC_DIR
        / f"trip_{trip_id}_nearest_route_alignment.csv"
    )

    route_geometry.to_csv(
        route_file,
        index=False
    )

    nearest_alignment.to_csv(
        nearest_file,
        index=False
    )

    print("\nSaved reference route geometry:")
    print(route_file)

    print("\nSaved nearest-route alignment:")
    print(nearest_file)

    if comparison is not None:

        comparison_file = (
            DIAGNOSTIC_DIR
            / f"trip_{trip_id}_alignment_comparison.csv"
        )

        comparison.to_csv(
            comparison_file,
            index=False
        )

        print("\nSaved alignment comparison:")
        print(comparison_file)


# ============================================================
# 10. PROCESS ONE TRIP
# ============================================================

def process_trip(trip_id):

    print("\n\n" + "#" * 75)
    print(f"PROCESSING TRIP {trip_id}")
    print("#" * 75)

    track, nodes, arcs, route = load_trip_data(
        trip_id
    )

    print("\nDataset dimensions:")

    print("GPS observations:", len(track))
    print("Road nodes:", len(nodes))
    print("Road arcs:", len(arcs))
    print("Reference route arcs:", len(route))

    route_geometry = build_reference_route(
        nodes,
        arcs,
        route
    )

    inspect_route_connectivity(
        route_geometry
    )

    nearest_alignment = find_nearest_route_arc(
        track,
        route_geometry
    )

    print("\n" + "=" * 65)
    print("INDEPENDENT NEAREST-ROUTE ALIGNMENT")
    print("=" * 65)

    analyse_progression(
        nearest_alignment,
        "nearest_route_position",
        "distance_to_route_m"
    )

    comparison = compare_with_monotonic_alignment(
        trip_id,
        nearest_alignment
    )

    save_results(
        trip_id,
        route_geometry,
        nearest_alignment,
        comparison
    )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("=" * 75)
    print("PATHSNAP - INDEPENDENT ROUTE ALIGNMENT DIAGNOSTICS")
    print("=" * 75)

    for trip_id in TRIP_IDS:

        try:

            process_trip(trip_id)

        except Exception as error:

            print(
                f"\nFAILED to process trip {trip_id}: {error}"
            )

    print("\n" + "=" * 75)
    print("ALL DIAGNOSTIC RUNS COMPLETED")
    print("=" * 75)

    print("\nDiagnostic files are stored in:")
    print(DIAGNOSTIC_DIR)


if __name__ == "__main__":
    main()
