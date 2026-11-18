
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\Krishitha\OneDrive\Desktop\Student folder"
)

TRAIN_DIR = DATA_ROOT / "train"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Maximum distance between sampled points on each road segment
SAMPLE_SPACING_M = 30

# Final candidates retained for each GPS observation
TOP_K = 20

# Search radii, expanded only if too few candidates are found
SEARCH_RADII_M = [100, 250, 500, 1000]

EARTH_RADIUS_M = 6_371_000


# ============================================================
# 2. LOAD TRIP DATA
# ============================================================

def load_trip(trip_id, data_dir=TRAIN_DIR):

    trip_dir = Path(data_dir) / str(trip_id)

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

    return track, nodes, arcs


# ============================================================
# 3. PREPARE ROAD ARC GEOMETRY
# ============================================================

def prepare_road_arcs(nodes, arcs):

    node_coordinates = nodes[
        ["longitude", "latitude"]
    ].to_numpy(dtype=float)

    from_ids = arcs["from_node_id"].to_numpy(dtype=int)
    to_ids = arcs["to_node_id"].to_numpy(dtype=int)

    start_coordinates = node_coordinates[from_ids]
    end_coordinates = node_coordinates[to_ids]

    return start_coordinates, end_coordinates


# ============================================================
# 4. GEOGRAPHIC DISTANCE
# ============================================================

def haversine_distance_m(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = np.radians(
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_M * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


# ============================================================
# 5. BUILD SEGMENT-AWARE SPATIAL INDEX
# ============================================================

def lonlat_to_xyz(coordinates):

    lon = np.radians(coordinates[:, 0])
    lat = np.radians(coordinates[:, 1])

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return np.column_stack((x, y, z))


def build_segment_spatial_index(
    start_coordinates,
    end_coordinates
):

    sampled_coordinates = []
    sampled_arc_ids = []

    print("\nSampling points along road segments...")

    for arc_id, (start, end) in enumerate(
        zip(start_coordinates, end_coordinates)
    ):

        length = haversine_distance_m(
            start[0], start[1],
            end[0], end[1]
        )

        intervals = max(
            1,
            int(np.ceil(length / SAMPLE_SPACING_M))
        )

        fractions = np.linspace(
            0, 1, intervals + 1
        )

        points = (
            start[None, :]
            + fractions[:, None]
            * (end - start)[None, :]
        )

        sampled_coordinates.append(points)

        sampled_arc_ids.append(
            np.full(len(points), arc_id, dtype=np.int32)
        )

        if (arc_id + 1) % 50000 == 0:
            print(f"Sampled {arc_id + 1} road segments")

    sampled_coordinates = np.vstack(sampled_coordinates)
    sampled_arc_ids = np.concatenate(sampled_arc_ids)

    xyz_coordinates = lonlat_to_xyz(
        sampled_coordinates
    )

    tree = cKDTree(xyz_coordinates)

    print("Total sampled road points:", len(sampled_arc_ids))
    print("Segment-aware spatial index created.")

    return tree, sampled_arc_ids


# ============================================================
# 6. POINT-TO-SEGMENT DISTANCE
# ============================================================

def calculate_point_to_segment_distance(
    gps_point,
    segment_start,
    segment_end
):

    lon0, lat0 = gps_point

    longitude_scale = (
        111320 * np.cos(np.radians(lat0))
    )

    latitude_scale = 110574

    start_x = (
        segment_start[0] - lon0
    ) * longitude_scale

    start_y = (
        segment_start[1] - lat0
    ) * latitude_scale

    end_x = (
        segment_end[0] - lon0
    ) * longitude_scale

    end_y = (
        segment_end[1] - lat0
    ) * latitude_scale

    dx = end_x - start_x
    dy = end_y - start_y

    segment_length_squared = dx * dx + dy * dy

    if segment_length_squared == 0:

        return float(
            np.hypot(start_x, start_y)
        )

    t = -(
        start_x * dx + start_y * dy
    ) / segment_length_squared

    t = np.clip(t, 0, 1)

    closest_x = start_x + t * dx
    closest_y = start_y + t * dy

    return float(
        np.hypot(closest_x, closest_y)
    )


# ============================================================
# 7. GENERATE CANDIDATES FOR ONE GPS OBSERVATION
# ============================================================

def generate_candidates_for_point(
    gps_point,
    tree,
    sampled_arc_ids,
    start_coordinates,
    end_coordinates,
    top_k=TOP_K
):

    gps_xyz = lonlat_to_xyz(
        np.asarray([gps_point], dtype=float)
    )[0]

    nearby_arc_ids = set()

    for radius_m in SEARCH_RADII_M:

        # Convert surface distance to unit-sphere chord distance
        chord_radius = 2 * np.sin(
            radius_m / (2 * EARTH_RADIUS_M)
        )

        nearby_sample_indices = tree.query_ball_point(
            gps_xyz,
            r=chord_radius
        )

        nearby_arc_ids.update(
            sampled_arc_ids[nearby_sample_indices].tolist()
        )

        if len(nearby_arc_ids) >= top_k:
            break

    if not nearby_arc_ids:
        return []

    candidates = []

    for arc_id in nearby_arc_ids:

        distance = calculate_point_to_segment_distance(
            gps_point,
            start_coordinates[arc_id],
            end_coordinates[arc_id]
        )

        candidates.append({
            "arc_id": int(arc_id),
            "distance_m": distance
        })

    candidates.sort(
        key=lambda candidate: candidate["distance_m"]
    )

    return candidates[:top_k]


# ============================================================
# 8. GENERATE CANDIDATES FOR A COMPLETE TRIP
# ============================================================

def generate_trip_candidates(trip_id, data_dir=TRAIN_DIR):

    print("\nLoading trip:", trip_id)

    track, nodes, arcs = load_trip(trip_id, data_dir=data_dir)

    print("GPS observations:", len(track))
    print("Road nodes:", len(nodes))
    print("Road arcs:", len(arcs))

    start_coordinates, end_coordinates = (
        prepare_road_arcs(nodes, arcs)
    )

    tree, sampled_arc_ids = build_segment_spatial_index(
        start_coordinates,
        end_coordinates
    )

    all_candidates = []

    for gps_index, gps_row in track.iterrows():

        gps_point = (
            gps_row["longitude"],
            gps_row["latitude"]
        )

        candidates = generate_candidates_for_point(
            gps_point,
            tree,
            sampled_arc_ids,
            start_coordinates,
            end_coordinates
        )

        for rank, candidate in enumerate(
            candidates,
            start=1
        ):

            all_candidates.append({
                "trip_id": trip_id,
                "gps_index": gps_index,
                "timestamp": gps_row["timestamp"],
                "longitude": gps_row["longitude"],
                "latitude": gps_row["latitude"],
                "candidate_rank": rank,
                "arc_id": candidate["arc_id"],
                "distance_m": candidate["distance_m"]
            })

        if (gps_index + 1) % 100 == 0:
            print(
                f"Processed {gps_index + 1}/"
                f"{len(track)} GPS observations"
            )

    return pd.DataFrame(all_candidates)


# ============================================================
# 9. INSPECT CANDIDATE RESULTS
# ============================================================

def inspect_candidates(candidate_df, track):

    print("\n" + "=" * 65)
    print("CANDIDATE GENERATION RESULTS")
    print("=" * 65)

    print("Total candidate records:", len(candidate_df))

    print(
        "GPS observations:",
        candidate_df["gps_index"].nunique()
    )

    print("\nCandidates per observation:")

    counts = candidate_df.groupby("gps_index").size()

    print(counts.describe())

    missing_observations = (
        len(track) - counts.index.nunique()
    )

    print(
        "\nObservations with no candidates:",
        missing_observations
    )

    print("\nFirst GPS observation candidates:")

    print(
        candidate_df[
            candidate_df["gps_index"] == 0
        ][
            ["candidate_rank", "arc_id", "distance_m"]
        ].to_string(index=False)
    )

    print("\nDistance statistics (metres):")

    print(
        candidate_df["distance_m"].describe().round(3)
    )


# ============================================================
# 10. SAVE RESULTS
# ============================================================

def save_candidates(candidate_df, trip_id):

    output_file = (
        OUTPUT_DIR / f"trip_{trip_id}_candidates.csv"
    )

    candidate_df.to_csv(
        output_file,
        index=False
    )

    print("\nCandidates saved to:")
    print(output_file)


# ============================================================
# 11. MAIN
# ============================================================

def main():

    trip_id = "00000001"

    track, _, _ = load_trip(trip_id)

    candidate_df = generate_trip_candidates(trip_id)

    inspect_candidates(candidate_df, track)

    save_candidates(candidate_df, trip_id)

    print("\nCandidate generation completed.")


if __name__ == "__main__":
    main()
