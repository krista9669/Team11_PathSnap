
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(
    r"C:\Users\Krishitha\OneDrive\Desktop\Student folder"
)

TRAIN_DIR = DATA_ROOT / "train"
PROCESSED_DIR = ROOT / "data" / "processed"

EARTH_RADIUS_M = 6_371_000


def load_trip(trip_id, data_dir=TRAIN_DIR):

    folder = Path(data_dir) / str(trip_id)

    track = pd.read_csv(
        folder / f"{trip_id}.track",
        sep=r"\s+", header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    nodes = pd.read_csv(
        folder / f"{trip_id}.nodes",
        sep=r"\s+", header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        folder / f"{trip_id}.arcs",
        sep=r"\s+", header=None,
        names=["from_node_id", "to_node_id"]
    )

    route = pd.read_csv(
        folder / f"{trip_id}.route",
        sep=r"\s+", header=None,
        names=["arc_id"]
    )

    return track, nodes, arcs, route


def calculate_distances(track, nodes, arcs, route):

    coordinates = nodes[
        ["longitude", "latitude"]
    ].to_numpy(dtype=float)

    arc_array = arcs[
        ["from_node_id", "to_node_id"]
    ].to_numpy(dtype=int)

    route_ids = route["arc_id"].to_numpy(dtype=int)

    route_pairs = arc_array[route_ids]

    starts = coordinates[route_pairs[:, 0]]
    ends = coordinates[route_pairs[:, 1]]

    distances = np.empty(
        (len(track), len(route_ids)),
        dtype=np.float32
    )

    for i, gps in enumerate(track.itertuples(index=False)):

        lon = gps.longitude
        lat = gps.latitude

        sx = (starts[:, 0] - lon) * (
            111320 * np.cos(np.radians(lat))
        )
        sy = (starts[:, 1] - lat) * 110574

        ex = (ends[:, 0] - lon) * (
            111320 * np.cos(np.radians(lat))
        )
        ey = (ends[:, 1] - lat) * 110574

        dx = ex - sx
        dy = ey - sy

        length_sq = dx**2 + dy**2

        numerator = -(sx * dx + sy * dy)

        fraction = np.divide(
            numerator,
            length_sq,
            out=np.zeros_like(numerator),
            where=length_sq > 0
        )

        fraction = np.clip(fraction, 0, 1)

        px = sx + fraction * dx
        py = sy + fraction * dy

        distances[i] = np.sqrt(px**2 + py**2)

    return distances, route_ids


def monotonic_alignment(distances):

    gps_count, route_count = distances.shape

    dp = np.empty(
        (gps_count, route_count),
        dtype=np.float64
    )

    backpointer = np.zeros(
        (gps_count, route_count),
        dtype=np.int32
    )

    dp[0] = distances[0] ** 2

    for i in range(1, gps_count):

        # Minimum previous cumulative cost over positions <= j
        previous = dp[i - 1]

        prefix_min = np.minimum.accumulate(previous)

        prefix_argmin = np.empty(
            route_count,
            dtype=np.int32
        )

        best_position = 0
        best_cost = previous[0]

        for j in range(route_count):

            if previous[j] < best_cost:
                best_cost = previous[j]
                best_position = j

            prefix_argmin[j] = best_position

        dp[i] = distances[i] ** 2 + prefix_min

        backpointer[i] = prefix_argmin

    positions = np.zeros(gps_count, dtype=int)

    positions[-1] = int(np.argmin(dp[-1]))

    for i in range(gps_count - 1, 0, -1):

        positions[i - 1] = backpointer[
            i, positions[i]
        ]

    return positions


def process_trip(trip_id, data_dir=TRAIN_DIR):

    track, nodes, arcs, route = load_trip(
        trip_id,
        data_dir=data_dir,
    )

    distances, route_ids = calculate_distances(
        track, nodes, arcs, route
    )

    positions = monotonic_alignment(distances)

    records = []

    for i, position in enumerate(positions):

        records.append({
            "trip_id": trip_id,
            "gps_index": i,
            "timestamp": track.iloc[i]["timestamp"],
            "longitude": track.iloc[i]["longitude"],
            "latitude": track.iloc[i]["latitude"],
            "aligned_route_position": int(position),
            "aligned_arc_id": int(route_ids[position]),
            "distance_to_aligned_arc_m": float(
                distances[i, position]
            )
        })

    alignment = pd.DataFrame(records)

    output = (
        PROCESSED_DIR /
        f"trip_{trip_id}_monotonic_alignment.csv"
    )

    alignment.to_csv(output, index=False)

    print(
        f"{trip_id}: {len(alignment)} observations aligned | "
        f"mean distance = "
        f"{alignment['distance_to_aligned_arc_m'].mean():.2f} m"
    )

    return len(alignment)


def main():

    feature_files = sorted(
        PROCESSED_DIR.glob("trip_*_features.csv")
    )

    trip_ids = [
        file.stem.replace("trip_", "").replace("_features", "")
        for file in feature_files
    ]

    print("Trips with feature files:", len(trip_ids))

    successful = 0

    for index, trip_id in enumerate(trip_ids, start=1):

        output = (
            PROCESSED_DIR /
            f"trip_{trip_id}_monotonic_alignment.csv"
        )

        if output.exists():
            print(f"Skipping {trip_id}: alignment already exists.")
            successful += 1
            continue

        print(f"\n[{index}/{len(trip_ids)}] Processing {trip_id}")

        try:
            process_trip(trip_id)
            successful += 1

        except Exception as error:
            print(f"FAILED {trip_id}: {error}")

    print("\nCompleted alignments:", successful)
    print("Feature files found:", len(trip_ids))


if __name__ == "__main__":
    main()