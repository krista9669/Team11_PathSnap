
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import heapq
import math
import time

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(
    r"C:\Users\Krishitha\OneDrive\Desktop\Student folder"
)

TRAIN_DIR = DATA_ROOT / "train"

HMM_DIR = ROOT / "outputs" / "hmm_results"
OUTPUT_DIR = HMM_DIR

TRIP_ID = 1

# Maximum shortest-path distance between the endpoints
# of two candidate arcs.
MAX_NETWORK_DISTANCE = 1000.0

# Maximum plausible vehicle speed in metres/second.
MAX_SPEED_MPS = 70.0

# Penalty for moving backwards on the same directed arc.
REVERSE_TRANSITION_PENALTY = 0.05

# Very small score for unreachable transitions.
UNREACHABLE_PROBABILITY = 1e-8

# Cache size controls how many source-node searches are retained.
SHORTEST_PATH_CACHE_SIZE = 256

# Numerical safety.
EPSILON = 1e-12

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_TIME = time.perf_counter()


# ============================================================
# 2. LOGGING
# ============================================================

def log(message):

    elapsed = time.perf_counter() - START_TIME

    print(
        f"[{elapsed:8.2f}s] {message}",
        flush=True
    )


# ============================================================
# 3. HAVERSINE DISTANCE
# ============================================================

def haversine_m(lat1, lon1, lat2, lon2):

    radius = 6371000.0

    lat1, lon1, lat2, lon2 = map(
        math.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return (
        2 * radius
        * math.asin(min(1.0, math.sqrt(a)))
    )


# ============================================================
# 4. FIND TRIP DIRECTORY
# ============================================================

def find_trip_directory(trip_id, partition="train"):

    trip_name = f"{int(trip_id):08d}"

    partition_dirs = {
        "train": TRAIN_DIR,
        "student_test": DATA_ROOT / "student_test",
    }
    if partition not in partition_dirs:
        raise ValueError(f"Unsupported data partition: {partition}")

    trip_directory = partition_dirs[partition] / trip_name

    if not trip_directory.exists():

        raise FileNotFoundError(
            f"Trip directory not found: {trip_directory}"
        )

    return trip_directory


# ============================================================
# 5. LOAD ROAD NETWORK
# ============================================================

def load_road_network(trip_directory, trip_id):

    trip_name = f"{int(trip_id):08d}"

    nodes_path = (
        trip_directory / f"{trip_name}.nodes"
    )

    arcs_path = (
        trip_directory / f"{trip_name}.arcs"
    )

    if not nodes_path.exists():
        raise FileNotFoundError(nodes_path)

    if not arcs_path.exists():
        raise FileNotFoundError(arcs_path)

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------
    # Actual dataset format:
    #
    # longitude latitude
    #
    # No header.
    # Node ID = zero-based row index.
    # --------------------------------------------------------

    log("Reading road nodes...")

    nodes = pd.read_csv(
        nodes_path,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    coordinates = nodes[
        ["latitude", "longitude"]
    ].to_numpy(dtype=np.float64)

    log(
        f"Loaded {len(coordinates):,} nodes."
    )

    # --------------------------------------------------------
    # Arcs
    # --------------------------------------------------------
    # Actual dataset format:
    #
    # from_node_id to_node_id
    #
    # No header.
    # Arc ID = zero-based row index.
    # --------------------------------------------------------

    log("Reading road arcs...")

    arcs = pd.read_csv(
        arcs_path,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"]
    )

    log(
        f"Loaded {len(arcs):,} arcs."
    )

    # --------------------------------------------------------
    # Construct directed adjacency list
    # --------------------------------------------------------

    adjacency = defaultdict(list)

    arc_information = {}

    invalid_arcs = 0
    zero_length_arcs = 0

    for arc_id, row in enumerate(
        arcs.itertuples(index=False)
    ):

        start_node = int(row.from_node)
        end_node = int(row.to_node)

        if (
            start_node < 0
            or end_node < 0
            or start_node >= len(coordinates)
            or end_node >= len(coordinates)
        ):

            invalid_arcs += 1
            continue

        start_lat, start_lon = coordinates[start_node]

        end_lat, end_lon = coordinates[end_node]

        length = haversine_m(
            start_lat,
            start_lon,
            end_lat,
            end_lon
        )

        if length <= 0:

            zero_length_arcs += 1
            continue

        # Directed connection.
        adjacency[start_node].append(
            (end_node, length)
        )

        # Store each arc independently.
        # This preserves parallel arcs.
        arc_information[arc_id] = {

            "start_node": start_node,

            "end_node": end_node,

            "length": length

        }

    log(
        f"Built directed adjacency for "
        f"{len(adjacency):,} nodes."
    )

    log(
        f"Usable road arcs: {len(arc_information):,}"
    )

    log(
        f"Invalid arcs skipped: {invalid_arcs:,}"
    )

    log(
        f"Zero-length arcs skipped: {zero_length_arcs:,}"
    )

    return adjacency, arc_information


# ============================================================
# 6. CACHED BOUNDED DIJKSTRA
# ============================================================

def create_shortest_path_function(adjacency):

    @lru_cache(
        maxsize=SHORTEST_PATH_CACHE_SIZE
    )
    def distances_from_source(source):

        distances = {
            source: 0.0
        }

        heap = [
            (0.0, source)
        ]

        while heap:

            current_distance, current_node = (
                heapq.heappop(heap)
            )

            if (
                current_distance
                > distances[current_node]
            ):
                continue

            if current_distance > MAX_NETWORK_DISTANCE:
                break

            for neighbor, edge_length in adjacency.get(
                current_node, []
            ):

                new_distance = (
                    current_distance + edge_length
                )

                if new_distance > MAX_NETWORK_DISTANCE:
                    continue

                if new_distance < distances.get(
                    neighbor,
                    float("inf")
                ):

                    distances[neighbor] = new_distance

                    heapq.heappush(
                        heap,
                        (new_distance, neighbor)
                    )

        return distances

    def shortest_path_distance(source, target):

        if source == target:
            return 0.0

        return distances_from_source(
            source
        ).get(
            target,
            float("inf")
        )

    return (
        shortest_path_distance,
        distances_from_source
    )


# ============================================================
# 7. NETWORK DISTANCE BETWEEN CANDIDATE ARCS
# ============================================================

def transition_network_distance(
    previous_arc_id,
    previous_fraction,
    current_arc_id,
    current_fraction,
    arc_information,
    shortest_path_distance
):

    previous_arc = arc_information.get(
        int(previous_arc_id)
    )

    current_arc = arc_information.get(
        int(current_arc_id)
    )

    if (
        previous_arc is None
        or current_arc is None
    ):

        return float("inf"), "missing_arc"

    previous_length = previous_arc["length"]

    current_length = current_arc["length"]

    previous_fraction = float(
        np.clip(previous_fraction, 0.0, 1.0)
    )

    current_fraction = float(
        np.clip(current_fraction, 0.0, 1.0)
    )

    # --------------------------------------------------------
    # Case 1: Same directed arc
    # --------------------------------------------------------

    if int(previous_arc_id) == int(current_arc_id):

        signed_distance = (
            current_fraction - previous_fraction
        ) * previous_length

        if signed_distance >= 0:

            return (
                signed_distance,
                "same_arc_forward"
            )

        return (
            abs(signed_distance),
            "same_arc_reverse"
        )

    # --------------------------------------------------------
    # Case 2: Different arcs
    # --------------------------------------------------------

    previous_remaining = (
        1.0 - previous_fraction
    ) * previous_length

    current_travelled = (
        current_fraction
    ) * current_length

    previous_end_node = (
        previous_arc["end_node"]
    )

    current_start_node = (
        current_arc["start_node"]
    )

    middle_distance = shortest_path_distance(
        previous_end_node,
        current_start_node
    )

    if not math.isfinite(middle_distance):

        return (
            float("inf"),
            "unreachable"
        )

    total_distance = (
        previous_remaining
        + middle_distance
        + current_travelled
    )

    return (
        total_distance,
        "different_arcs"
    )


# ============================================================
# 8. TRANSITION SCORE
# ============================================================

def calculate_transition_probability(
    network_distance,
    gps_displacement,
    time_gap,
    transition_type
):

    if not math.isfinite(network_distance):

        return UNREACHABLE_PROBABILITY

    gps_displacement = max(
        0.0,
        float(gps_displacement)
    )

    # Distance-error tolerance.
    sigma = max(
        15.0,
        0.5 * gps_displacement + 10.0
    )

    distance_error = abs(
        network_distance - gps_displacement
    )

    # Exponential transition score.
    probability = math.exp(
        -distance_error / sigma
    )

    # Penalize reverse movement on the same directed arc.
    if transition_type == "same_arc_reverse":

        probability *= (
            REVERSE_TRANSITION_PENALTY
        )

    # Penalize implausible implied speed.
    if time_gap > 0:

        implied_speed = (
            network_distance / time_gap
        )

        if implied_speed > MAX_SPEED_MPS:

            probability *= 0.001

    return max(
        probability,
        UNREACHABLE_PROBABILITY
    )


# ============================================================
# 9. BUILD TRANSITION MATRICES
# ============================================================

def build_transition_matrices(
    emissions,
    arc_information,
    shortest_path_distance
):

    required_columns = [

        "gps_index",

        "arc_id",

        "projection_fraction",

        "emission_probability"

    ]

    for column in required_columns:

        if column not in emissions.columns:

            raise ValueError(
                f"Missing required emission column: {column}"
            )

    emissions = emissions.copy()

    numeric_columns = [

        "gps_index",

        "arc_id",

        "projection_fraction",

        "emission_probability"

    ]

    for column in numeric_columns:

        emissions[column] = pd.to_numeric(
            emissions[column],
            errors="coerce"
        )

    emissions = emissions.dropna(
        subset=required_columns
    )

    emissions["gps_index"] = (
        emissions["gps_index"].astype(int)
    )

    emissions["arc_id"] = (
        emissions["arc_id"].astype(int)
    )

    if "candidate_rank" in emissions.columns:

        emissions = emissions.sort_values(
            ["gps_index", "candidate_rank"]
        )

    else:

        emissions = emissions.sort_values(
            ["gps_index", "arc_id"]
        )

    # One candidate group per GPS observation.
    groups = [

        group.reset_index(drop=True)

        for _, group in emissions.groupby(
            "gps_index",
            sort=True
        )

    ]

    if len(groups) < 2:

        raise ValueError(
            "At least two GPS observations are required."
        )

    log(
        f"Building transitions for "
        f"{len(groups):,} observations."
    )

    matrices = []

    transition_records = []

    reachable_count = 0

    unreachable_count = 0

    fallback_rows = 0

    total_pairs = len(groups) - 1

    # --------------------------------------------------------
    # Process consecutive GPS observations
    # --------------------------------------------------------

    for pair_index in range(total_pairs):

        previous_group = groups[pair_index]

        current_group = groups[pair_index + 1]

        previous_gps_index = int(
            previous_group.iloc[0]["gps_index"]
        )

        current_gps_index = int(
            current_group.iloc[0]["gps_index"]
        )

        previous_count = len(previous_group)

        current_count = len(current_group)

        matrix = np.full(
            (previous_count, current_count),
            UNREACHABLE_PROBABILITY,
            dtype=np.float64
        )

        # ----------------------------------------------------
        # GPS displacement
        # ----------------------------------------------------

        if "gps_displacement_m" in current_group.columns:

            displacement_values = pd.to_numeric(
                current_group["gps_displacement_m"],
                errors="coerce"
            )

            gps_displacement = (
                displacement_values.median()
            )

            if pd.isna(gps_displacement):

                gps_displacement = 0.0

        else:

            gps_displacement = 0.0

        # ----------------------------------------------------
        # Time gap
        # ----------------------------------------------------

        if "time_gap_s" in current_group.columns:

            gap_values = pd.to_numeric(
                current_group["time_gap_s"],
                errors="coerce"
            )

            time_gap = gap_values.median()

            if pd.isna(time_gap):

                time_gap = 0.0

        else:

            time_gap = 0.0

        # ----------------------------------------------------
        # Evaluate every candidate pair
        # ----------------------------------------------------

        for i, previous_candidate in enumerate(
            previous_group.itertuples(index=False)
        ):

            row_has_reachable = False

            for j, current_candidate in enumerate(
                current_group.itertuples(index=False)
            ):

                network_distance, transition_type = (
                    transition_network_distance(

                        previous_candidate.arc_id,

                        previous_candidate.projection_fraction,

                        current_candidate.arc_id,

                        current_candidate.projection_fraction,

                        arc_information,

                        shortest_path_distance

                    )
                )

                reachable = math.isfinite(
                    network_distance
                )

                if reachable:

                    reachable_count += 1

                    row_has_reachable = True

                else:

                    unreachable_count += 1

                probability = (
                    calculate_transition_probability(

                        network_distance,

                        gps_displacement,

                        time_gap,

                        transition_type

                    )
                )

                matrix[i, j] = probability

                transition_records.append({

                    "previous_gps_index":
                        previous_gps_index,

                    "current_gps_index":
                        current_gps_index,

                    "previous_arc_id":
                        int(previous_candidate.arc_id),

                    "current_arc_id":
                        int(current_candidate.arc_id),

                    "network_distance_m":
                        network_distance,

                    "gps_displacement_m":
                        gps_displacement,

                    "time_gap_s":
                        time_gap,

                    "transition_type":
                        transition_type,

                    "reachable":
                        reachable,

                    "transition_probability":
                        probability

                })

            # ------------------------------------------------
            # Row handling
            # ------------------------------------------------
            # If no next candidate is reachable, preserve
            # the low scores. Do not assign a uniform row.
            #
            # Otherwise normalize the row.
            # ------------------------------------------------

            if not row_has_reachable:

                fallback_rows += 1

                matrix[i, :] = (
                    UNREACHABLE_PROBABILITY
                )

            else:

                row_sum = matrix[i].sum()

                if (
                    row_sum <= 0
                    or not np.isfinite(row_sum)
                ):

                    fallback_rows += 1

                    matrix[i, :] = (
                        UNREACHABLE_PROBABILITY
                    )

                else:

                    matrix[i, :] /= row_sum

        matrices.append(matrix)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (pair_index + 1) % 50 == 0
            or pair_index == total_pairs - 1
        ):

            log(
                f"Processed {pair_index + 1:,}/"
                f"{total_pairs:,} observation pairs."
            )

    transition_df = pd.DataFrame(
        transition_records
    )

    return (
        groups,
        matrices,
        transition_df,
        reachable_count,
        unreachable_count,
        fallback_rows
    )


# ============================================================
# 10. SAVE OUTPUTS
# ============================================================

def save_outputs(
    matrices,
    transition_df,
    number_of_observations,
    trip_id
):

    # --------------------------------------------------------
    # Transition details CSV
    # --------------------------------------------------------

    transition_csv_path = (
        OUTPUT_DIR
        / f"trip_{int(trip_id):08d}_transitions.csv"
    )

    transition_df.to_csv(
        transition_csv_path,
        index=False
    )

    log(
        f"Saved transition details: {transition_csv_path}"
    )

    # --------------------------------------------------------
    # Transition matrices NPZ
    # --------------------------------------------------------

    matrix_path = (
        OUTPUT_DIR
        / f"trip_{int(trip_id):08d}_transition_matrices.npz"
    )

    matrix_data = {

        f"transition_{i}": matrix

        for i, matrix in enumerate(matrices)

    }

    matrix_data["num_observations"] = np.array(
        [number_of_observations]
    )

    matrix_data["num_transition_matrices"] = np.array(
        [len(matrices)]
    )

    np.savez_compressed(
        matrix_path,
        **matrix_data
    )

    log(
        f"Saved transition matrices: {matrix_path}"
    )


# ============================================================
# 11. SUMMARY
# ============================================================

def print_summary(
    number_of_observations,
    matrices,
    reachable_count,
    unreachable_count,
    fallback_rows,
    cached_search
):

    total = (
        reachable_count + unreachable_count
    )

    reachable_percentage = (

        100.0 * reachable_count / total

        if total > 0

        else 0.0

    )

    log(
        "========== TRANSITION SUMMARY =========="
    )

    print(
        f"GPS observations: {number_of_observations:,}"
    )

    print(
        f"Transition matrices: {len(matrices):,}"
    )

    print(
        f"Candidate-pair transitions: {total:,}"
    )

    print(
        f"Reachable transitions: {reachable_count:,}"
    )

    print(
        f"Unreachable transitions: {unreachable_count:,}"
    )

    print(
        f"Reachability: {reachable_percentage:.2f}%"
    )

    print(
        f"Fallback rows: {fallback_rows:,}"
    )

    print(
        f"Shortest-path cache: "
        f"{cached_search.cache_info()}"
    )

    print(
        f"Total runtime: "
        f"{time.perf_counter() - START_TIME:.2f} seconds"
    )

    log(
        "Transition calculation completed."
    )


# ============================================================
# 12. MAIN
# ============================================================

def main(trip_id=TRIP_ID, partition="train"):

    global START_TIME

    START_TIME = time.perf_counter()

    log(
        f"Processing trip {int(trip_id):08d}."
    )

    # --------------------------------------------------------
    # Find trip
    # --------------------------------------------------------

    trip_directory = find_trip_directory(trip_id, partition)

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    adjacency, arc_information = load_road_network(
        trip_directory,
        trip_id
    )

    # --------------------------------------------------------
    # Initialize cached shortest-path search
    # --------------------------------------------------------

    (
        shortest_path_distance,
        cached_search
    ) = create_shortest_path_function(
        adjacency
    )

    # --------------------------------------------------------
    # Load emission probabilities
    # --------------------------------------------------------

    emission_path = (
        HMM_DIR
        / f"trip_{int(trip_id):08d}_emissions.csv"
    )

    if not emission_path.exists():

        raise FileNotFoundError(
            f"Emission file not found: {emission_path}"
        )

    log(
        "Loading emission probabilities..."
    )

    emissions = pd.read_csv(
        emission_path
    )

    log(
        f"Loaded {len(emissions):,} candidate records."
    )

    # --------------------------------------------------------
    # Build matrices
    # --------------------------------------------------------

    (
        groups,
        matrices,
        transition_df,
        reachable_count,
        unreachable_count,
        fallback_rows
    ) = build_transition_matrices(

        emissions,

        arc_information,

        shortest_path_distance

    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        matrices,
        transition_df,
        len(groups),
        trip_id
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(

        len(groups),

        matrices,

        reachable_count,

        unreachable_count,

        fallback_rows,

        cached_search

    )


if __name__ == "__main__":

    main()