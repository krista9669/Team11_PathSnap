
from pathlib import Path
import time

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

HMM_DIR = ROOT / "outputs" / "hmm_results"
OUTPUT_DIR = HMM_DIR
TRIP_ID = 1

EPSILON = 1e-12


# ============================================================
# 2. LOAD AND PREPARE EMISSIONS
# ============================================================

def load_emissions(trip_id=TRIP_ID):

    trip_name = f"{int(trip_id):08d}"
    emission_path = HMM_DIR / f"trip_{trip_name}_emissions.csv"

    if not emission_path.exists():
        raise FileNotFoundError(
            f"Emission file not found: {emission_path}"
        )

    print("Loading emission scores...")

    df = pd.read_csv(emission_path)

    required_columns = [
        "gps_index",
        "arc_id",
        "projection_fraction",
        "emission_probability",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    df = df.copy()

    numeric_columns = [
        "gps_index",
        "arc_id",
        "projection_fraction",
        "emission_probability",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_columns
    )

    df["gps_index"] = df["gps_index"].astype(int)
    df["arc_id"] = df["arc_id"].astype(int)

    if "candidate_rank" in df.columns:
        df = df.sort_values(
            ["gps_index", "candidate_rank"]
        )
    else:
        df = df.sort_values(
            ["gps_index", "arc_id"]
        )

    groups = [
        group.reset_index(drop=True)
        for _, group in df.groupby(
            "gps_index",
            sort=True
        )
    ]

    if len(groups) < 2:
        raise ValueError(
            "At least two GPS observations are required."
        )

    for gps_index, group in zip(
        df["gps_index"].drop_duplicates(),
        groups,
    ):
        probabilities = group["emission_probability"].to_numpy(
            dtype=np.float64
        )
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.isclose(probabilities.sum(), 1.0)
        ):
            raise ValueError(
                f"Invalid emission distribution for GPS observation {gps_index}."
            )

    print(f"GPS observations: {len(groups):,}")
    print(f"Candidate records: {len(df):,}")

    return df, groups


# ============================================================
# 3. LOAD TRANSITION MATRICES
# ============================================================

def load_transition_matrices(trip_id=TRIP_ID):

    trip_name = f"{int(trip_id):08d}"
    transition_path = (
        HMM_DIR / f"trip_{trip_name}_transition_matrices.npz"
    )

    if not transition_path.exists():
        raise FileNotFoundError(
            f"Transition matrix file not found: {transition_path}"
        )

    print("Loading transition matrices...")

    matrices = []

    with np.load(transition_path) as data:

        number_of_observations = int(
            data["num_observations"][0]
        )

        number_of_matrices = int(
            data["num_transition_matrices"][0]
        )

        for i in range(number_of_matrices):

            key = f"transition_{i}"

            if key not in data:
                raise KeyError(
                    f"Missing transition matrix: {key}"
                )

            matrices.append(
                data[key].astype(np.float64)
            )

    print(
        f"Loaded {len(matrices):,} transition matrices."
    )

    if len(matrices) != number_of_observations - 1:
        raise ValueError(
            "Transition matrix count does not equal "
            "number of observations minus one."
        )

    return matrices


# ============================================================
# 4. VITERBI DECODING
# ============================================================

def viterbi_decode(groups, transition_matrices):

    number_of_observations = len(groups)

    if len(transition_matrices) != number_of_observations - 1:
        raise ValueError(
            "Number of transition matrices does not match "
            "the number of GPS observations."
        )

    print("\nStarting Viterbi decoding...")

    # --------------------------------------------------------
    # Initial observation
    # --------------------------------------------------------

    first_group = groups[0]

    first_emissions = first_group[
        "emission_probability"
    ].to_numpy(dtype=np.float64)

    if (
        not np.isfinite(first_emissions).all()
        or (first_emissions < 0).any()
        or (first_emissions > 1).any()
    ):
        raise ValueError("Invalid emission probabilities in first observation.")

    first_emissions = np.clip(first_emissions, EPSILON, 1.0)

    # Log probabilities prevent numerical underflow.
    previous_scores = np.log(first_emissions)

    # Backpointer for every observation.
    backpointers = [None]

    # --------------------------------------------------------
    # Dynamic programming
    # --------------------------------------------------------

    for t in range(1, number_of_observations):

        transition_matrix = transition_matrices[t - 1]

        previous_count = len(groups[t - 1])
        current_count = len(groups[t])

        expected_shape = (
            previous_count,
            current_count
        )

        if transition_matrix.shape != expected_shape:
            raise ValueError(
                f"Transition matrix {t - 1} has shape "
                f"{transition_matrix.shape}, expected "
                f"{expected_shape}."
            )

        if (
            not np.isfinite(transition_matrix).all()
            or (transition_matrix < 0).any()
            or (transition_matrix > 1).any()
        ):
            raise ValueError(
                f"Transition matrix {t - 1} contains invalid probabilities."
            )

        current_emissions = groups[t][
            "emission_probability"
        ].to_numpy(dtype=np.float64)

        if (
            not np.isfinite(current_emissions).all()
            or (current_emissions < 0).any()
            or (current_emissions > 1).any()
        ):
            raise ValueError(
                f"Invalid emission probabilities at observation {t}."
            )

        current_emissions = np.clip(current_emissions, EPSILON, 1.0)

        log_emissions = np.log(current_emissions)

        # Convert transition scores to log space.
        log_transitions = np.log(
            np.clip(
                transition_matrix,
                EPSILON,
                1.0
            )
        )

        # For every current candidate, find the best
        # previous candidate.
        #
        # Shape:
        # previous_scores: (previous_count,)
        # log_transitions: (previous_count, current_count)

        scores = (
            previous_scores[:, np.newaxis]
            + log_transitions
        )

        best_previous_indices = np.argmax(
            scores,
            axis=0
        )

        best_previous_scores = scores[
            best_previous_indices,
            np.arange(current_count)
        ]

        # Add current observation's emission score.
        current_scores = (
            best_previous_scores
            + log_emissions
        )

        backpointers.append(
            best_previous_indices
        )

        previous_scores = current_scores

        if (
            t % 100 == 0
            or t == number_of_observations - 1
        ):
            print(
                f"Processed observation "
                f"{t + 1}/{number_of_observations}",
                flush=True
            )

    # --------------------------------------------------------
    # Backtracking
    # --------------------------------------------------------

    print("\nBacktracking the optimal sequence...")

    final_candidate_index = int(
        np.argmax(previous_scores)
    )

    best_path = [
        0
    ] * number_of_observations

    best_path[-1] = final_candidate_index

    for t in range(
        number_of_observations - 1,
        0,
        -1
    ):

        best_path[t - 1] = int(
            backpointers[t][best_path[t]]
        )

    best_log_score = float(
        previous_scores[final_candidate_index]
    )

    print("Viterbi decoding completed.")

    return best_path, best_log_score


# ============================================================
# 5. BUILD PREDICTION DATAFRAME
# ============================================================

def build_predictions(
    trip_id,
    groups,
    best_path,
    best_log_score
):

    records = []

    for t, candidate_index in enumerate(best_path):

        group = groups[t]

        selected = group.iloc[candidate_index]

        record = {
            "trip_id": f"{int(trip_id):08d}",
            "gps_index": int(selected["gps_index"]),
            "viterbi_candidate_index": int(candidate_index),
            "viterbi_arc_id": int(selected["arc_id"]),
            "viterbi_emission_probability": float(
                selected["emission_probability"]
            ),
            "viterbi_log_sequence_score": best_log_score,
        }

        # Preserve useful candidate information.
        optional_columns = [
            "timestamp",
            "candidate_rank",
            "distance_m",
            "projection_fraction",
            "gps_displacement_m",
            "gps_speed_mps",
            "heading_difference",
            "rf_probability",
        ]

        for column in optional_columns:
            if column in selected.index:
                record[column] = selected[column]

        # Match the classifier-only baseline's distance tie-break.
        score_column = (
            "rf_probability"
            if "rf_probability" in group.columns
            else "emission_probability"
        )
        sort_columns = [score_column]
        ascending = [False]
        if "distance_m" in group.columns:
            sort_columns.append("distance_m")
            ascending.append(True)
        classifier_candidate = group.sort_values(
            sort_columns,
            ascending=ascending,
            kind="mergesort",
        ).iloc[0]

        record["classifier_top1_arc_id"] = int(
            classifier_candidate["arc_id"]
        )

        record["classifier_top1_emission_probability"] = float(
            classifier_candidate["emission_probability"]
        )

        record["classifier_top1_distance_m"] = (
            float(classifier_candidate["distance_m"])
            if "distance_m" in group.columns
            else np.nan
        )

        record["prediction_changed"] = (
            record["viterbi_arc_id"]
            != record["classifier_top1_arc_id"]
        )

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 6. TRANSITION QUALITY OF SELECTED PATH
# ============================================================

def evaluate_selected_transitions(predictions, trip_id):

    transition_details_path = (
        HMM_DIR / f"trip_{int(trip_id):08d}_transitions.csv"
    )

    if not transition_details_path.exists():
        print(
            "Transition detail CSV unavailable; "
            "skipping selected-transition diagnostics."
        )
        return

    transitions = pd.read_csv(transition_details_path)

    selected_records = []

    for t in range(1, len(predictions)):

        previous = predictions.iloc[t - 1]
        current = predictions.iloc[t]

        matches = transitions[
            (transitions["previous_gps_index"] == previous["gps_index"])
            & (transitions["current_gps_index"] == current["gps_index"])
            & (transitions["previous_arc_id"] == previous["viterbi_arc_id"])
            & (transitions["current_arc_id"] == current["viterbi_arc_id"])
        ]

        if matches.empty:
            selected_records.append({
                "gps_index": int(current["gps_index"]),
                "reachable": False,
                "network_distance_m": np.nan,
                "transition_type": "not_found"
            })
        else:
            match = matches.iloc[0]

            selected_records.append({
                "gps_index": int(current["gps_index"]),
                "reachable": bool(match["reachable"]),
                "network_distance_m": match["network_distance_m"],
                "transition_type": match["transition_type"]
            })

    selected_df = pd.DataFrame(selected_records)

    selected_path = (
        OUTPUT_DIR
        / f"trip_{int(trip_id):08d}_viterbi_transition_quality.csv"
    )

    selected_df.to_csv(
        selected_path,
        index=False
    )

    print("\nSelected-path transition diagnostics:")

    print(
        f"Reachable selected transitions: "
        f"{selected_df['reachable'].sum():,}/"
        f"{len(selected_df):,}"
    )

    print(
        f"Selected transition reachability: "
        f"{100 * selected_df['reachable'].mean():.2f}%"
    )

    print(
        f"Saved: {selected_path}"
    )


# ============================================================
# 7. SUMMARY AND SAVE
# ============================================================

def save_results(predictions, best_log_score, trip_id):

    trip_name = f"{int(trip_id):08d}"
    prediction_path = OUTPUT_DIR / f"trip_{trip_name}_viterbi_predictions.csv"
    comparison_path = OUTPUT_DIR / f"trip_{trip_name}_classifier_vs_viterbi.csv"

    predictions.to_csv(
        prediction_path,
        index=False
    )

    predictions.to_csv(
        comparison_path,
        index=False
    )

    print("\n========== VITERBI SUMMARY ==========")

    print(
        f"GPS observations: {len(predictions):,}"
    )

    print(
        f"Classifier Top-1 vs Viterbi differences: "
        f"{predictions['prediction_changed'].sum():,}"
    )

    print(
        f"Percentage of changed predictions: "
        f"{100 * predictions['prediction_changed'].mean():.2f}%"
    )

    if "distance_m" in predictions.columns:

        print(
            f"Mean selected GPS-to-road distance: "
            f"{predictions['distance_m'].mean():.3f} m"
        )

        print(
            f"Median selected GPS-to-road distance: "
            f"{predictions['distance_m'].median():.3f} m"
        )

    print(
        f"Best sequence log score: {best_log_score:.6f}"
    )

    print(f"\nPredictions saved to:\n{prediction_path}")

    print(f"\nComparison saved to:\n{comparison_path}")

    print("=====================================")


# ============================================================
# 8. MAIN
# ============================================================

def main(trip_id=TRIP_ID):

    start_time = time.perf_counter()

    print("=====================================")
    print(f"VITERBI DECODING - TRIP {int(trip_id):08d}")
    print("=====================================")

    # Load emissions.
    _, groups = load_emissions(trip_id)

    # Load transition matrices.
    matrices = load_transition_matrices(trip_id)

    # Decode best sequence.
    best_path, best_log_score = viterbi_decode(
        groups,
        matrices
    )

    # Build predictions.
    predictions = build_predictions(
        trip_id,
        groups,
        best_path,
        best_log_score
    )

    # Evaluate selected transitions.
    evaluate_selected_transitions(
        predictions,
        trip_id
    )

    # Save predictions.
    save_results(
        predictions,
        best_log_score,
        trip_id
    )

    elapsed = time.perf_counter() - start_time

    print(
        f"\nTotal Viterbi runtime: {elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    main()