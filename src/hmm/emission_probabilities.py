
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"

MODEL_PATH = (
    ROOT
    / "outputs"
    / "model_results"
    / "saved_models"
    / "random_forest.joblib"
)

OUTPUT_DIR = ROOT / "outputs" / "hmm_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Must match train_models.py exactly.
BASE_FEATURES = [
    "distance_m",
    "gps_speed_mps",
    "gps_displacement_m",
    "time_gap_s",
    "segment_length_m",
    "heading_difference",
    "projection_fraction",
]


MODEL_FEATURES = BASE_FEATURES + [
    "gps_bearing_sin",
    "gps_bearing_cos",
    "segment_bearing_sin",
    "segment_bearing_cos",
]


# ============================================================
# LOAD TRAINED RANDOM FOREST
# ============================================================

def load_random_forest():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Random Forest model not found: {MODEL_PATH}"
        )

    print("Loading trained Random Forest...")

    model = joblib.load(MODEL_PATH)

    print("Random Forest loaded successfully.")

    return model


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(data):

    missing = [
        column
        for column in BASE_FEATURES
        + ["gps_bearing", "segment_bearing"]
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}"
        )

    X = data[BASE_FEATURES].copy()

    # Circular encoding of GPS bearing.
    gps_radians = np.radians(data["gps_bearing"])

    X["gps_bearing_sin"] = np.sin(gps_radians)
    X["gps_bearing_cos"] = np.cos(gps_radians)

    # Circular encoding of road-segment bearing.
    segment_radians = np.radians(
        data["segment_bearing"]
    )

    X["segment_bearing_sin"] = np.sin(
        segment_radians
    )

    X["segment_bearing_cos"] = np.cos(
        segment_radians
    )

    return X[MODEL_FEATURES]


# ============================================================
# GENERATE EMISSION PROBABILITIES
# ============================================================

def calculate_emissions(model, feature_data):

    print("\nPreparing candidate features...")

    X = prepare_features(feature_data)

    print("Calculating Random Forest probabilities...")

    probabilities = model.predict_proba(X)

    # Identify the probability column corresponding to class 1.
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "Trained model does not contain positive class 1."
        )

    positive_class_index = classes.index(1)

    raw_probabilities = probabilities[
        :, positive_class_index
    ]

    results = feature_data.copy()

    results["rf_probability"] = raw_probabilities

    # Convert candidate scores into a distribution within
    # each GPS observation. This provides normalized
    # discriminative emission scores for Viterbi.
    group_sum = results.groupby(
        ["trip_id", "gps_index"]
    )["rf_probability"].transform("sum")

    candidate_count = results.groupby(
        ["trip_id", "gps_index"]
    )["arc_id"].transform("count")

    results["emission_probability"] = np.where(
        group_sum > 0,
        results["rf_probability"] / group_sum,
        1.0 / candidate_count
    )

    emission_sums = results.groupby(
        ["trip_id", "gps_index"]
    )["emission_probability"].sum()

    if (
        not np.isfinite(results["emission_probability"]).all()
        or (results["emission_probability"] < 0).any()
        or not np.allclose(emission_sums.to_numpy(), 1.0)
    ):
        raise ValueError(
            "Emission probabilities must be finite, non-negative, "
            "and sum to 1 for every GPS observation."
        )

    # Avoid log(0) during Viterbi decoding.
    results["log_emission"] = np.log(
        np.maximum(
            results["emission_probability"],
            1e-12
        )
    )

    return results


# ============================================================
# CLASSIFIER-ONLY TOP-1 BASELINE
# ============================================================

def calculate_classifier_predictions(emissions):

    ordered = emissions.sort_values(
        [
            "trip_id",
            "gps_index",
            "rf_probability",
            "distance_m"
        ],
        ascending=[
            True,
            True,
            False,
            True
        ]
    )

    predictions = ordered.drop_duplicates(
        subset=["trip_id", "gps_index"]
    ).copy()

    predictions = predictions[
        [
            "trip_id",
            "gps_index",
            "arc_id",
            "rf_probability",
            "distance_m"
        ]
    ]

    predictions = predictions.rename(
        columns={
            "arc_id": "classifier_predicted_arc_id",
            "rf_probability": "classifier_probability",
            "distance_m": "classifier_distance_m"
        }
    )

    return predictions


# ============================================================
# PROCESS ONE TRIP
# ============================================================

def process_trip(trip_id, model=None):

    print("\n" + "=" * 65)
    print(f"RANDOM FOREST EMISSIONS: TRIP {trip_id}")
    print("=" * 65)

    feature_file = (
        PROCESSED_DIR
        / f"trip_{trip_id}_features.csv"
    )

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_file}"
        )

    feature_data = pd.read_csv(feature_file)

    print("Candidate feature shape:", feature_data.shape)

    if model is None:
        model = load_random_forest()

    emissions = calculate_emissions(
        model,
        feature_data
    )

    classifier_predictions = (
        calculate_classifier_predictions(emissions)
    )

    # Save candidate-level emission scores.
    emission_file = (
        OUTPUT_DIR
        / f"trip_{trip_id}_emissions.csv"
    )

    emissions.to_csv(
        emission_file,
        index=False
    )

    # Save classifier-only predictions for later comparison.
    prediction_file = (
        OUTPUT_DIR
        / f"trip_{trip_id}_classifier_predictions.csv"
    )

    classifier_predictions.to_csv(
        prediction_file,
        index=False
    )

    print("\nEmission summary:")

    print(
        emissions[
            [
                "rf_probability",
                "emission_probability"
            ]
        ].describe().round(4)
    )

    print(
        "\nGPS observations:",
        emissions["gps_index"].nunique()
    )

    print(
        "Candidate records:",
        len(emissions)
    )

    print(
        "\nSaved emissions:",
        emission_file
    )

    print(
        "Saved classifier predictions:",
        prediction_file
    )

    return emissions


# ============================================================
# MAIN
# ============================================================

def main():

    # Start with a single trip for verification.
    trip_id = "00000001"

    process_trip(trip_id)

    print("\nEmission probability generation completed.")


if __name__ == "__main__":
    main()
