
"""
Evaluate Random Forest Top-1 predictions against HMM/Viterbi predictions.

The reference labels come from the monotonic alignment CSV. They are
provisional alignment labels, not independently verified ground truth.

Run from the project root:
    python src/hmm/evaluate_viterbi.py
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
HMM_DIR = PROJECT_ROOT / "outputs" / "hmm_results"

OUTPUT_DIR = HMM_DIR / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def normalize_trip_id(value):
    """Convert trip IDs to a consistent eight-digit string."""
    if pd.isna(value):
        return None

    return str(value).split(".")[0].zfill(8)


def calculate_metrics(y_true, y_pred):
    """
    Calculate multiclass arc-prediction metrics.

    Macro metrics give each arc equal importance.
    Weighted metrics account for the frequency of each arc.
    """

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def distance_statistics(series):
    """Return mean and median distance, ignoring missing values."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return {
            "mean": float("nan"),
            "median": float("nan"),
        }

    return {
        "mean": values.mean(),
        "median": values.median(),
    }


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def evaluate_trip(trip_id):

    trip_name = f"{int(trip_id):08d}"
    alignment_file = (
        DATA_DIR / f"trip_{trip_name}_monotonic_alignment.csv"
    )
    prediction_file = (
        HMM_DIR / f"trip_{trip_name}_viterbi_predictions.csv"
    )

    print("=" * 65)
    print("PATHSNAP - RANDOM FOREST VS HMM/VITERBI EVALUATION")
    print("=" * 65)

    if not alignment_file.exists():
        raise FileNotFoundError(
            f"Alignment file not found: {alignment_file}"
        )

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Viterbi prediction file not found: {prediction_file}"
        )

    alignment = pd.read_csv(alignment_file)
    predictions = pd.read_csv(prediction_file)

    print("\nLoaded files successfully.")
    print(f"Alignment rows: {len(alignment):,}")
    print(f"Prediction rows: {len(predictions):,}")

    # -----------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # -----------------------------------------------------

    alignment_required = {
        "trip_id",
        "gps_index",
        "aligned_arc_id",
        "distance_to_aligned_arc_m",
    }

    prediction_required = {
        "trip_id",
        "gps_index",
        "classifier_top1_arc_id",
        "viterbi_arc_id",
        "classifier_top1_distance_m",
        "distance_m",
        "prediction_changed",
    }

    missing_alignment = alignment_required - set(alignment.columns)
    missing_predictions = prediction_required - set(predictions.columns)

    if missing_alignment:
        raise ValueError(
            f"Missing alignment columns: {sorted(missing_alignment)}"
        )

    if missing_predictions:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing_predictions)}"
        )

    # -----------------------------------------------------
    # NORMALIZE KEYS AND NUMERIC COLUMNS
    # -----------------------------------------------------

    alignment["trip_id"] = alignment["trip_id"].apply(
        normalize_trip_id
    )

    predictions["trip_id"] = predictions["trip_id"].apply(
        normalize_trip_id
    )

    numeric_columns = [
        "gps_index",
        "aligned_arc_id",
        "distance_to_aligned_arc_m",
    ]

    for column in numeric_columns:
        alignment[column] = pd.to_numeric(
            alignment[column],
            errors="coerce",
        )

    prediction_numeric_columns = [
        "gps_index",
        "classifier_top1_arc_id",
        "viterbi_arc_id",
        "classifier_top1_distance_m",
        "distance_m",
    ]

    for column in prediction_numeric_columns:
        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="coerce",
        )

    # Keep only the requested trip.
    alignment = alignment[
        alignment["trip_id"] == trip_name
    ].copy()

    predictions = predictions[
        predictions["trip_id"] == trip_name
    ].copy()

    if alignment.empty or predictions.empty:
        raise ValueError(
            f"No alignment or prediction rows found for Trip {trip_name}."
        )

    # -----------------------------------------------------
    # CHECK FOR DUPLICATE OBSERVATIONS
    # -----------------------------------------------------

    if alignment.duplicated(["trip_id", "gps_index"]).any():
        raise ValueError(
            "Duplicate trip_id/gps_index keys in alignment file."
        )

    if predictions.duplicated(["trip_id", "gps_index"]).any():
        raise ValueError(
            "Duplicate trip_id/gps_index keys in prediction file."
        )

    # -----------------------------------------------------
    # MERGE ALIGNMENT LABELS WITH PREDICTIONS
    # -----------------------------------------------------

    comparison = alignment.merge(
        predictions,
        on=["trip_id", "gps_index"],
        how="inner",
        validate="one_to_one",
    )

    print("\nMerge completed.")
    print(f"Matched observations: {len(comparison):,}")

    if comparison.empty:
        raise ValueError(
            "No matching GPS observations between the two files."
        )

    if len(comparison) != len(alignment) or len(comparison) != len(predictions):
        raise ValueError(
            "Alignment and prediction observations do not match one-to-one: "
            f"{len(alignment)} alignment rows, {len(predictions)} prediction "
            f"rows, {len(comparison)} matched rows."
        )

    # Remove observations without a reference or either prediction.
    valid = comparison.dropna(
        subset=[
            "aligned_arc_id",
            "classifier_top1_arc_id",
            "viterbi_arc_id",
        ]
    ).copy()

    valid["aligned_arc_id"] = valid["aligned_arc_id"].astype(int)
    valid["classifier_top1_arc_id"] = (
        valid["classifier_top1_arc_id"].astype(int)
    )
    valid["viterbi_arc_id"] = valid["viterbi_arc_id"].astype(int)

    print(f"Valid evaluation observations: {len(valid):,}")

    if valid.empty:
        raise ValueError("No valid observations available for evaluation.")

    # -----------------------------------------------------
    # CALCULATE ARC-PREDICTION METRICS
    # -----------------------------------------------------

    y_true = valid["aligned_arc_id"]

    rf_predictions = valid["classifier_top1_arc_id"]
    hmm_predictions = valid["viterbi_arc_id"]

    rf_metrics = calculate_metrics(
        y_true,
        rf_predictions,
    )

    hmm_metrics = calculate_metrics(
        y_true,
        hmm_predictions,
    )

    # -----------------------------------------------------
    # CALCULATE GEOMETRIC DISTANCE STATISTICS
    # -----------------------------------------------------

    rf_distance = distance_statistics(
        valid["classifier_top1_distance_m"]
    )

    hmm_distance = distance_statistics(
        valid["distance_m"]
    )

    reference_distance = distance_statistics(
        valid["distance_to_aligned_arc_m"]
    )

    # -----------------------------------------------------
    # PREDICTION CHANGE ANALYSIS
    # -----------------------------------------------------

    prediction_changed = (
        rf_predictions != hmm_predictions
    )

    changed_count = int(prediction_changed.sum())

    changed_percentage = (
        changed_count / len(valid)
    ) * 100

    rf_correct = rf_predictions == y_true
    hmm_correct = hmm_predictions == y_true

    # Cases where Viterbi corrected or changed a correct RF prediction.
    rf_wrong_hmm_correct = int(
        ((~rf_correct) & hmm_correct).sum()
    )

    rf_correct_hmm_wrong = int(
        (rf_correct & (~hmm_correct)).sum()
    )

    both_correct = int(
        (rf_correct & hmm_correct).sum()
    )

    both_wrong = int(
        ((~rf_correct) & (~hmm_correct)).sum()
    )

    # -----------------------------------------------------
    # BUILD SUMMARY
    # -----------------------------------------------------

    summary_rows = []

    for method, metrics, distances in [
        ("Random Forest Top-1", rf_metrics, rf_distance),
        ("HMM Viterbi", hmm_metrics, hmm_distance),
    ]:

        row = {
            "trip_id": trip_name,
            "method": method,
            "evaluation_observations": len(valid),
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_f1": metrics["weighted_f1"],
            "mean_selected_distance_m": distances["mean"],
            "median_selected_distance_m": distances["median"],
            "reference_mean_distance_m": reference_distance["mean"],
            "reference_median_distance_m": reference_distance["median"],
            "prediction_changes": changed_count,
            "prediction_change_percentage": changed_percentage,
            "viterbi_improved_count": rf_wrong_hmm_correct,
            "viterbi_worsened_count": rf_correct_hmm_wrong,
            "both_correct_count": both_correct,
            "both_wrong_count": both_wrong,
        }

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # -----------------------------------------------------
    # SAVE PER-OBSERVATION COMPARISON
    # -----------------------------------------------------

    comparison["rf_correct"] = (
        comparison["classifier_top1_arc_id"]
        == comparison["aligned_arc_id"]
    )

    comparison["viterbi_correct"] = (
        comparison["viterbi_arc_id"]
        == comparison["aligned_arc_id"]
    )

    comparison["rf_vs_viterbi_same_arc"] = (
        comparison["classifier_top1_arc_id"]
        == comparison["viterbi_arc_id"]
    )

    comparison["viterbi_improved_over_rf"] = (
        (~comparison["rf_correct"])
        & comparison["viterbi_correct"]
    )

    comparison["viterbi_worsened_over_rf"] = (
        comparison["rf_correct"]
        & (~comparison["viterbi_correct"])
    )

    comparison_file = (
        OUTPUT_DIR
        / f"trip_{trip_name}_rf_vs_viterbi_comparison.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False,
    )

    # -----------------------------------------------------
    # SAVE SUMMARY
    # -----------------------------------------------------

    summary_file = (
        OUTPUT_DIR
        / f"trip_{trip_name}_evaluation_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)

    print(f"\nTrip ID: {trip_name}")
    print(f"Alignment observations: {len(alignment):,}")
    print(f"Prediction observations: {len(predictions):,}")
    print(f"Matched observations: {len(comparison):,}")
    print(f"Valid evaluation observations: {len(valid):,}")

    print("\n--- ARC PREDICTION METRICS ---")

    display_columns = [
        "method",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n--- SELECTED ROAD DISTANCE ---")

    print(
        f"Random Forest mean:   {rf_distance['mean']:.3f} m"
    )
    print(
        f"Random Forest median: {rf_distance['median']:.3f} m"
    )

    print(
        f"HMM mean:             {hmm_distance['mean']:.3f} m"
    )
    print(
        f"HMM median:           {hmm_distance['median']:.3f} m"
    )

    print(
        f"Reference mean:       {reference_distance['mean']:.3f} m"
    )
    print(
        f"Reference median:     {reference_distance['median']:.3f} m"
    )

    print("\n--- VITERBI CHANGE ANALYSIS ---")

    print(
        f"Predictions changed: {changed_count:,}/{len(valid):,}"
    )

    print(
        f"Change percentage: {changed_percentage:.2f}%"
    )

    print(
        f"RF wrong -> Viterbi correct: {rf_wrong_hmm_correct:,}"
    )

    print(
        f"RF correct -> Viterbi wrong: {rf_correct_hmm_wrong:,}"
    )

    print(f"Both correct: {both_correct:,}")
    print(f"Both wrong:   {both_wrong:,}")

    print("\n--- OUTPUT FILES ---")

    print(f"Comparison: {comparison_file}")
    print(f"Summary:    {summary_file}")

    print("\nEvaluation completed successfully.")
    print("=" * 65)

    return summary, comparison


if __name__ == "__main__":
    evaluate_trip(1)