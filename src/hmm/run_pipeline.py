"""Run HMM map matching on selected training or student-test trips."""

from .. import batch_alignment
import argparse
import gc
import time
from pathlib import Path

import pandas as pd

from .. import batch_alignment
from .. import batch_feature_extraction
from .. import candidate_generation
from .. import feature_extraction
from . import emission_probabilities
from . import evaluate_viterbi
from . import transition_probabilities
from . import viterbi


ROOT = Path(__file__).resolve().parents[2]
HMM_DIR = ROOT / "outputs" / "hmm_results"
EVALUATION_DIR = HMM_DIR / "evaluation"
SUMMARY_PATH = EVALUATION_DIR / "pilot_batch_summary.csv"
PILOT_TRIPS = (5, 13, 16)
STUDENT_TEST_TRIPS = (0, 7, 9, 40, 52, 56, 65, 76, 77, 91)
STUDENT_TEST_DIR = (
    Path(r"C:\Users\Krishitha\OneDrive\Desktop\Student folder")
    / "student_test"
)
PROCESSED_DIR = ROOT / "data" / "processed"


def _trip_output_files(trip_id):
    trip_name = f"{trip_id:08d}"
    return [
        HMM_DIR / f"trip_{trip_name}_emissions.csv",
        HMM_DIR / f"trip_{trip_name}_transitions.csv",
        HMM_DIR / f"trip_{trip_name}_transition_matrices.npz",
        HMM_DIR / f"trip_{trip_name}_viterbi_predictions.csv",
        HMM_DIR / f"trip_{trip_name}_viterbi_transition_quality.csv",
        EVALUATION_DIR / f"trip_{trip_name}_evaluation_summary.csv",
        EVALUATION_DIR / f"trip_{trip_name}_rf_vs_viterbi_comparison.csv",
    ]


def prepare_student_test_trip(trip_id):
    trip_name = f"{trip_id:08d}"
    track, nodes, arcs = candidate_generation.load_trip(
        trip_name,
        data_dir=STUDENT_TEST_DIR,
    )

    candidates = batch_feature_extraction.generate_candidates(
        trip_name,
        track,
        nodes,
        arcs,
    )
    candidate_path = PROCESSED_DIR / f"trip_{trip_name}_candidates.csv"
    candidates.to_csv(candidate_path, index=False)

    features = feature_extraction.extract_features(
        trip_name,
        track,
        nodes,
        arcs,
        candidates,
    )
    feature_path = PROCESSED_DIR / f"trip_{trip_name}_features.csv"
    features.to_csv(feature_path, index=False)
    del candidates, features

    batch_alignment.process_trip(
        trip_name,
        data_dir=STUDENT_TEST_DIR,
    )


def process_trip(trip_id, model, model_load_seconds, partition="train"):
    trip_name = f"{trip_id:08d}"
    started = time.perf_counter()

    emissions = emission_probabilities.process_trip(
        trip_name,
        model=model,
    )
    observation_count = emissions["gps_index"].nunique()
    candidate_count = len(emissions)
    del emissions

    transition_probabilities.main(trip_id, partition=partition)
    viterbi.main(trip_id)
    summary, comparison = evaluate_viterbi.evaluate_trip(trip_id)

    prediction_path = (
        HMM_DIR / f"trip_{trip_name}_viterbi_predictions.csv"
    )
    predictions = pd.read_csv(prediction_path, usecols=["gps_index"])
    if len(predictions) != observation_count:
        raise ValueError(
            f"Trip {trip_name}: expected {observation_count} predictions "
            f"for GPS observations, got {len(predictions)}."
        )

    quality_path = (
        HMM_DIR / f"trip_{trip_name}_viterbi_transition_quality.csv"
    )
    quality = pd.read_csv(quality_path)
    if len(quality) != max(0, observation_count - 1):
        raise ValueError(
            f"Trip {trip_name}: expected {max(0, observation_count - 1)} "
            f"transition diagnostics, got {len(quality)}."
        )

    selected_reachability = (
        float(quality["reachable"].astype(bool).mean())
        if not quality.empty
        else float("nan")
    )
    runtime_seconds = time.perf_counter() - started
    output_bytes = sum(path.stat().st_size for path in _trip_output_files(trip_id))

    summary["candidate_records"] = candidate_count
    summary["prediction_observations"] = len(predictions)
    summary["selected_transition_reachability"] = selected_reachability
    summary["runtime_seconds"] = runtime_seconds
    summary["model_load_seconds"] = model_load_seconds
    summary["output_size_mb"] = output_bytes / (1024 * 1024)

    print(
        f"Trip {trip_name}: {observation_count:,} observations, "
        f"{candidate_count:,} candidates, "
        f"{runtime_seconds:.1f}s, "
        f"{output_bytes / (1024 * 1024):.1f} MiB output."
    )

    return summary, comparison, quality


def build_aggregate_summary(comparisons, qualities, partition):
    comparison = pd.concat(comparisons, ignore_index=True)
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

    # Arc IDs are only unique within a trip's road network.
    y_true = valid["trip_id"] + ":" + valid["aligned_arc_id"].astype(str)
    rf_pred = valid["trip_id"] + ":" + valid[
        "classifier_top1_arc_id"
    ].astype(str)
    hmm_pred = valid["trip_id"] + ":" + valid["viterbi_arc_id"].astype(str)

    rows = []
    for method, y_pred, distance_column in [
        ("Random Forest Top-1", rf_pred, "classifier_top1_distance_m"),
        ("HMM Viterbi", hmm_pred, "distance_m"),
    ]:
        metrics = evaluate_viterbi.calculate_metrics(y_true, y_pred)
        distance = evaluate_viterbi.distance_statistics(
            valid[distance_column]
        )
        reference = evaluate_viterbi.distance_statistics(
            valid["distance_to_aligned_arc_m"]
        )
        rows.append({
            "trip_id": f"ALL_{partition.upper()}",
            "method": method,
            "evaluation_observations": len(valid),
            **metrics,
            "mean_selected_distance_m": distance["mean"],
            "median_selected_distance_m": distance["median"],
            "reference_mean_distance_m": reference["mean"],
            "reference_median_distance_m": reference["median"],
            "prediction_changes": int(
                (valid["classifier_top1_arc_id"] != valid["viterbi_arc_id"]).sum()
            ),
            "prediction_change_percentage": 100 * (
                valid["classifier_top1_arc_id"] != valid["viterbi_arc_id"]
            ).mean(),
            "viterbi_improved_count": int(
                ((valid["classifier_top1_arc_id"] != valid["aligned_arc_id"])
                 & (valid["viterbi_arc_id"] == valid["aligned_arc_id"])).sum()
            ),
            "viterbi_worsened_count": int(
                ((valid["classifier_top1_arc_id"] == valid["aligned_arc_id"])
                 & (valid["viterbi_arc_id"] != valid["aligned_arc_id"])).sum()
            ),
            "both_correct_count": int(
                ((valid["classifier_top1_arc_id"] == valid["aligned_arc_id"])
                 & (valid["viterbi_arc_id"] == valid["aligned_arc_id"])).sum()
            ),
            "both_wrong_count": int(
                ((valid["classifier_top1_arc_id"] != valid["aligned_arc_id"])
                 & (valid["viterbi_arc_id"] != valid["aligned_arc_id"])).sum()
            ),
            "selected_transition_reachability": (
                sum(int(item["reachable"].astype(bool).sum()) for item in qualities)
                / sum(len(item) for item in qualities)
                if sum(len(item) for item in qualities)
                else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def run(trip_ids, partition="train"):
    HMM_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    if partition == "student_test":
        unexpected = set(trip_ids) - set(STUDENT_TEST_TRIPS)
        if unexpected:
            raise ValueError(
                "Only designated student-test trip IDs can be processed "
                f"from this partition: {sorted(unexpected)}"
            )

    model_started = time.perf_counter()
    model = emission_probabilities.load_random_forest()
    model_load_seconds = time.perf_counter() - model_started

    summaries = []
    comparisons = []
    qualities = []
    for trip_id in trip_ids:
        trip_started = time.perf_counter()
        preparation_seconds = 0.0
        if partition == "student_test":
            preparation_started = time.perf_counter()
            prepare_student_test_trip(trip_id)
            preparation_seconds = (
                time.perf_counter() - preparation_started
            )
        summary, comparison, quality = process_trip(
            trip_id,
            model,
            model_load_seconds,
            partition=partition,
        )
        summary["input_preparation_seconds"] = preparation_seconds
        summary["runtime_seconds"] = time.perf_counter() - trip_started
        summaries.append(summary)
        comparisons.append(comparison)
        qualities.append(quality)
        gc.collect()

    per_trip_summary = pd.concat(summaries, ignore_index=True)
    aggregate_summary = build_aggregate_summary(
        comparisons,
        qualities,
        partition,
    )
    summary_path = (
        EVALUATION_DIR / "student_test_hmm_summary.csv"
        if partition == "student_test"
        else SUMMARY_PATH
    )
    result = pd.concat(
        [per_trip_summary, aggregate_summary],
        ignore_index=True,
    )
    result.to_csv(summary_path, index=False)

    print(f"\n{partition.replace('_', ' ').title()} batch summary:")
    print(per_trip_summary[
        [
            "trip_id",
            "method",
            "evaluation_observations",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "selected_transition_reachability",
            "runtime_seconds",
            "output_size_mb",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nAggregate summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trip-ids",
        nargs="+",
        type=int,
        default=None,
        help="Trip IDs to process (partition-specific defaults are used).",
    )
    parser.add_argument(
        "--partition",
        choices=["train", "student_test"],
        default="train",
        help="Use training data (pilot) or the designated student-test trips.",
    )
    args = parser.parse_args()
    trip_ids = args.trip_ids
    if trip_ids is None:
        trip_ids = (
            STUDENT_TEST_TRIPS
            if args.partition == "student_test"
            else PILOT_TRIPS
        )
    run(trip_ids, partition=args.partition)


if __name__ == "__main__":
    main()
