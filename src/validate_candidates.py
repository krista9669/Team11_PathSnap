
from pathlib import Path
import pandas as pd

DATA_ROOT = Path(r"C:\Users\Krishitha\OneDrive\Desktop\Student folder")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_DIR = DATA_ROOT / "train"

# Identify all expected training trips from the raw dataset.
expected_trip_ids = {
    folder.name
    for folder in TRAIN_DIR.iterdir()
    if folder.is_dir()
}

candidate_files = {
    path.stem.replace("trip_", "").replace("_candidates", ""): path
    for path in PROCESSED_DIR.glob("trip_*_candidates.csv")
}

alignment_files = {
    path.stem.replace("trip_", "").replace("_monotonic_alignment", ""): path
    for path in PROCESSED_DIR.glob("trip_*_monotonic_alignment.csv")
}

results = []
missing_files = []
uncovered_observations = []

for trip_id in sorted(expected_trip_ids):

    candidate_path = candidate_files.get(trip_id)
    alignment_path = alignment_files.get(trip_id)

    if candidate_path is None or alignment_path is None:
        missing_files.append({
            "trip_id": trip_id,
            "missing_candidates": candidate_path is None,
            "missing_alignment": alignment_path is None
        })
        continue

    candidates = pd.read_csv(
        candidate_path,
        dtype={"trip_id": str}
    )

    alignment = pd.read_csv(
        alignment_path,
        dtype={"trip_id": str}
    )

    # Keep one aligned road segment per GPS observation.
    aligned = alignment[
        ["trip_id", "gps_index", "aligned_arc_id"]
    ].drop_duplicates(
        subset=["trip_id", "gps_index"]
    )

    # Match each candidate list with its corresponding GPS observation.
    merged = candidates.merge(
        aligned,
        on=["trip_id", "gps_index"],
        how="inner"
    )

    merged["is_match"] = (
        merged["arc_id"].astype(int)
        == merged["aligned_arc_id"].astype(int)
    )

    # An observation is covered if its aligned arc is in its candidate list.
    matched_observations = (
        merged[merged["is_match"]]
        ["gps_index"]
        .nunique()
    )

    total_observations = aligned["gps_index"].nunique()
    missing_observations = total_observations - matched_observations

    coverage = (
        matched_observations / total_observations * 100
        if total_observations > 0 else 0
    )

    results.append({
        "trip_id": trip_id,
        "total_observations": total_observations,
        "matched_observations": matched_observations,
        "missing_observations": missing_observations,
        "top20_coverage_percent": round(coverage, 2)
    })

    # Record observations whose aligned arc was not among their candidates.
    matched_indices = set(
        merged.loc[merged["is_match"], "gps_index"].unique()
    )

    for _, row in aligned.iterrows():
        if row["gps_index"] not in matched_indices:
            uncovered_observations.append({
                "trip_id": trip_id,
                "gps_index": row["gps_index"],
                "aligned_arc_id": row["aligned_arc_id"]
            })


# Save reports.
summary = pd.DataFrame(results)
missing = pd.DataFrame(missing_files)
uncovered = pd.DataFrame(uncovered_observations)

summary_path = PROCESSED_DIR / "all_training_candidate_coverage.csv"
missing_path = PROCESSED_DIR / "missing_training_outputs.csv"
uncovered_path = PROCESSED_DIR / "uncovered_training_observations.csv"

summary.to_csv(summary_path, index=False)
missing.to_csv(missing_path, index=False)
uncovered.to_csv(uncovered_path, index=False)

print("\n========== TOP-20 CANDIDATE COVERAGE ==========")
print(f"Expected training trips: {len(expected_trip_ids)}")
print(f"Trips successfully checked: {len(results)}")
print(f"Trips with missing files: {len(missing_files)}")

if not summary.empty:
    print(f"\nTotal GPS observations: {summary['total_observations'].sum()}")
    print(f"Matched observations: {summary['matched_observations'].sum()}")
    print(f"Uncovered observations: {summary['missing_observations'].sum()}")

    overall = (
        summary["matched_observations"].sum()
        / summary["total_observations"].sum()
        * 100
    )

    print(f"\nOverall Top-20 coverage: {overall:.2f}%")
    print(f"Average trip coverage: {summary['top20_coverage_percent'].mean():.2f}%")

    print("\nLowest coverage trips:")
    print(
        summary.sort_values("top20_coverage_percent")
        .head(10)
        .to_string(index=False)
    )

if not missing.empty:
    print("\nTrips with missing outputs:")
    print(missing.to_string(index=False))

print("\nReports saved:")
print(summary_path)
print(missing_path)
print(uncovered_path)
