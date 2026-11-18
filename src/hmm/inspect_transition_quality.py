
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

PATH = (
    ROOT / "outputs" / "hmm_results"
    / "trip_00000001_transitions.csv"
)

df = pd.read_csv(PATH)

# Each previous GPS observation + previous candidate arc
# represents one transition-matrix row.
keys = ["previous_gps_index", "previous_arc_id"]

rows = (
    df.groupby(keys)
    .agg(
        has_reachable=("reachable", "any"),
        reachable_candidates=("reachable", "sum"),
        total_candidates=("reachable", "size"),
    )
    .reset_index()
)

fallback = rows[~rows["has_reachable"]]

print("\n========== TRANSITION QUALITY ==========")

print("\n1. Previous candidate rows:")
print(f"Total rows: {len(rows):,}")
print(f"Rows with a reachable next candidate: {rows['has_reachable'].sum():,}")
print(f"Fallback rows: {len(fallback):,}")

print("\n2. Reachable next candidates per previous candidate:")
print(rows["reachable_candidates"].describe())

print("\n3. Observations with the most fallback candidates:")
print(
    fallback.groupby("previous_gps_index")
    .size()
    .sort_values(ascending=False)
    .head(20)
)

print("\n4. Fallback percentage by GPS observation:")
fallback_by_gps = (
    rows.groupby("previous_gps_index")
    .agg(
        candidate_rows=("has_reachable", "size"),
        fallback_rows=("has_reachable", lambda x: (~x).sum())
    )
)

fallback_by_gps["fallback_percentage"] = (
    100 * fallback_by_gps["fallback_rows"]
    / fallback_by_gps["candidate_rows"]
)

print(
    fallback_by_gps.sort_values(
        "fallback_percentage",
        ascending=False
    ).head(20)
)

# Evaluate the best distance match among reachable transitions
reachable = df[df["reachable"]].copy()

reachable["distance_error_m"] = (
    reachable["network_distance_m"]
    - reachable["gps_displacement_m"]
).abs()

best = (
    reachable.groupby(keys)
    .agg(
        best_distance_error_m=("distance_error_m", "min"),
        best_network_distance_m=("network_distance_m", "first"),
        gps_displacement_m=("gps_displacement_m", "first")
    )
    .reset_index()
)

print("\n5. Best reachable distance mismatch per previous candidate:")
print(best["best_distance_error_m"].describe())

print("\n6. Percentage of rows with at least one reachable transition:")
for threshold in [25, 50, 100, 200]:
    percentage = (
        100 * (best["best_distance_error_m"] <= threshold).mean()
    )
    print(f"Within {threshold:3d} m: {percentage:.2f}%")

print("\n7. Worst GPS observations by best possible distance mismatch:")
worst = (
    best.groupby("previous_gps_index")
    ["best_distance_error_m"]
    .median()
    .sort_values(ascending=False)
    .head(20)
)

print(worst)

print("\n========================================")