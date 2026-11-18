
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

path = (
    ROOT
    / "outputs"
    / "hmm_results"
    / "trip_00000001_transitions.csv"
)

df = pd.read_csv(path)

print("\n========== TRANSITION DIAGNOSTICS ==========")

print("\n1. Transition types:")
print(df["transition_type"].value_counts())

print("\n2. Reachability by transition type:")
print(
    pd.crosstab(
        df["transition_type"],
        df["reachable"],
        normalize="index"
    ).round(4)
)

print("\n3. GPS displacement statistics:")
print(
    df["gps_displacement_m"].describe()
)

print("\n4. Network distance statistics (reachable only):")
print(
    df.loc[
        df["reachable"],
        "network_distance_m"
    ].describe()
)

print("\n5. Unreachable transitions by GPS observation:")

unreachable = df[~df["reachable"]]

print(
    unreachable.groupby("current_gps_index")
    .size()
    .describe()
)

print("\n6. Observations with the most unreachable transitions:")

print(
    unreachable.groupby("current_gps_index")
    .size()
    .sort_values(ascending=False)
    .head(15)
)

print("\n7. Same-arc reverse transitions:")
print(
    df[df["transition_type"] == "same_arc_reverse"].shape[0]
)

print("\n8. Reachable transitions with unusually long network distance:")
print(
    df[
        df["reachable"]
        & (df["network_distance_m"] > 500)
    ][
        [
            "previous_gps_index",
            "current_gps_index",
            "network_distance_m",
            "gps_displacement_m",
            "transition_type"
        ]
    ].head(15).to_string(index=False)
)

print("\n============================================")