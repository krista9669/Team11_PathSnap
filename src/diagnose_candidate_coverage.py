
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

for trip_id in ["00000033", "00000054"]:

    candidates = pd.read_csv(DATA / f"trip_{trip_id}_candidates.csv")
    alignment = pd.read_csv(DATA / f"trip_{trip_id}_monotonic_alignment.csv")

    # Find the aligned road segment for each GPS observation.
    merged = candidates.merge(
        alignment[
            ["gps_index", "aligned_arc_id",
             "distance_to_aligned_arc_m", "timestamp"]
        ],
        on="gps_index",
        how="inner",
        suffixes=("_candidate", "_alignment")
    )

    merged["is_match"] = (
        merged["arc_id"].astype(int)
        == merged["aligned_arc_id"].astype(int)
    )

    matched_indices = set(
        merged.loc[merged["is_match"], "gps_index"]
    )

    observation_info = alignment.copy()
    observation_info["covered"] = (
        observation_info["gps_index"].isin(matched_indices)
    )

    # Candidate distance statistics for each GPS observation.
    candidate_stats = candidates.groupby("gps_index").agg(
        candidate_count=("arc_id", "count"),
        nearest_candidate_distance=("distance_m", "min"),
        farthest_candidate_distance=("distance_m", "max")
    ).reset_index()

    observation_info = observation_info.merge(
        candidate_stats, on="gps_index", how="left"
    )

    print(f"\n========== TRIP {trip_id} ==========")
    print(f"Observations: {len(observation_info)}")
    print(f"Covered: {observation_info['covered'].sum()}")
    print(f"Uncovered: {(~observation_info['covered']).sum()}")

    for label, subset in [
        ("MATCHED", observation_info[observation_info["covered"]]),
        ("UNMATCHED", observation_info[~observation_info["covered"]])
    ]:
        print(f"\n{label} observations:")
        print(
            subset[
                ["distance_to_aligned_arc_m",
                 "nearest_candidate_distance",
                 "farthest_candidate_distance",
                 "candidate_count"]
            ].describe().round(2)
        )

    print("\nFirst 15 uncovered observations:")
    print(
        observation_info.loc[
            ~observation_info["covered"],
            ["gps_index", "timestamp", "aligned_arc_id",
             "distance_to_aligned_arc_m",
             "nearest_candidate_distance", "candidate_count"]
        ].head(15).to_string(index=False)
    )
