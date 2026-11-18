
from pathlib import Path
import pandas as pd

if __package__:
    from .candidate_generation import (
        load_trip,
        prepare_road_arcs,
        build_segment_spatial_index,
        generate_candidates_for_point,
    )
    from .feature_extraction import extract_features
else:
    from candidate_generation import (
        load_trip,
        prepare_road_arcs,
        build_segment_spatial_index,
        generate_candidates_for_point,
    )
    from feature_extraction import extract_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_ROOT = Path(
    r"C:\Users\Krishitha\OneDrive\Desktop\Student folder"
)

TRAIN_DIR = DATA_ROOT / "train"
TOP_K = 20


def generate_candidates(trip_id, track, nodes, arcs):

    starts, ends = prepare_road_arcs(nodes, arcs)

    tree, sampled_arc_ids = build_segment_spatial_index(
        starts, ends
    )

    records = []

    for gps_index, gps in track.iterrows():

        point = (
            gps["longitude"],
            gps["latitude"]
        )

        candidates = generate_candidates_for_point(
            point,
            tree,
            sampled_arc_ids,
            starts,
            ends,
            top_k=TOP_K
        )

        for rank, candidate in enumerate(candidates, start=1):

            records.append({
                "trip_id": trip_id,
                "gps_index": gps_index,
                "timestamp": gps["timestamp"],
                "longitude": gps["longitude"],
                "latitude": gps["latitude"],
                "candidate_rank": rank,
                "arc_id": candidate["arc_id"],
                "distance_m": candidate["distance_m"]
            })

        if (gps_index + 1) % 500 == 0:
            print(
                f"  GPS observations: {gps_index + 1}/{len(track)}"
            )

    return pd.DataFrame(records)


def main():

    trip_ids = sorted(
        folder.name
        for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
        and (folder / f"{folder.name}.track").exists()
    )

    print("Training trips found:", len(trip_ids))

    summaries = []

    for number, trip_id in enumerate(trip_ids, start=1):

        print("\n" + "=" * 60)
        print(f"TRIP {number}/{len(trip_ids)}: {trip_id}")
        print("=" * 60)

        try:
            track, nodes, arcs = load_trip(trip_id)

            candidate_file = (
                PROCESSED_DIR / f"trip_{trip_id}_candidates.csv"
            )

            # Reuse existing candidates if available.
            if candidate_file.exists():

                print("Using existing candidate file.")
                candidates = pd.read_csv(candidate_file)

            else:

                print("Generating candidates...")
                candidates = generate_candidates(
                    trip_id, track, nodes, arcs
                )

                candidates.to_csv(
                    candidate_file, index=False
                )

            # Extract features.
            print("Extracting features...")

            features = extract_features(
                trip_id,
                track,
                nodes,
                arcs,
                candidates
            )

            feature_file = (
                PROCESSED_DIR / f"trip_{trip_id}_features.csv"
            )

            features.to_csv(feature_file, index=False)

            gps_count = features["gps_index"].nunique()

            summaries.append({
                "trip_id": trip_id,
                "gps_observations": len(track),
                "candidate_rows": len(candidates),
                "feature_rows": len(features),
                "gps_observations_with_candidates": gps_count,
                "status": "success"
            })

            print("Saved:", feature_file)

        except Exception as error:

            print("ERROR:", error)

            summaries.append({
                "trip_id": trip_id,
                "status": "failed",
                "error": str(error)
            })

    summary = pd.DataFrame(summaries)

    summary.to_csv(
        PROCESSED_DIR / "batch_processing_summary.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)

    print(summary.to_string(index=False))
    print("\nSuccessful trips:", (summary["status"] == "success").sum())
    print("Failed trips:", (summary["status"] == "failed").sum())


if __name__ == "__main__":
    main()
