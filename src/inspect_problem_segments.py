
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(r"C:\Users\Krishitha\OneDrive\Desktop\Student folder")

def inspect_trip(trip_id, start, end):

    trip_dir = DATA / "train" / trip_id
    processed = ROOT / "data" / "processed"

    # Load original road network.
    nodes = pd.read_csv(
        trip_dir / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        trip_dir / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"]
    )

    track = pd.read_csv(
        trip_dir / f"{trip_id}.track",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    candidates = pd.read_csv(
        processed / f"trip_{trip_id}_candidates.csv"
    )

    alignment = pd.read_csv(
        processed / f"trip_{trip_id}_monotonic_alignment.csv"
    )

    print(f"\n{'=' * 75}")
    print(f"TRIP {trip_id}: GPS INDICES {start}–{end}")
    print(f"{'=' * 75}")

    for gps_index in range(start, end + 1):

        gps = track.iloc[gps_index]
        aligned = alignment[
            alignment["gps_index"] == gps_index
        ]

        if aligned.empty:
            continue

        aligned = aligned.iloc[0]
        aligned_arc_id = int(aligned["aligned_arc_id"])

        def get_arc_coordinates(arc_id):
            arc = arcs.iloc[arc_id]

            from_node = nodes.iloc[int(arc["from_node"])]
            to_node = nodes.iloc[int(arc["to_node"])]

            return (
                (from_node["longitude"], from_node["latitude"]),
                (to_node["longitude"], to_node["latitude"])
            )

        aligned_coords = get_arc_coordinates(aligned_arc_id)

        top_candidates = candidates[
            candidates["gps_index"] == gps_index
        ].sort_values("candidate_rank").head(5)

        print(f"\nGPS index: {gps_index}")
        print(
            f"GPS: ({gps['longitude']:.6f}, "
            f"{gps['latitude']:.6f})"
        )
        print(
            f"Aligned arc: {aligned_arc_id} | "
            f"Route position: {aligned['aligned_route_position']} | "
            f"Distance: {aligned['distance_to_aligned_arc_m']:.2f} m"
        )
        print(f"Aligned endpoints: {aligned_coords}")

        print("Top 5 candidate arcs:")
        for _, candidate in top_candidates.iterrows():
            arc_id = int(candidate["arc_id"])
            coords = get_arc_coordinates(arc_id)

            print(
                f"  Rank {int(candidate['candidate_rank'])}: "
                f"arc {arc_id}, "
                f"distance {candidate['distance_m']:.2f} m, "
                f"endpoints {coords}"
            )


inspect_trip("00000033", 631, 645)
inspect_trip("00000054", 1425, 1440)