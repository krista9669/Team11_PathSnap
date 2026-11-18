import numpy as np
import pandas as pd
import pytest

from src.hmm.emission_probabilities import calculate_emissions
from src.hmm.transition_probabilities import (
    UNREACHABLE_PROBABILITY,
    build_transition_matrices,
    calculate_transition_probability,
)
from src.hmm.viterbi import viterbi_decode


class FixedProbabilityModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        return np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])


def make_features():
    rows = []
    for gps_index, arc_id in [(0, 10), (0, 11), (1, 10)]:
        rows.append({
            "trip_id": "00000005",
            "gps_index": gps_index,
            "arc_id": arc_id,
            "distance_m": 2.0,
            "gps_speed_mps": 5.0,
            "gps_displacement_m": 10.0,
            "time_gap_s": 2.0,
            "segment_length_m": 30.0,
            "heading_difference": 0.1,
            "projection_fraction": 0.5,
            "gps_bearing": 0.2,
            "segment_bearing": 0.3,
        })
    return pd.DataFrame(rows)


def test_emissions_are_normalized_per_observation():
    emissions = calculate_emissions(FixedProbabilityModel(), make_features())

    sums = emissions.groupby("gps_index")["emission_probability"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)
    assert np.isfinite(emissions["emission_probability"]).all()
    assert emissions["emission_probability"].between(0, 1).all()


def test_transition_matrices_match_candidate_dimensions_and_penalize_unreachable():
    emissions = make_features()
    emissions["emission_probability"] = [0.6, 0.4, 1.0]
    arc_information = {
        10: {"start_node": 0, "end_node": 1, "length": 100.0},
        11: {"start_node": 2, "end_node": 3, "length": 100.0},
    }

    def shortest_path_distance(source, target):
        return 0.0 if source == target else float("inf")

    groups, matrices, _, _, _, _ = build_transition_matrices(
        emissions,
        arc_information,
        shortest_path_distance,
    )

    assert len(matrices) == len(groups) - 1 == 1
    assert matrices[0].shape == (2, 1)
    assert matrices[0][0, 0] > UNREACHABLE_PROBABILITY
    assert matrices[0][1, 0] == UNREACHABLE_PROBABILITY
    assert calculate_transition_probability(
        float("inf"), 10, 1, "unreachable"
    ) == UNREACHABLE_PROBABILITY


def test_viterbi_returns_one_candidate_per_observation():
    groups = [
        pd.DataFrame({
            "gps_index": [0, 0],
            "arc_id": [20, 21],
            "emission_probability": [0.6, 0.4],
        }),
        pd.DataFrame({
            "gps_index": [1, 1],
            "arc_id": [20, 21],
            "emission_probability": [0.1, 0.9],
        }),
    ]

    path, _ = viterbi_decode(
        groups,
        [np.array([[0.99, 0.01], [0.01, 0.99]])],
    )

    assert len(path) == len(groups)
    assert path == [1, 1]


def test_viterbi_rejects_invalid_transition_probabilities():
    groups = [
        pd.DataFrame({"emission_probability": [1.0]}),
        pd.DataFrame({"emission_probability": [1.0]}),
    ]

    with pytest.raises(ValueError, match="invalid probabilities"):
        viterbi_decode(groups, [np.array([[-0.1]])])
