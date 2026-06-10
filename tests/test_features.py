import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import compute_features_for_prediction


def test_retourne_12_features():
    result = compute_features_for_prediction(
        -119.6, 35.6, 29, 1425, 3.87, 5.4, 1.1, 3.1, '<1H OCEAN'
    )
    assert len(result) == 12


def test_rooms_per_bedroom_index_8():
    # avg_rooms=6.0, avg_bedrooms=2.0 → rooms_per_bedroom = 3.0 (index 8)
    result = compute_features_for_prediction(
        -119.6, 35.6, 29, 1425, 3.87, 6.0, 2.0, 3.1, '<1H OCEAN'
    )
    assert abs(result[8] - 3.0) < 1e-9


def test_ocean_proximity_reference_all_zeros():
    # <1H OCEAN est la référence — les 3 dummies doivent être 0
    result = compute_features_for_prediction(
        -119.6, 35.6, 29, 1425, 3.87, 5.4, 1.1, 3.1, '<1H OCEAN'
    )
    assert result[9] == 0   # ocean_proximity_INLAND
    assert result[10] == 0  # ocean_proximity_NEAR BAY
    assert result[11] == 0  # ocean_proximity_NEAR OCEAN


def test_ocean_proximity_inland():
    result = compute_features_for_prediction(
        -119.6, 35.6, 29, 1425, 3.87, 5.4, 1.1, 3.1, 'INLAND'
    )
    assert result[9] == 1   # ocean_proximity_INLAND
    assert result[10] == 0
    assert result[11] == 0


def test_ocean_proximity_near_bay():
    result = compute_features_for_prediction(
        -119.6, 35.6, 29, 1425, 3.87, 5.4, 1.1, 3.1, 'NEAR BAY'
    )
    assert result[9] == 0
    assert result[10] == 1  # ocean_proximity_NEAR BAY
    assert result[11] == 0


def test_ordre_8_features_numeriques():
    result = compute_features_for_prediction(
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, '<1H OCEAN'
    )
    assert result[0] == 1.0   # longitude
    assert result[1] == 2.0   # latitude
    assert result[2] == 3.0   # housing_median_age
    assert result[3] == 4.0   # population
    assert result[4] == 5.0   # median_income
    assert result[5] == 6.0   # avg_rooms
    assert result[6] == 7.0   # avg_bedrooms
    assert result[7] == 8.0   # avg_occupancy
