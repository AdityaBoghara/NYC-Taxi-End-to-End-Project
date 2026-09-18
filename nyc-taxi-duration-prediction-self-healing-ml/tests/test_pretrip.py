import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.pretrip import FEATURES, build_features, predict, split_by_time


@pytest.fixture
def geography():
    return {"borough_by_zone": {1: 0, 2: 1}, "airport_zones": [2]}


@pytest.fixture
def trip_inputs():
    return pd.DataFrame([{
        "PULocationID": 1, "DOLocationID": 2,
        "pickup_datetime": "2023-01-07 08:30:00",
    }])


def test_features_ignore_post_trip_values_and_derive_time(trip_inputs, geography):
    expected = build_features(trip_inputs, geography, [8])
    changed = trip_inputs.assign(trip_distance=999, RatecodeID=99, trip_duration_min=1000,
                             hour=23, is_airport_trip=0, PU_borough_id=99)
    pd.testing.assert_frame_equal(expected, build_features(changed, geography, [8]))
    assert expected.iloc[0].to_dict() == {
        "PULocationID": 1, "DOLocationID": 2, "hour": 8, "dayofweek": 5,
        "is_weekend": 1, "is_rush_hour": 1, "PU_borough_id": 0,
        "DO_borough_id": 1, "is_airport_trip": 1,
    }
    assert not {"trip_distance", "RatecodeID", "passenger_count"} & set(expected.columns)


@pytest.mark.parametrize("zone", [999, 1.5, None])
def test_rejects_unknown_or_invalid_zones(trip_inputs, geography, zone):
    trip_inputs["PULocationID"] = zone
    with pytest.raises(ValueError, match="known integer"):
        build_features(trip_inputs, geography, [8])


def test_rejects_missing_and_offset_aware_departure(trip_inputs, geography):
    with pytest.raises(ValueError, match="Missing pre-trip"):
        build_features(trip_inputs.drop(columns="pickup_datetime"), geography, [8])
    for timestamp in [None, "2023-01-07T08:00:00-05:00"]:
        trip_inputs["pickup_datetime"] = timestamp
        with pytest.raises(ValueError):
            build_features(trip_inputs, geography, [8])


def test_time_split_boundaries_are_disjoint_and_exhaustive():
    rows = pd.DataFrame({"pickup_datetime": pd.to_datetime([
        "2023-01-01 12:00", "2023-01-17 12:00", "2023-01-17 12:01",
        "2023-01-24 12:00", "2023-01-24 12:01", "2023-01-31 12:00",
    ])})
    masks, dates = split_by_time(rows, 7)
    assert dates["val_cutoff"] == "2023-01-17T12:00:00"
    assert masks["train"].tolist() == [True, True, False, False, False, False]
    assert masks["val"].tolist() == [False, False, True, True, False, False]
    assert masks["test"].tolist() == [False, False, False, False, True, True]
    assert (sum(mask.astype(int) for mask in masks.values()) == 1).all()
    with pytest.raises(ValueError, match="nonempty"):
        split_by_time(rows.iloc[:1], 7)


def test_saved_prediction_uses_bundled_mapping(tmp_path, trip_inputs, geography):
    X = build_features(trip_inputs, geography, [8])
    model = DummyRegressor(strategy="constant", constant=12).fit(X, [12])
    path = tmp_path / "model.pkl"
    bundle = {"model": model, "features": FEATURES, "geography": geography, "rush_hours": [8]}
    path.write_bytes(pickle.dumps(bundle))
    np.testing.assert_array_equal(predict(trip_inputs, path), [12])
    bundle["features"] = list(reversed(FEATURES))
    path.write_bytes(pickle.dumps(bundle))
    with pytest.raises(ValueError, match="schema"):
        predict(trip_inputs, path)
