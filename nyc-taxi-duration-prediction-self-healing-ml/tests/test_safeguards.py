import numpy as np
import pandas as pd
import pytest

from src.baselines import RouteMedian
from src.data_pipeline import clean
from src.promotion import monitor_action, promotion_gate
from src.replay import join_available_outcomes, simulated_release


def test_new_cohort_ignores_irrelevant_metadata_and_includes_month_start():
    raw = pd.DataFrame({
        "tpep_pickup_datetime": ["2023-01-01 00:00", "2023-01-02", "2023-02-01", "bad"],
        "tpep_dropoff_datetime": ["2023-01-01 00:10", "2023-01-02 00:20", "2023-02-01 00:10", "bad"],
        "PULocationID": [1, 1, 1, 1], "DOLocationID": [2, 2, 2, 2],
        "trip_distance": [0, 999, 0, 0], "passenger_count": [None] * 4,
        "RatecodeID": [None] * 4, "payment_type": [0] * 4,
    })
    rows, report = clean(raw, "2023-01", {1, 2})
    assert rows.trip_duration_min.tolist() == [10, 20]
    assert report["removed"]["outside_pickup_month"] == 1
    assert report["raw_rows"] == report["retained_rows"] + sum(report["removed"].values())
    altered = raw.assign(trip_distance=3, passenger_count=1, RatecodeID=1, payment_type=1)
    pd.testing.assert_frame_equal(rows, clean(altered, "2023-01", {1, 2})[0])


def test_duration_and_unknown_location_scope():
    raw = pd.DataFrame({"pickup_datetime": pd.to_datetime(["2023-01-10"] * 5),
                        "PULocationID": [1, 1, 1, 1, 264], "DOLocationID": [2] * 5})
    raw["dropoff_datetime"] = raw.pickup_datetime + pd.to_timedelta([1, 120, -1, 121, 5], unit="m")
    rows, report = clean(raw, "2023-01", {1, 2})
    assert rows.trip_duration_min.tolist() == [1, 120]
    assert report["removed"]["unknown_locations"] == 1


def test_route_baseline_hierarchical_fallback_and_training_only():
    X = pd.DataFrame({"PULocationID": [1, 1, 1, 1, 2, 2],
                      "DOLocationID": [2, 2, 2, 3, 3, 3],
                      "dayofweek": [0] * 6, "hour": [8, 8, 12, 8, 8, 8]})
    model = RouteMedian(min_count=2).fit(X, [10, 20, 60, 80, 100, 120])
    query = pd.DataFrame({"PULocationID": [1, 1, 1, 9], "DOLocationID": [2, 2, 9, 9],
                          "dayofweek": [0, 1, 0, 0], "hour": [8, 8, 8, 8]})
    np.testing.assert_allclose(model.predict(query), [15, 20, 40, 70])
    # Mutating evaluation labels cannot refit a training-only statistic.
    query["target"] = 9999
    np.testing.assert_allclose(model.predict(query), [15, 20, 40, 70])


def gate(y, old, new, segments):
    return promotion_gate(y, old, new, segments, min_rows=4, min_segment_rows=2)


def test_gate_rejects_segment_regression_despite_better_overall_mae():
    y = np.zeros(10)
    old = np.full(10, 10)
    new = np.array([1] * 8 + [12] * 2)
    result = gate(y, old, new, pd.DataFrame({"airport": [0] * 8 + [1] * 2}))
    assert result["candidate_mae"] < result["incumbent_mae"]
    assert not result["approved"]
    assert "segment_regression:airport=1" in result["reasons"]


def test_gate_passes_supported_improvement_and_blocks_sparse_evidence():
    segments = pd.DataFrame({"airport": [0, 0, 1, 1]})
    assert gate([0] * 4, [10] * 4, [8] * 4, segments)["approved"]
    sparse = pd.DataFrame({"airport": [0, 0, 0, 1]})
    assert not gate([0] * 4, [10] * 4, [8] * 4, sparse)["approved"]
    assert not gate([0] * 4, [10] * 4, [11] * 4, segments)["approved"]
    assert not gate([0] * 4, [10] * 4, [np.nan] * 4, segments)["approved"]


@pytest.mark.parametrize("drift,degraded", [(True, False), (False, True), (True, True)])
def test_drift_never_authorizes_replacement(drift, degraded):
    assert monitor_action(drift, degraded) == "investigate_or_train_candidate"


def ledgers():
    predictions = pd.DataFrame({"trip_id": ["a", "b"], "prediction": [10, 20],
                                "model_version": ["v1"] * 2, "predicted_at": ["2023-04-10"] * 2})
    outcomes = pd.DataFrame({"trip_id": ["a", "b"], "actual_duration": [12, 18],
                            "completed_at": ["2023-04-10 00:20"] * 2,
                            "available_at": [simulated_release("2023-04")] * 2})
    return predictions, outcomes


def test_outcomes_are_unavailable_until_release_and_join_by_id():
    predictions, outcomes = ledgers()
    assert join_available_outcomes(predictions, outcomes, "2023-06-30").empty
    joined = join_available_outcomes(predictions, outcomes.iloc[::-1], "2023-07-01")
    assert joined.actual_duration.tolist() == [12, 18]
    assert simulated_release("2023-10") == pd.Timestamp("2024-01-01")


def test_replay_rejects_duplicate_ids_and_impossible_timing():
    predictions, outcomes = ledgers()
    with pytest.raises(ValueError, match="Duplicate"):
        join_available_outcomes(predictions, pd.concat([outcomes, outcomes]), "2023-07-01")
    outcomes["available_at"] = "2023-01-01"
    with pytest.raises(ValueError, match="ordering"):
        join_available_outcomes(predictions, outcomes, "2023-07-01")


def test_replay_refuses_model_trained_with_unavailable_outcomes():
    predictions, outcomes = ledgers()
    predictions["candidate_ready_at"] = pd.Timestamp("2023-05-01")
    with pytest.raises(ValueError, match="training outcomes"):
        join_available_outcomes(predictions, outcomes, "2023-07-01")


def test_gate_requires_segment_labels():
    with pytest.raises(ValueError, match="segment labels"):
        gate([0] * 4, [10] * 4, [8] * 4, pd.DataFrame(index=range(4)))


def test_absent_required_segment_blocks_promotion():
    result = promotion_gate([0] * 4, [10] * 4, [8] * 4,
                            pd.DataFrame({"airport": [0] * 4}), min_rows=4,
                            min_segment_rows=2, required_segments={"airport": [0, 1]})
    assert not result["approved"]
    assert "insufficient_segment_evidence:airport=1" in result["reasons"]
