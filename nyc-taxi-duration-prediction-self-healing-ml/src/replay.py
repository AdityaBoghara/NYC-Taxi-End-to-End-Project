"""Point-in-time outcome joins for simulated monthly TLC publication delays."""
import numpy as np
import pandas as pd


def simulated_release(month: str) -> pd.Timestamp:
    """Assumption: first day three months after pickup month, not actual release history."""
    return (pd.Period(month, freq="M") + 3).start_time


def join_available_outcomes(predictions: pd.DataFrame, outcomes: pd.DataFrame, as_of) -> pd.DataFrame:
    """Evaluate only recorded predictions with finished and released outcomes.

    Offline backtests may have been generated later than trip time; callers must
    label those separately from a prospective replay and enforce model readiness.
    """
    pred_required = {"trip_id", "prediction", "model_version", "predicted_at"}
    outcome_required = {"trip_id", "actual_duration", "completed_at", "available_at"}
    if not pred_required <= set(predictions) or not outcome_required <= set(outcomes):
        raise ValueError("Missing prediction/outcome ledger columns.")
    if predictions.trip_id.duplicated().any() or outcomes.trip_id.duplicated().any():
        raise ValueError("Duplicate trip IDs would corrupt outcome joins.")
    if predictions[list(pred_required)].isna().any().any() or outcomes[list(outcome_required)].isna().any().any():
        raise ValueError("Ledger fields must not be missing.")
    joined = predictions.merge(outcomes, on="trip_id", how="inner", validate="one_to_one")
    as_of = pd.Timestamp(as_of)
    available = pd.to_datetime(joined.available_at)
    completed = pd.to_datetime(joined.completed_at)
    predicted = pd.to_datetime(joined.predicted_at)
    if "candidate_ready_at" in joined:
        ready = pd.to_datetime(joined.candidate_ready_at)
        if ready.isna().any() or (ready > predicted).any():
            raise ValueError("Model cannot predict before its training outcomes are available.")
    if (available < completed).any() or (predicted > completed).any():
        raise ValueError("Invalid prediction/completion/availability ordering.")
    joined = joined.loc[(available <= as_of) & (completed <= as_of) & (predicted <= as_of)].copy()
    if not np.isfinite(joined[["prediction", "actual_duration"]].to_numpy(dtype=float)).all():
        raise ValueError("Evaluation requires finite predictions and outcomes.")
    return joined
