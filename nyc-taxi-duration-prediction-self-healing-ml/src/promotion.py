"""Pure evaluation gate. Drift can request training, never authorize promotion."""
import numpy as np
import pandas as pd


def promotion_gate(actual, incumbent, candidate, segments, *, min_rows=1000,
                   min_segment_rows=100, min_improvement_pct=2.0,
                   max_segment_regression_pct=5.0, required_segments=None) -> dict:
    y, old, new = (np.asarray(values, dtype=float) for values in (actual, incumbent, candidate))
    reasons = []
    if len(y) < min_rows:
        reasons.append("insufficient_labeled_rows")
    if not (y.shape == old.shape == new.shape) or y.ndim != 1:
        raise ValueError("Paired predictions and outcomes must have matching one-dimensional shapes.")
    if not all(np.isfinite(values).all() for values in (y, old, new)):
        return {"approved": False, "reasons": ["nonfinite_outcomes_or_predictions"]}
    if len(segments.columns) == 0 or len(segments) != len(y) or segments.isna().any().any():
        raise ValueError("Every evaluated row needs complete segment labels.")
    old_error, new_error = np.abs(y - old), np.abs(y - new)
    old_mae = float(old_error.mean()) if len(y) else None
    new_mae = float(new_error.mean()) if len(y) else None
    if len(y) and not new_mae < old_mae * (1 - min_improvement_pct / 100):
        reasons.append("insufficient_overall_improvement")
    checks = {}
    required_segments = required_segments or {}
    if not set(required_segments) <= set(segments.columns):
        raise ValueError("Required segment dimensions are missing.")
    for column in segments:
        values = set(segments[column].unique()) | set(required_segments.get(column, []))
        for segment in sorted(values):
            mask = segments[column].to_numpy() == segment
            key = f"{column}={segment}"
            count = int(mask.sum())
            checks[key] = {"rows": count}
            if count < min_segment_rows:
                reasons.append(f"insufficient_segment_evidence:{key}")
                continue
            before, after = float(old_error[mask].mean()), float(new_error[mask].mean())
            checks[key].update(incumbent_mae=before, candidate_mae=after)
            if after > before * (1 + max_segment_regression_pct / 100):
                reasons.append(f"segment_regression:{key}")
    return {"approved": not reasons, "reasons": reasons,
            "rows": len(y), "incumbent_mae": old_mae, "candidate_mae": new_mae,
            "segments": checks}


def monitor_action(feature_drift: bool, labeled_mae_degraded: bool) -> str:
    return "investigate_or_train_candidate" if feature_drift or labeled_mae_degraded else "no_action"
