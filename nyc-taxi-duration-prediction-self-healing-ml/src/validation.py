"""Reproducible raw-data training, seasonal evaluation, and delayed-outcome replay.

Run ``python -m src.validation`` after ``python -m src.data_pipeline`` downloads.
Outputs are immutable run directories. This command never replaces an active model.
"""
from __future__ import annotations

import json
import os
import pickle
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor

from src.baselines import RouteMedian
from src.config import CFG, ROOT, TARGET, ZONE_LOOKUP_PATH
from src.data_pipeline import POLICY_VERSION, load_month, sha256
from src.pretrip import FEATURES, build_features, load_geography, metrics
from src.promotion import promotion_gate
from src.replay import join_available_outcomes, simulated_release


def sample(rows, limit, seed):
    if limit < 1:
        raise ValueError("Sample limit must be positive.")
    return rows.sample(n=min(limit, len(rows)), random_state=seed).sort_values("pickup_datetime").reset_index(drop=True)


def segment_labels(X, geography):
    names = geography["borough_names"]
    return pd.DataFrame({
        "pickup_borough": X.PU_borough_id.map(dict(enumerate(names))),
        "airport": X.is_airport_trip.map({0: "nonairport", 1: "airport"}),
        "rush_hour": X.is_rush_hour.map({0: "offpeak", 1: "rush"}),
    }, index=X.index)


def summarize(y, predictions, segments):
    result = {**metrics(y, predictions), "rows": len(y),
              "p90_absolute_error": float(np.quantile(np.abs(np.asarray(y) - predictions), .9))}
    result["segments"] = {}
    for column in segments:
        for value in sorted(segments[column].unique()):
            mask = segments[column].eq(value).to_numpy()
            result["segments"][f"{column}={value}"] = {
                "rows": int(mask.sum()), "mae": float(np.abs(np.asarray(y)[mask] - predictions[mask]).mean())}
    return result


def run() -> dict:
    cfg = CFG["validation"]
    seed = CFG["training"]["random_state"]
    train_month = cfg["train_month"]
    evaluation_months = [*cfg["diagnostic_months"], cfg["promotion_month"], *cfg["holdout_months"]]
    if len(set(evaluation_months)) != len(evaluation_months):
        raise ValueError("Diagnostic, promotion, and holdout months must be distinct.")
    if any(pd.Period(month, "M") <= pd.Period(train_month, "M") for month in evaluation_months):
        raise ValueError("All evaluations must be later than training.")
    if any(pd.Period(month, "M").start_time < simulated_release(cfg["promotion_month"])
           for month in cfg["holdout_months"]):
        raise ValueError("Holdout prediction months must follow promotion-label availability.")
    geography = load_geography()
    lookup = pd.read_csv(ZONE_LOOKUP_PATH)
    # New schema version: exclude unknown-location placeholders and include EWR.
    valid = lookup.Borough.isin(["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island", "EWR"])
    known_zones = set(lookup.loc[valid, "LocationID"].astype(int))
    geography["borough_by_zone"] = {key: val for key, val in geography["borough_by_zone"].items() if key in known_zones}
    geography["airport_zones"] = sorted(lookup.loc[lookup.service_zone.isin(["Airports", "EWR"]), "LocationID"].astype(int).tolist())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = ROOT / "models" / "validation_runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "run_id": run_id, "policy_version": POLICY_VERSION,
        "status": "running", "config": CFG,
        "python": platform.python_version(),
        "package_versions": {name: version(name) for name in ["pandas", "numpy", "scikit-learn", "xgboost", "pyarrow"]},
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / "src").glob("*.py"))},
        "lookup_sha256": sha256(ZONE_LOOKUP_PATH),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "data": {}, "evaluations": {},
        "release_policy": "Simulated availability: month start + 3 months. Not verified historical publication dates.",
        "limitations": ["Historical 2023 evaluation, not evidence of present-day accuracy.",
                        "Uniform seeded sampling; row counts and sample limits recorded.",
                        "Only completed trips lasting 1–120 minutes with known zones are evaluated.",
                        "A release-gated replay is a simulation, not live outcome collection.",
                        "Promotion recommendation only; no active service or automatic replacement."],
    }
    rows, report["data"][train_month] = load_month(train_month, known_zones)
    # Last seven days for early stopping only; later-month gate is untouched.
    cutoff = (pd.Period(train_month, "M") + 1).start_time - pd.Timedelta(days=CFG["training"]["val_days"])
    train_rows = sample(rows.loc[rows.pickup_datetime < cutoff], cfg["max_train_rows"], seed)
    val_rows = sample(rows.loc[rows.pickup_datetime >= cutoff], cfg["max_eval_rows"], seed)
    del rows
    rush_hours = CFG["features"]["rush_hours"]
    train_X = build_features(train_rows, geography, rush_hours)
    val_X = build_features(val_rows, geography, rush_hours)
    params = CFG["training"]["models"]["xgboost"].copy()
    candidate = XGBRegressor(**params, early_stopping_rounds=CFG["pretrip"]["early_stopping_rounds"])
    candidate.fit(train_X, train_rows[TARGET], eval_set=[(val_X, val_rows[TARGET])], verbose=False)
    baseline = RouteMedian(cfg["route_min_count"]).fit(train_X, train_rows[TARGET])
    dummy = DummyRegressor(strategy="median").fit(train_X, train_rows[TARGET])
    ready_at = simulated_release(train_month)
    report["training"] = {"rows": len(train_rows), "early_stopping_rows": len(val_rows),
                          "early_stopping_cutoff": cutoff.isoformat(),
                          "best_iteration": int(candidate.best_iteration), "ready_at": ready_at.isoformat()}
    report["early_stopping_metrics"] = {
        name: metrics(val_rows[TARGET], estimator.predict(val_X))
        for name, estimator in [("candidate", candidate), ("route_median", baseline), ("global_median", dummy)]}
    for name, estimator in [("candidate", candidate), ("route_median", baseline)]:
        bundle = {"model": estimator, "features": FEATURES, "geography": geography,
                  "rush_hours": rush_hours, "model_version": f"{run_id}:{name}", "policy_version": POLICY_VERSION}
        (output / f"{name}.pkl").write_bytes(pickle.dumps(bundle))
    pd.concat([train_rows[["trip_id"]].assign(split="train"), val_rows[["trip_id"]].assign(split="early_stopping")]).to_parquet(output / "training_ids.parquet", index=False)
    del train_rows, val_rows, train_X, val_X
    print(f"Trained version {run_id}; evaluating later months.", flush=True)
    for month in sorted(evaluation_months):
        rows, report["data"][month] = load_month(month, known_zones)
        rows = sample(rows, cfg["max_eval_rows"], seed)
        X = build_features(rows, geography, rush_hours)
        segments = segment_labels(X, geography)
        forecast = candidate.predict(X)
        route_prediction = baseline.predict(X)
        global_prediction = dummy.predict(X)
        scores = {name: summarize(rows[TARGET], values, segments)
                  for name, values in [("candidate", forecast), ("route_median", route_prediction), ("global_median", global_prediction)]}
        role = "diagnostic" if month in cfg["diagnostic_months"] else "promotion_gate" if month == cfg["promotion_month"] else "holdout"
        entry = {"role": role, "metrics": scores}
        release = simulated_release(month)
        # February is an offline diagnostic: January labels were unavailable then.
        if rows.pickup_datetime.min() < ready_at:
            if role != "diagnostic":
                raise ValueError("Cannot replay predictions before training labels were available.")
            entry["replay"] = "offline_backtest_only_model_not_available_at_trip_time"
        else:
            predictions = pd.DataFrame({
                "trip_id": rows.trip_id, "predicted_at": rows.pickup_datetime,
                "prediction": forecast, "model_version": f"{run_id}:candidate",
                "incumbent_prediction": route_prediction,
                "candidate_ready_at": ready_at,
            })
            for column in segments:
                predictions[column] = segments[column]
            outcomes = pd.DataFrame({"trip_id": rows.trip_id, "actual_duration": rows[TARGET],
                                     "completed_at": rows.dropoff_datetime, "available_at": release})
            before = join_available_outcomes(predictions, outcomes, release - pd.Timedelta(seconds=1))
            after = join_available_outcomes(predictions, outcomes, release)
            assert len(before) == 0 and len(after) == len(rows)
            predictions.to_parquet(output / f"{month}_predictions.parquet", index=False)
            outcomes.to_parquet(output / f"{month}_outcomes.parquet", index=False)
            entry["replay"] = {"outcomes_before_release": len(before), "outcomes_after_release": len(after),
                               "release_at": release.isoformat(), "prediction_mode": "frozen_candidate_shadow"}
            if role == "promotion_gate":
                gate = promotion_gate(
                    after.actual_duration, after.incumbent_prediction, after.prediction,
                    after[list(segments.columns)], min_rows=cfg["min_labeled_rows"],
                    min_segment_rows=cfg["min_segment_rows"],
                    min_improvement_pct=cfg["min_improvement_pct"],
                    max_segment_regression_pct=cfg["max_segment_regression_pct"],
                    required_segments={
                        "pickup_borough": ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island", "EWR"],
                        "airport": ["airport", "nonairport"], "rush_hour": ["rush", "offpeak"],
                    },
                )
                gate.update(evaluated_at=release.isoformat(), gate_month=month,
                            incumbent="route_median", candidate=f"{run_id}:candidate",
                            automatic_replacement=False)
                report["promotion"] = gate
        report["evaluations"][month] = entry
        print(month, role, {name: round(values["mae"], 4) for name, values in scores.items()}, flush=True)
    report["status"] = "complete"
    report["candidate_sha256"] = sha256(output / "candidate.pkl")
    report["baseline_sha256"] = sha256(output / "route_median.pkl")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    # An inspection pointer, never an active-model pointer. Publish only on success.
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    temporary = reports / f"validation_latest.{run_id}.tmp"
    temporary.write_text(json.dumps({"report": str((output / "report.json").relative_to(ROOT))}, indent=2) + "\n")
    os.replace(temporary, reports / "validation_latest.json")
    print(f"Report: {output / 'report.json'}", flush=True)
    print(f"Promotion recommendation: {report['promotion']['approved']}; {report['promotion']['reasons']}", flush=True)
    return report


if __name__ == "__main__":
    run()
