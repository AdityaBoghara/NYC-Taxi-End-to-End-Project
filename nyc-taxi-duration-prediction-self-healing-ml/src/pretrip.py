"""Pre-departure features, chronological training, and local prediction.

Run with ``python -m src.pretrip``. Existing retrospective model artifacts
are preserved. No network access or routing service is required.
"""

from __future__ import annotations

import json
import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.config import CFG, ROOT, TARGET, ZONE_LOOKUP_PATH, processed_path

INPUTS = ["PULocationID", "DOLocationID", "pickup_datetime"]
FEATURES = [
    "PULocationID", "DOLocationID", "hour", "dayofweek", "is_weekend",
    "is_rush_hour", "PU_borough_id", "DO_borough_id", "is_airport_trip",
]


def load_geography(path: Path = ZONE_LOOKUP_PATH) -> dict:
    """Build a deterministic map; persist it in the model bundle for inference."""
    zones = pd.read_csv(path)
    if zones["LocationID"].duplicated().any():
        raise ValueError("Zone lookup must contain unique LocationID values.")
    borough_names = sorted(zones["Borough"].fillna("Unknown").unique())
    borough_codes = {name: i for i, name in enumerate(borough_names)}
    return {
        "borough_names": borough_names,
        "borough_by_zone": {
            int(row.LocationID): borough_codes.get(row.Borough, borough_codes.get("Unknown", -1))
            for row in zones.itertuples()
        },
        # Matches the historical experiment's service-zone rule.
        "airport_zones": zones.loc[zones["service_zone"].eq("Airports"), "LocationID"].astype(int).tolist(),
    }


def build_features(rows: pd.DataFrame, geography: dict, rush_hours: list[int]) -> pd.DataFrame:
    """Accept local NYC wall-clock departure timestamps and known TLC zone IDs.

    Extra columns (including actual distance, rate code, and target) are ignored.
    The public contract is explicitly timezone-naive America/New_York time,
    matching TLC timestamps. Offset-aware timestamps must be converted first.
    """
    missing = set(INPUTS) - set(rows.columns)
    if missing:
        raise ValueError(f"Missing pre-trip inputs: {sorted(missing)}")
    departure = pd.to_datetime(rows["pickup_datetime"], errors="raise")
    if departure.isna().any():
        raise ValueError("Departure time must not be missing.")
    if departure.dt.tz is not None:
        raise ValueError("Use timezone-naive NYC local departure time.")
    features = pd.DataFrame(index=rows.index)
    boroughs = geography["borough_by_zone"]
    for column in INPUTS[:2]:
        values = pd.to_numeric(rows[column], errors="raise")
        if values.isna().any() or not values.isin(boroughs).all():
            raise ValueError(f"{column} must contain known integer TLC zone IDs.")
        features[column] = values.astype("int32")
    features["hour"] = departure.dt.hour
    features["dayofweek"] = departure.dt.dayofweek
    features["is_weekend"] = (features["dayofweek"] >= 5).astype(int)
    features["is_rush_hour"] = features["hour"].isin(rush_hours).astype(int)
    features["PU_borough_id"] = features["PULocationID"].map(boroughs)
    features["DO_borough_id"] = features["DOLocationID"].map(boroughs)
    airports = geography["airport_zones"]
    features["is_airport_trip"] = (
        features["PULocationID"].isin(airports) | features["DOLocationID"].isin(airports)
    ).astype(int)
    return features[FEATURES].astype("int32")


def split_by_time(rows: pd.DataFrame, days: int) -> tuple[dict, dict]:
    """Match the notebook's exact timestamp-based split, including boundary times."""
    if days < 1:
        raise ValueError("Split window must be at least one day.")
    timestamps = pd.to_datetime(rows["pickup_datetime"], errors="raise")
    if timestamps.isna().any():
        raise ValueError("Cannot split missing pickup timestamps.")
    test_cutoff = timestamps.max() - pd.Timedelta(days=days)
    val_cutoff = test_cutoff - pd.Timedelta(days=days)
    masks = {
        "train": timestamps <= val_cutoff,
        "val": (timestamps > val_cutoff) & (timestamps <= test_cutoff),
        "test": timestamps > test_cutoff,
    }
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("Data must span nonempty train, validation, and test windows.")
    return masks, {"val_cutoff": val_cutoff.isoformat(), "test_cutoff": test_cutoff.isoformat()}


def metrics(actual, predicted) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def train() -> dict:
    """Train a pre-trip candidate on the existing processed cohort and save results."""
    data_cfg = CFG["data"]
    path = processed_path(data_cfg["year"], data_cfg["month"])
    rows = pd.read_parquet(path)
    rows = rows.sort_values("pickup_datetime").reset_index(drop=True)
    if not np.isfinite(rows[TARGET]).all():
        raise ValueError("Targets must be finite.")
    geography = load_geography()
    rush_hours = CFG["features"]["rush_hours"]
    X = build_features(rows, geography, rush_hours)
    y = rows[TARGET]
    masks, cutoffs = split_by_time(rows, CFG["training"]["val_days"])
    print(f"Split sizes: { {key: int(mask.sum()) for key, mask in masks.items()} }", flush=True)
    params = CFG["training"]["models"]["xgboost"].copy()
    model = XGBRegressor(
        **params, early_stopping_rounds=CFG["pretrip"]["early_stopping_rounds"]
    )
    model.fit(
        X.loc[masks["train"]], y.loc[masks["train"]],
        eval_set=[(X.loc[masks["val"]], y.loc[masks["val"]])], verbose=False,
    )
    dummy = DummyRegressor(strategy="mean").fit(X.loc[masks["train"]], y.loc[masks["train"]])
    results = {}
    for name, estimator in [("pretrip_xgboost", model), ("dummy_mean", dummy)]:
        results[name] = {
            split: metrics(y.loc[masks[split]], estimator.predict(X.loc[masks[split]]))
            for split in ("val", "test")
        }
    baseline_path = ROOT / CFG["api"]["metadata_path"]
    historical = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    historical_comparison = None
    if historical:
        historical_comparison = {
            "model_name": historical["model_name"],
            "val": historical["val_metrics"], "test": historical["test_metrics"],
            "source": str(baseline_path.relative_to(ROOT)),
            "note": "Previously recorded retrospective scores, not rerun. Uses actual distance and final rate code; not a deployable pre-trip comparator or controlled feature ablation.",
        }
        legacy_path = ROOT / CFG["api"]["model_path"]
        if legacy_path.exists():
            with legacy_path.open("rb") as stream:
                legacy_model = pickle.load(stream)
            legacy_scores = {}
            for split in ("val", "test"):
                legacy_X = rows.loc[masks[split], historical["features"]]
                predictions = legacy_model.predict(legacy_X)
                if historical.get("log_transform", False):
                    predictions = np.expm1(predictions)
                legacy_scores[split] = metrics(y.loc[masks[split]], predictions)
            results["retrospective_saved_model"] = legacy_scores
    report = {
        "model_name": "pretrip_xgboost",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trained_on": f"{data_cfg['year']}-{data_cfg['month']:02d}",
        "processed_data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "input_columns": INPUTS, "features": FEATURES,
        "target": TARGET, "log_transform": False,
        "timezone_contract": "Timezone-naive America/New_York local departure time",
        "split": cutoffs,
        "row_counts": {key: int(mask.sum()) for key, mask in masks.items()},
        "hyperparams": params, "best_iteration": int(model.best_iteration),
        "metrics": results, "historical_reference": historical_comparison,
        "beats_dummy_on_validation": results["pretrip_xgboost"]["val"]["mae"] < results["dummy_mean"]["val"]["mae"],
        "limitations": [
            "Uses the current processed January file and original split rule; historical artifact training-row provenance is unavailable. Selection still depends on historical distance, rate/passenger completeness, and duration filters.",
            "The saved retrospective model is re-evaluated on the same rows when available, but uses post-trip features and separate tuning; this is not a controlled feature ablation.",
            "Predictions are zone-level estimates, not exact-address routing or live traffic ETAs.",
            "No later-month validation, retuning, or new SHAP validation yet.",
            "Airport flag follows service_zone == Airports, matching the prior experiment; EWR service zone is not included.",
        ],
    }
    bundle = {"model": model, "features": FEATURES, "geography": geography, "rush_hours": rush_hours}
    for key, value in [("model_path", bundle), ("metadata_path", report), ("report_path", report)]:
        destination = ROOT / CFG["pretrip"][key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if key == "model_path":
            with destination.open("wb") as stream:
                pickle.dump(value, stream)
        else:
            destination.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    return report


def predict(rows: pd.DataFrame, model_path: Path | None = None) -> np.ndarray:
    """Predict minutes from raw pre-trip inputs using a trusted local bundle."""
    path = model_path if model_path is not None else ROOT / CFG["pretrip"]["model_path"]
    with Path(path).open("rb") as stream:
        bundle = pickle.load(stream)
    X = build_features(rows, bundle["geography"], bundle["rush_hours"])
    if list(X.columns) != bundle["features"]:
        raise ValueError("Saved model schema differs from the feature builder.")
    return bundle["model"].predict(X)


if __name__ == "__main__":
    train()
