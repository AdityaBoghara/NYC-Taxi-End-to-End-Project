"""Train a versioned pre-trip XGBoost candidate from a raw TLC month."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from uuid import uuid4

from xgboost import XGBRegressor

from src.config import CFG, ROOT, TARGET, ZONE_LOOKUP_PATH
from src.data_pipeline import POLICY_VERSION, load_month, sha256
from src.pretrip import INPUTS, FEATURES, build_features, load_v2_geography, metrics, split_by_time


def run(month: str | None = None) -> dict:
    month = month or CFG["validation"]["train_month"]
    geography = load_v2_geography(ZONE_LOOKUP_PATH)
    rows, data_report = load_month(month, set(geography["borough_by_zone"]))
    masks, cutoffs = split_by_time(rows, CFG["training"]["val_days"])
    cfg = CFG["validation"]
    limits = {"train": cfg["max_train_rows"], "val": cfg["max_eval_rows"], "test": cfg["max_eval_rows"]}
    if any(limit < 1 for limit in limits.values()):
        raise ValueError("Training and evaluation sample limits must be positive.")
    seed = CFG["training"]["random_state"]
    splits = {
        name: rows.loc[mask].sample(n=min(limits[name], int(mask.sum())), random_state=seed)
        .sort_values("pickup_datetime").reset_index(drop=True)
        for name, mask in masks.items()
    }
    rush_hours = CFG["features"]["rush_hours"]
    features = {name: build_features(part, geography, rush_hours) for name, part in splits.items()}
    params = CFG["training"]["models"]["xgboost"].copy()
    model = XGBRegressor(**params, early_stopping_rounds=CFG["pretrip"]["early_stopping_rounds"])
    model.fit(features["train"], splits["train"][TARGET],
              eval_set=[(features["val"], splits["val"][TARGET])], verbose=False)
    scores = {name: metrics(splits[name][TARGET], model.predict(features[name]))
              for name in ("val", "test")}

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = ROOT / "models" / "training_runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    bundle = {"model": model, "features": FEATURES, "geography": geography,
              "rush_hours": rush_hours, "model_version": run_id, "policy_version": POLICY_VERSION}
    (output / "model.pkl").write_bytes(pickle.dumps(bundle))
    metadata = {
        "run_id": run_id, "model_name": "pretrip_xgboost", "created_at": datetime.now(timezone.utc).isoformat(),
        "train_month": month, "policy_version": POLICY_VERSION, "data": data_report,
        "lookup_sha256": sha256(ZONE_LOOKUP_PATH), "input_columns": INPUTS, "features": FEATURES,
        "target": TARGET, "timezone_contract": "Timezone-naive America/New_York local departure time",
        "split": cutoffs, "row_counts": {name: len(part) for name, part in splits.items()},
        "sampling": {"seed": seed, "limits": limits}, "hyperparams": params,
        "best_iteration": int(model.best_iteration), "metrics": scores, "model_file": "model.pkl",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", help="TLC month in YYYY-MM format; defaults to validation.train_month")
    args = parser.parse_args()
    print(json.dumps(run(args.month), indent=2))
