"""Notebook-free training from a small TLC-style raw month."""

from copy import deepcopy
import json
import pickle

import numpy as np
import pandas as pd

from src import data_pipeline, train
from src.pretrip import FEATURES, predict


def test_train_writes_versioned_bundle_and_metadata(tmp_path, monkeypatch):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    lookup = tmp_path / "data" / "taxi_zone_lookup.csv"
    pd.DataFrame({
        "LocationID": [1, 2, 264],
        "Borough": ["Manhattan", "EWR", "Unknown"],
        "service_zone": ["Yellow Zone", "EWR", "N/A"],
    }).to_csv(lookup, index=False)
    trips = []
    for day in (1, 10, 20, 26, 31):
        for hour in (8, 9, 10, 11):
            pickup = pd.Timestamp(f"2023-01-{day:02d} {hour:02d}:00")
            trips.append({
                "tpep_pickup_datetime": pickup,
                "tpep_dropoff_datetime": pickup + pd.Timedelta(minutes=10 + hour),
                "PULocationID": 1,
                "DOLocationID": 2,
            })
    pd.DataFrame(trips).to_parquet(raw_dir / "rides_2023-01.parquet", index=False)
    config = deepcopy(train.CFG)
    config["training"]["models"]["xgboost"].update(n_estimators=8, max_depth=2, n_jobs=1)
    config["pretrip"]["early_stopping_rounds"] = 2
    config["validation"].update(max_train_rows=100, max_eval_rows=100)
    monkeypatch.setattr(train, "CFG", config)
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "ZONE_LOOKUP_PATH", lookup)
    monkeypatch.setattr(data_pipeline, "ROOT", tmp_path)
    existing_model = tmp_path / "models" / "pretrip_model.pkl"
    existing_model.parent.mkdir()
    existing_model.write_bytes(b"unchanged")

    metadata = train.run()

    run_dir = tmp_path / "models" / "training_runs" / metadata["run_id"]
    assert metadata["policy_version"] == "pretrip-v2"
    assert metadata["row_counts"] == {"train": 8, "val": 4, "test": 8}
    assert metadata["features"] == FEATURES
    assert all(np.isfinite(metadata["metrics"][split]["mae"]) for split in ("val", "test"))
    assert json.loads((run_dir / "metadata.json").read_text()) == metadata
    bundle = pickle.loads((run_dir / "model.pkl").read_bytes())
    assert bundle["features"] == FEATURES
    assert 2 in bundle["geography"]["airport_zones"]
    assert 264 not in bundle["geography"]["borough_by_zone"]
    assert np.isfinite(predict(pd.DataFrame([{
        "PULocationID": 1, "DOLocationID": 2, "pickup_datetime": "2023-02-01 08:00"
    }]), run_dir / "model.pkl")).all()
    assert existing_model.read_bytes() == b"unchanged"
