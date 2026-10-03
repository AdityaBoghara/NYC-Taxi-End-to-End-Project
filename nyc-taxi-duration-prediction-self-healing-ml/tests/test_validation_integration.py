"""Exercise the complete seasonal run with small, isolated TLC-style files."""

from copy import deepcopy
import json
import pickle
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from src import data_pipeline, validation
from src.pretrip import FEATURES, load_v2_geography, predict


def test_validation_run_trains_replays_and_writes_inspection_artifacts(tmp_path, monkeypatch):
    shutil.copytree(Path(validation.__file__).parent, tmp_path / "src",
                    ignore=shutil.ignore_patterns("__pycache__"))
    zones = pd.DataFrame({
        "LocationID": [1, 2, 3, 4, 5, 6],
        "Borough": ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island", "EWR"],
        "service_zone": ["Boro Zone"] * 5 + ["EWR"],
    })
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    lookup_path = data_dir / "taxi_zone_lookup.csv"
    zones.to_csv(lookup_path, index=False)

    for month in ("2023-01", "2023-02", "2023-04", "2023-07", "2023-10"):
        trips = []
        for day in ((10, 28) if month == "2023-01" else (10,)):
            for zone in zones.LocationID:
                for hour in (8, 12):
                    pickup = pd.Timestamp(f"{month}-{day:02d} {hour:02d}:00")
                    trips.append({
                        "tpep_pickup_datetime": pickup,
                        "tpep_dropoff_datetime": pickup + pd.Timedelta(minutes=8 + zone + hour // 4),
                        "PULocationID": zone,
                        "DOLocationID": 1 + zone % 6,
                    })
        pd.DataFrame(trips).to_parquet(raw_dir / f"rides_{month}.parquet", index=False)

    config = deepcopy(validation.CFG)
    config["training"]["models"]["xgboost"].update(n_estimators=8, max_depth=2, n_jobs=1)
    config["pretrip"]["early_stopping_rounds"] = 2
    config["validation"].update(max_train_rows=100, max_eval_rows=100, route_min_count=1,
                                min_labeled_rows=1, min_segment_rows=1)
    monkeypatch.setattr(validation, "CFG", config)
    monkeypatch.setattr(validation, "ROOT", tmp_path)
    monkeypatch.setattr(data_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(validation, "ZONE_LOOKUP_PATH", lookup_path)
    monkeypatch.setattr(validation, "load_v2_geography", lambda path: load_v2_geography(lookup_path))

    report = validation.run()
    run_dir = tmp_path / "models" / "validation_runs" / report["run_id"]
    assert report["status"] == "complete"
    assert "src/validation.py" in report["source_sha256"]
    assert report["training"]["rows"] == report["training"]["early_stopping_rows"] == 12
    assert report["evaluations"]["2023-02"]["replay"] == "offline_backtest_only_model_not_available_at_trip_time"
    assert [report["evaluations"][month]["role"] for month in ("2023-04", "2023-07", "2023-10")] == [
        "promotion_gate", "holdout", "holdout"]
    for month in ("2023-04", "2023-07", "2023-10"):
        assert report["evaluations"][month]["replay"]["outcomes_before_release"] == 0
        assert report["evaluations"][month]["replay"]["outcomes_after_release"] == 12
        assert (run_dir / f"{month}_predictions.parquet").exists()
        assert (run_dir / f"{month}_outcomes.parquet").exists()
    assert report["promotion"]["automatic_replacement"] is False
    assert len(pd.read_parquet(run_dir / "training_ids.parquet")) == 24
    assert json.loads((run_dir / "report.json").read_text())["status"] == "complete"
    pointer = json.loads((tmp_path / "reports" / "validation_latest.json").read_text())
    assert pointer["report"] == str((run_dir / "report.json").relative_to(tmp_path))
    assert not (tmp_path / "models" / "best_model.pkl").exists()

    bundle = pickle.loads((run_dir / "candidate.pkl").read_bytes())
    assert bundle["features"] == FEATURES
    prediction = predict(pd.DataFrame([{
        "PULocationID": 1, "DOLocationID": 2, "pickup_datetime": "2023-10-10 08:00"
    }]), model_path=run_dir / "candidate.pkl")
    assert prediction.shape == (1,) and np.isfinite(prediction).all()

    monkeypatch.setattr(validation, "join_available_outcomes", lambda *args: pd.DataFrame())
    with pytest.raises(ValueError, match="Replay availability"):
        validation.run()
    assert json.loads((tmp_path / "reports" / "validation_latest.json").read_text()) == pointer
