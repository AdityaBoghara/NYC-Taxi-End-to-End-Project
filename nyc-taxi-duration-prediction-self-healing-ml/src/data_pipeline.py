"""Versioned pre-trip cohort from raw TLC files; no notebook state required."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import ROOT

POLICY_VERSION = "pretrip-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def month_path(month: str) -> Path:
    parsed = pd.Period(month, freq="M")
    if str(parsed) != month:
        raise ValueError("Month must use YYYY-MM format.")
    return ROOT / "data" / "raw" / f"rides_{month}.parquet"


def download(month: str) -> Path:
    path = month_path(month)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.part")
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    # Validate before exposing a completed download.
    import pyarrow.parquet as pq
    pq.ParquetFile(temporary)
    os.replace(temporary, path)
    return path


def clean(raw: pd.DataFrame, month: str, known_zones: set[int]) -> tuple[pd.DataFrame, dict]:
    """Keep valid in-month timestamps, known locations, and 1–120 minute labels.

    Distance, passenger count, payment type, and rate code never decide inclusion.
    Duration bounds define a restricted evaluation cohort, not a runtime filter.
    """
    renamed = raw.rename(columns={"tpep_pickup_datetime": "pickup_datetime",
                                  "tpep_dropoff_datetime": "dropoff_datetime"})
    required = ["pickup_datetime", "dropoff_datetime", "PULocationID", "DOLocationID"]
    if not set(required) <= set(renamed.columns):
        raise ValueError(f"Missing raw columns: {sorted(set(required) - set(renamed.columns))}")
    rows = renamed[required].copy()
    # Stable within the exact raw file; not a real-world trip identifier.
    rows["source_row"] = np.arange(len(rows), dtype=np.int64)
    for key in ["pickup_datetime", "dropoff_datetime"]:
        rows[key] = pd.to_datetime(rows[key], errors="coerce", format="mixed")
        if rows[key].dt.tz is not None:
            raise ValueError("Raw TLC timestamps must be timezone-naive NYC local time.")
    rows["trip_duration_min"] = (rows.dropoff_datetime - rows.pickup_datetime).dt.total_seconds() / 60
    start = pd.Period(month, freq="M").start_time
    end = start + pd.offsets.MonthBegin(1)
    criteria = {
        "invalid_timestamps": rows.pickup_datetime.notna() & rows.dropoff_datetime.notna(),
        "outside_pickup_month": rows.pickup_datetime.ge(start) & rows.pickup_datetime.lt(end),
        "unknown_locations": rows.PULocationID.isin(known_zones) & rows.DOLocationID.isin(known_zones),
        "outside_duration_scope": rows.trip_duration_min.between(1, 120),
    }
    retained = pd.Series(True, index=rows.index)
    removed = {}
    for reason, valid in criteria.items():
        removed[reason] = int((retained & ~valid).sum())
        retained &= valid
    rows = rows.loc[retained].sort_values(["pickup_datetime", "source_row"]).reset_index(drop=True)
    for key in ["PULocationID", "DOLocationID"]:
        rows[key] = rows[key].astype(int)
    return rows, {
        "policy_version": POLICY_VERSION, "month": month,
        "raw_rows": len(raw), "retained_rows": len(rows), "removed": removed,
        "duration_scope_minutes": [1, 120],
        "unused_for_filtering": ["trip_distance", "passenger_count", "RatecodeID", "payment_type"],
        "duplicates": "Retained: available fields cannot reliably identify duplicate trips.",
    }


def load_month(month: str, known_zones: set[int]) -> tuple[pd.DataFrame, dict]:
    path = month_path(month)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run python -m src.data_pipeline {month}")
    digest = sha256(path)
    rows, report = clean(pd.read_parquet(path), month, known_zones)
    rows["trip_id"] = month + ":" + digest[:16] + ":" + rows.source_row.astype(str)
    report["raw_sha256"] = digest
    return rows, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("months", nargs="+")
    for month in parser.parse_args().months:
        print(download(month), flush=True)
