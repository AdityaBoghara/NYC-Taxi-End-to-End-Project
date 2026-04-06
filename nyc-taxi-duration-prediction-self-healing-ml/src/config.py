"""
config.py
---------
Loads configs/config.yaml and exposes typed constants used by every
other module. Import from here — never hard-code paths or thresholds
elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Project root is one level above src/
_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "configs" / "config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


_cfg = _load_yaml(_CONFIG_PATH)


# ── Helpers ────────────────────────────────────────────────────────────────

def _resolve(template: str, year: int, month: int) -> Path:
    """Resolve a path template like 'data/raw/rides_{year}-{month:02d}.parquet'."""
    return _ROOT / template.format(year=year, month=month)


# ── Public constants ────────────────────────────────────────────────────────

ROOT: Path = _ROOT

# Raw config dict — available for anything not covered below
CFG: dict[str, Any] = _cfg


# Data
DATA_YEAR: int = _cfg["data"]["year"]
DATA_MONTH: int = _cfg["data"]["month"]
ZONE_LOOKUP_URL: str = _cfg["data"]["zone_lookup_url"]
ZONE_LOOKUP_PATH: Path = _ROOT / _cfg["data"]["paths"]["zone_lookup"]
MODELS_DIR: Path = _ROOT / _cfg["data"]["paths"]["models"]
LOGS_DIR: Path = _ROOT / _cfg["data"]["paths"]["logs"]
REPORTS_DIR: Path = _ROOT / _cfg["data"]["paths"]["reports"]

# Model artifact paths
MODEL_PATH: Path = _ROOT / _cfg["api"]["model_path"]
METADATA_PATH: Path = _ROOT / _cfg["api"]["metadata_path"]


def raw_path(year: int, month: int) -> Path:
    return _resolve(_cfg["data"]["paths"]["raw"], year, month)


def interim_path(year: int, month: int) -> Path:
    return _resolve(_cfg["data"]["paths"]["interim"], year, month)


def processed_path(year: int, month: int) -> Path:
    return _resolve(_cfg["data"]["paths"]["processed"], year, month)


# Filters
FILTER = _cfg["filters"]
DURATION_MIN: float = FILTER["duration_min_minutes"]
DURATION_MAX: float = FILTER["duration_max_minutes"]
DISTANCE_MIN: float = FILTER["distance_min_miles"]    # exclusive >
DISTANCE_MAX: float = FILTER["distance_max_miles"]
PASSENGER_MIN: int = FILTER["passenger_min"]
PASSENGER_MAX: int = FILTER["passenger_max"]

# Features
_feat = _cfg["features"]
CATEGORICAL_FEATURES: list[str] = _feat["categorical"]
NUMERIC_FEATURES: list[str] = _feat["numeric"]
ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET: str = _feat["target"]
RUSH_HOURS: list[int] = _feat["rush_hours"]

# Training
TRAIN_CFG = _cfg["training"]
VAL_DAYS: int = TRAIN_CFG["val_days"]
RANDOM_STATE: int = TRAIN_CFG["random_state"]
LOG_TRANSFORM: bool = TRAIN_CFG["log_transform_target"]

# MLflow
MLFLOW_EXPERIMENT: str = _cfg["mlflow"]["experiment_name"]
MLFLOW_TRACKING_URI: str = str(_ROOT / _cfg["mlflow"]["tracking_uri"])

# Monitoring
MON_CFG = _cfg["monitoring"]
MAE_DEGRADATION_THRESHOLD: float = MON_CFG["mae_degradation_threshold_pct"]
FEATURE_DRIFT_THRESHOLD: float = MON_CFG["feature_drift_threshold_pct"]
CHECK_INTERVAL_HOURS: int = MON_CFG["check_interval_hours"]

# API
API_HOST: str = _cfg["api"]["host"]
API_PORT: int = _cfg["api"]["port"]


def ensure_dirs() -> None:
    """Create runtime directories if they don't exist."""
    for d in (MODELS_DIR, LOGS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
