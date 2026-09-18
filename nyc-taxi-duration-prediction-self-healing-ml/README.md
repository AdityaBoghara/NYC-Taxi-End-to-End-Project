# NYC Taxi Trip Duration Prediction — Self-Healing ML System

A machine learning project that predicts NYC yellow taxi trip duration in minutes. Raw-data preparation, pre-trip training, seasonal evaluation, promotion safeguards, and delayed-outcome replay are implemented. FastAPI serving, live drift detection, and automated retraining orchestration are planned.

The pre-trip model requires only pickup zone, destination zone, and NYC local departure time. The original 12-feature notebook model remains a retrospective benchmark because it uses actual trip distance and final rate code.

## Project Overview

| Component | Description |
|---|---|
| **Data** | NYC TLC Yellow Taxi trip records (Jan 2023, ~3M trips) |
| **Target** | `trip_duration_min` — elapsed time from pickup to dropoff |
| **Models** | XGBoost / LightGBM (best chosen by validation MAE) |
| **Tracking** | MLflow experiment tracking for every run |
| **Serving** | Local Python prediction implemented; FastAPI planned |
| **Monitoring** | Planned: Evidently drift detection + retraining trigger |

---

## Project Structure

This combines implemented files with the intended module layout. Reusable modules now include `config.py`, `pretrip.py`, `data_pipeline.py`, `baselines.py`, `validation.py`, `promotion.py`, `replay.py`, and `audit_legacy.py`. FastAPI and a live monitoring service are still planned.

```
nyc-taxi-duration-prediction-self-healing-ml/
├── data/
│   ├── raw/            # Unmodified TLC download
│   ├── interim/        # Date-filtered, negatives removed
│   └── processed/      # Model-ready (12 features + target)
├── notebooks/
│   ├── 1. load_validate_raw_data.ipynb
│   └── 2. eda_feature_engineering.ipynb
├── output/jupyter-notebook/
│   └── pretrip-eta-baseline.ipynb # Executed pre-trip experiment
├── src/
│   ├── pretrip.py              # Implemented training + local prediction
│   ├── config.py               # Load configs/config.yaml
│   ├── data_loader.py          # Load/download parquet data
│   ├── feature_engineering.py  # Filters + feature derivation
│   ├── train.py                # Training pipeline + MLflow
│   ├── evaluate.py             # Metrics + SHAP analysis
│   ├── predict.py              # Load model + predict
│   ├── monitor.py              # Drift detection + auto-retrain
│   └── api.py                  # FastAPI app
├── tests/
├── configs/
│   └── config.yaml             # All constants and thresholds
├── models/                     # Saved model artifacts (git-ignored)
├── logs/                       # Drift logs (git-ignored)
├── docs/
│   └── design_decisions.md
├── environment.yml
└── requirements.txt
```

---

## Setup

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate nyc-taxi-ml

# 2. Verify
python -c "import pandas, sklearn, xgboost, mlflow, fastapi; print('OK')"
```

---

## Validated pipeline (recommended)

This workflow starts from raw data and does not require executing notebooks:

```bash
# Download the public monthly files once (existing files are reused).
python -m src.data_pipeline 2023-01 2023-02 2023-04 2023-07 2023-10

# Train, compare baselines, evaluate across seasons, and replay delayed outcomes.
python -m src.validation

# Investigate the original artifact/score mismatch independently.
python -m src.audit_legacy

python -m pytest tests/ -q
```

See [validation results and safeguards](docs/validation_results.md) for scope, metrics, the blocked promotion decision, and remaining work.

The `validation` section in `configs/config.yaml` declares the evaluation months, sampling limits, and promotion thresholds. Each run writes an immutable directory under `models/validation_runs/`, including model bundles, data/code hashes, exact training row IDs, prediction/outcome ledgers, and a report. `reports/validation_latest.json` points to the last completed report; it is **not** an active model pointer.

January supplies training and early-stopping data. February is an offline diagnostic; April is the independent promotion gate; July and October are holdouts. Samples are capped at 300,000 training rows and 100,000 rows per evaluation window. These results describe historical 2023 data, not current NYC accuracy.

The `pretrip-v2` cohort does not filter on distance, passenger count, payment type, or rate code. It requires valid local timestamps, known geographic zones, and completed durations of 1–120 minutes. It includes EWR in the airport feature and excludes unknown-location placeholders.

No run automatically replaces a served model. Drift requests investigation or candidate training; promotion requires labeled improvement and sufficient segment evidence. Replay delays labels using an explicit simulated release schedule, not verified historical publication dates.

To predict with a particular completed run:

```python
import json
from pathlib import Path
import pandas as pd
from src.pretrip import predict

pointer = json.loads(Path("reports/validation_latest.json").read_text())
run_directory = Path(pointer["report"]).parent
inputs = pd.DataFrame([{
    "PULocationID": 161, "DOLocationID": 141,
    "pickup_datetime": "2023-07-10 08:00:00",
}])
print(predict(inputs, model_path=run_directory / "candidate.pkl"))
```

This is an explicit candidate prediction for inspection, not automatic deployment approval.

## Original notebook experiment (historical)

The steps below reproduce the earlier notebook-dependent experiment and its separate `pretrip_model.pkl` artifact. Use the validated pipeline above for new work.

### Step 1 — Download & Validate Data
Open and run `notebooks/1. load_validate_raw_data.ipynb`

### Step 2 — EDA & Feature Engineering
Open and run `notebooks/2. eda_feature_engineering.ipynb`

### Step 3 — Train the pre-trip model

```bash
python -m src.pretrip
```

This trains XGBoost on the existing processed dataset, using nine features derived from pickup zone, destination zone, and departure time. The final seven days are test data; the preceding seven are validation data. Validation controls early stopping. A mean predictor and the saved retrospective model (when present) are evaluated on the same rows.

Outputs:

- `models/pretrip_model.pkl`: estimator plus the geographic mapping and feature settings used at training time.
- `models/pretrip_metadata.json`: metrics, input contract, exact split boundaries, data fingerprint, and limitations.
- `reports/pretrip_comparison.json`: evaluation report.

The existing `models/best_model.pkl` remains the historical model. The pre-trip experiment does not write new MLflow runs; its results are recorded in the JSON artifacts and notebook. Historical notebook runs remain available in MLflow.

For the executed experiment and comparison, open [Pre-trip ETA baseline](output/jupyter-notebook/pretrip-eta-baseline.ipynb). Its training cell reruns the experiment.

### Step 4 — Predict locally

```python
import pandas as pd
from src.pretrip import predict

trips = pd.DataFrame([{
    "PULocationID": 161,
    "DOLocationID": 141,
    "pickup_datetime": "2023-01-25 08:00:00",
}])
print(predict(trips))  # Predicted duration in minutes
```

Use timezone-naive **America/New_York local time**. Unknown zone IDs and missing inputs are rejected. Time, borough, and airport features are generated internally; no actual distance or final rate code is required. These are zone-level estimates, without live traffic or exact-address routing.

### Step 5 — Run tests

```bash
python -m pytest tests/ -v
```

### Planned next stages

- Validate on later months and repeat explainability for the pre-trip model.
- Expand reusable data preparation beyond the current notebooks.
- Build the FastAPI endpoints described below.
- Implement drift detection, evaluation-gated retraining, and model promotion.

The earlier `src.train`, `src.api`, and `src.monitor` commands are not available yet. The project structure above includes the intended architecture, not just implemented files.

---

## MLflow Setup

MLflow tracks every training run (params, metrics, model artifacts) in a local SQLite database.

### First-time setup

```bash
cd nyc-taxi-duration-prediction-self-healing-ml

# Initialise the SQLite backend (run once)
mlflow db upgrade sqlite:///mlflow.db
```

### Launch the UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --dev
```

Then open `http://127.0.0.1:5000` in your browser.

> `--dev` disables MLflow 3.x security middleware for local single-user use.

### Navigating the UI

1. Click **nyc-taxi-duration** in the left sidebar (not "Default")
2. All runs are listed — sortable by `val_mae`, `val_rmse`, `val_r2`
3. Click any run to see its params, metrics, and saved model artifact
4. Check multiple runs → click **Compare** to overlay metrics in charts

### Tracking URI

Configured in `configs/config.yaml`:

```yaml
mlflow:
  experiment_name: nyc-taxi-duration
  tracking_uri: mlflow.db      # resolves to sqlite:///mlflow.db at runtime
```

The notebook builds the full URI as:
```python
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / cfg['mlflow']['tracking_uri']}"
```

### What gets logged per run

| Logged item | Example |
|---|---|
| Params | `n_estimators=500`, `max_depth=8`, `target_transform=none` |
| Val metrics | `val_mae`, `val_rmse`, `val_r2` |
| Test metrics | `test_mae`, `test_rmse`, `test_r2` |
| Best iteration | XGBoost / LightGBM early stopping round |
| Model artifact | Serialised estimator loadable via `mlflow.<flavour>.load_model()` |

---

## Key Design Decisions

See [docs/design_decisions.md](docs/design_decisions.md) for the complete decision register, implementation status, detailed rationale, and open questions.

| Decision | Choice |
|---|---|
| Train/val/test split | Time-based: final 7 days test, preceding 7 days validation |
| Primary metric | MAE (interpretable in minutes) |
| Feature set | Pre-trip: 9 derived features; historical benchmark: 12 |
| Pre-trip input contract | Pickup zone, destination zone, local departure time |
| Drift response | Investigate/train a candidate; promotion requires an independent labeled gate |

---

## Historical Model Feature Reference

These 12 features describe the original retrospective model. The pre-trip model excludes `trip_distance`, `RatecodeID`, and `passenger_count`, and derives its remaining nine features internally.

| Feature | Type | Description |
|---|---|---|
| `PULocationID` | Categorical | TLC pickup zone ID (1–265) |
| `DOLocationID` | Categorical | TLC dropoff zone ID (1–265) |
| `trip_distance` | Numeric | Miles reported by taximeter |
| `passenger_count` | Numeric | Number of passengers (1–6) |
| `RatecodeID` | Categorical | 1=Standard, 2=JFK, 3=Newark, 4=Nassau, 5=Negotiated |
| `hour` | Numeric | Hour of pickup (0–23) |
| `dayofweek` | Numeric | Day of week (0=Mon, 6=Sun) |
| `is_weekend` | Binary | 1 if Saturday or Sunday |
| `is_rush_hour` | Binary | 1 if hour in {7,8,9,17,18,19} |
| `PU_borough_id` | Categorical | Encoded borough of pickup zone |
| `DO_borough_id` | Categorical | Encoded borough of dropoff zone |
| `is_airport_trip` | Binary | 1 if pickup or dropoff is at JFK/LaGuardia/EWR |

---

## Planned API Endpoints

These endpoints are not implemented yet.

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Predict trip duration in minutes |
| `GET` | `/health` | Health check |
| `GET` | `/model-info` | Model name, training date, validation MAE |
