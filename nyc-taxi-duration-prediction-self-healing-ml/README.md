# NYC Taxi Trip Duration Prediction — Self-Healing ML System

An end-to-end machine learning system that predicts NYC yellow taxi trip duration in minutes, with automated drift detection and self-healing retraining.

## Project Overview

| Component | Description |
|---|---|
| **Data** | NYC TLC Yellow Taxi trip records (Jan 2023, ~3M trips) |
| **Target** | `trip_duration_min` — elapsed time from pickup to dropoff |
| **Models** | XGBoost / LightGBM (best chosen by validation MAE) |
| **Tracking** | MLflow experiment tracking for every run |
| **Serving** | FastAPI REST API (`POST /predict`) |
| **Monitoring** | Evidently drift detection + automatic retraining trigger |

---

## Project Structure

```
nyc-taxi-duration-prediction-self-healing-ml/
├── data/
│   ├── raw/            # Unmodified TLC download
│   ├── interim/        # Date-filtered, negatives removed
│   └── processed/      # Model-ready (12 features + target)
├── notebooks/
│   ├── 1. load_validate_raw_data.ipynb
│   └── 2. eda_feature_engineering.ipynb
├── src/
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

## Running the Pipeline

### Step 1 — Download & Validate Data
Open and run `notebooks/1. load_validate_raw_data.ipynb`

### Step 2 — EDA & Feature Engineering
Open and run `notebooks/2. eda_feature_engineering.ipynb`

### Step 3 — Train Models
```bash
python -m src.train
```
This trains 5 models (Dummy → Linear → Random Forest → XGBoost → LightGBM), logs all runs to MLflow, and saves the best model to `models/best_model.pkl`.

View results:
```bash
mlflow ui
# Open http://localhost:5000
```

### Step 4 — Serve the API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Predict:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "PULocationID": 161, "DOLocationID": 141,
    "trip_distance": 2.5, "passenger_count": 1,
    "RatecodeID": 1, "hour": 8, "dayofweek": 1,
    "is_weekend": 0, "is_rush_hour": 1,
    "PU_borough_id": 3, "DO_borough_id": 3,
    "is_airport_trip": 0
  }'
```

### Step 5 — Monitor for Drift
```bash
python -m src.monitor
```
Checks for data drift and MAE degradation every 6 hours (configurable). Automatically triggers retraining if thresholds are exceeded.

### Step 6 — Run Tests
```bash
pytest tests/ -v
```

---

## Key Design Decisions

See [docs/design_decisions.md](docs/design_decisions.md) for full rationale.

| Decision | Choice |
|---|---|
| Train/val split | Time-based (last 7 days = validation) |
| Primary metric | MAE (interpretable in minutes) |
| Feature set | 12 features — time, location, trip type |
| No leakage | All financial columns excluded (post-trip values) |
| Drift threshold | MAE +15% or >30% features drifted → retrain |

---

## Feature Reference

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

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Predict trip duration in minutes |
| `GET` | `/health` | Health check |
| `GET` | `/model-info` | Model name, training date, validation MAE |
