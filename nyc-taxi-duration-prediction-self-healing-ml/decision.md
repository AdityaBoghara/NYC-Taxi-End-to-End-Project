# Project Decision Register

NYC Taxi Trip Duration Prediction — Self-Healing ML System

Consolidated on 2026-09-18 from the existing design record, notebook code, configuration, and saved model metadata. This records existing choices; it does not approve new implementation proposals. Original decision dates were not consistently recorded.

## Status and evidence

- **Implemented:** present in the notebooks, configuration, or saved artifacts. This does not imply a production service exists.
- **Planned:** selected in project documentation or configuration, but not implemented as a working service.
- **Deferred/open:** an alternative or follow-up without a final implementation decision.

Sources:

- [Original design decisions](docs/design_decisions.md)
- [Project overview and intended architecture](README.md)
- [Configuration](configs/config.yaml) and [configuration loader](src/config.py)
- [Data loading notebook](notebooks/1.%20load_validate_raw_data.ipynb)
- [EDA and feature engineering notebook](notebooks/2.%20eda_feature_engineering.ipynb)
- [Baseline training notebook](notebooks/3.%20baseline_model_training.ipynb)
- [FLAML comparison notebook](notebooks/Side%20Quest%3A%20automl_flaml_comparison.ipynb)
- [SHAP notebook](notebooks/4.%20shap_explainability.ipynb)
- [Saved model metadata](models/model_metadata.json) (local, git-ignored artifact)

Where prose and code disagree, the implementation details below follow the notebook code and saved metadata. The existing design document remains the historical rationale reference.

## 1. Problem, data, and storage

| ID | Decision | Reason and limitations | Status |
|---|---|---|---|
| D01 | Predict `trip_duration_min`, computed as dropoff minus pickup in minutes. | Provides an interpretable regression target; excludes time before the recorded pickup. | Implemented |
| D02 | Use official NYC TLC yellow taxi Parquet records, initially January 2023. | Establishes a baseline on approximately 3 million raw trips. Generalization to later months remains unverified. | Implemented |
| D03 | Retain all boroughs and add borough features instead of restricting to Manhattan. | EDA found meaningful geographic differences; retains outer-borough and airport coverage. | Implemented |
| D04 | Store raw, interim, and processed datasets separately in Parquet. | Preserves types and makes transformation stages inspectable. Paths use year/month templates; the existing interim filename prefix is `intrim_rides_`. | Implemented |
| D05 | Keep `pickup_datetime` in the processed dataset but exclude it from model inputs. | Required for chronological train/validation/test splits. Downstream notebooks check that it exists. | Implemented |
| D06 | Keep shared paths, filters, features, model defaults, and monitoring settings in `configs/config.yaml`. | Centralizes configuration. `src/config.py` exposes constants; notebooks also load YAML directly, and some settings remain notebook-local. | Implemented |

## 2. Data quality and filtering

These are baseline modeling rules, not proof that every excluded record is invalid.

| ID | Decision | Reason and limitations | Status |
|---|---|---|---|
| D07 | Restrict pickups to the downloaded month. | Avoids adjacent-month submissions. Notebook 1 currently uses `pickup_datetime > 2023-01-01` and `< 2023-02-01`, excluding exactly midnight at the start as well. | Implemented |
| D08 | Remove negative durations. | Dropoff-before-pickup records cannot represent the intended target; three were identified in the January analysis. | Implemented |
| D09 | Keep durations from 1 through 120 minutes. | Limits very short and extreme-duration records. This is a fixed heuristic and excludes some potentially legitimate edge cases. | Implemented |
| D10 | Remove the `payment_type = 0` block rather than impute its missing metadata. | The existing analysis identifies 71,742 Flex Fare records with missing passenger/rate metadata. Modeling this segment separately is deferred. | Implemented |
| D11 | Keep distances greater than 0 and at most 50 miles. | Excludes missing/zero distance and extreme readings; can also exclude legitimate long trips. | Implemented |
| D12 | Keep passenger counts from 1 through 6. | Applies the project's baseline quality threshold instead of imputing zero or extreme counts. | Implemented |
| D13 | Do not remove additional records on fare values alone. | Financial values are examined for data quality but excluded from predictors. | Implemented |
| D14 | Keep stored-and-forwarded (`Y`) records that pass other filters. | EDA did not identify a material duration-distribution difference; exclude the transmission flag itself from predictors. | Implemented |
| D15 | Keep rate code 99 records that pass other filters. | Unknown rate code alone is not used to reject a trip; uncertainty within this segment remains a limitation. | Implemented |
| D16 | Drop remaining nulls in the selected final dataset. | Produces complete model inputs without inventing missing values. Duplicate counts are inspected, but no deduplication policy is established. | Implemented |

## 3. Features and prediction assumptions

| ID | Decision | Reason and limitations | Status |
|---|---|---|---|
| D17 | Use 12 features in the current saved model. | Combines distance, passenger count, trip type, time, and geography. Exact order is recorded below and in model metadata. | Implemented |
| D18 | Retain raw integer pickup/dropoff zone IDs and integer-encoded boroughs. | Keeps fine and coarse spatial information. Integer coding imposes an ordering; target encoding remains a candidate improvement. | Implemented |
| D19 | Build one borough map from sorted observed pickup-borough names and apply it to both pickup and dropoff. | Keeps the two columns on the same mapping within a run. A stable persisted mapping for serving and future data is still needed. | Implemented |
| D20 | Derive `hour`, `dayofweek`, `is_weekend`, and `is_rush_hour`. | Monday is 0; weekend is day 5 or 6; rush hours are `{7,8,9,17,18,19}`. Hour/day remain integer values, without sine/cosine encoding. | Implemented |
| D21 | Set `is_airport_trip` when either joined service-zone value equals `Airports`. | Adds a direct trip-type signal. Actual airport coverage depends on the lookup table; the code does not independently enumerate all airport zone IDs. | Implemented |
| D22 | Retain integer `RatecodeID` as a trip-type feature; discard the EDA helper `rate_label`. | Project documentation treats rate code as available at dispatch. Production input availability must follow the eventual serving contract. | Implemented |
| D23 | Retain passenger count despite low observed importance. | Preserves the current feature schema. Also retain correlated flags such as weekend/day-of-week and airport/rate-code. | Implemented |
| D24 | Exclude financial columns, payment type, vendor ID, transmission flag, dropoff timestamp, and duration-in-seconds from predictors. | Financial/payment/dropoff values are post-trip; vendor and transmission fields were judged unhelpful. Duration-in-seconds would directly reveal the target. | Implemented |
| D25 | Use recorded `trip_distance` in the baseline. | It is a strong predictor, but the final metered distance is unavailable before a trip. A pre-trip route-distance estimate or a different product definition remains unresolved. | Implemented baseline; serving assumption open |
| D26 | Allow notebook geographic fallback when the lookup is unavailable. | Notebook 2 assigns borough IDs `-1` and airport flag `0`; final assembly can omit unavailable geographic columns. This can change the schema and is not a finalized production fallback policy. | Implemented notebook behavior |

Current saved feature order:

```text
trip_distance, passenger_count, hour, dayofweek,
PU_borough_id, DO_borough_id, PULocationID, DOLocationID,
RatecodeID, is_weekend, is_rush_hour, is_airport_trip
```

Excluded financial fields: `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `total_amount`, `congestion_surcharge`, and `airport_fee`.

## 4. Training, evaluation, and model selection

| ID | Decision | Reason and limitations | Status |
|---|---|---|---|
| D27 | Use chronological train/validation/test partitions rather than random splitting. | Evaluate on later trips. Code sets test cutoff to maximum pickup timestamp minus 7 days, and validation cutoff another 7 days earlier. Train is `<= val_cutoff`, validation is `> val_cutoff` and `<= test_cutoff`, and test is `> test_cutoff`. | Implemented |
| D28 | Select the best model by validation MAE; report RMSE and R² as secondary metrics. | MAE expresses average absolute error in minutes. Test metrics are reported separately rather than used by the selection condition. | Implemented |
| D29 | Compare six baseline configurations: mean dummy, scaled linear regression, random forest, raw-target XGBoost, log-target XGBoost, and LightGBM. | Establishes simple and nonlinear reference models. There are six configurations across five model families. | Implemented |
| D30 | Put `StandardScaler` inside the linear regression pipeline and clip its evaluation predictions to 1–120 minutes. | Standardizes inputs and bounds baseline predictions. Clipping is evaluation-side code, not part of the serialized linear pipeline. | Implemented |
| D31 | Compare raw targets with `log1p` targets for XGBoost; use `expm1` before evaluating log-model predictions. | Tests a transformation for the skewed target while keeping reported errors in minutes. The selected FLAML model uses the raw target. | Implemented |
| D32 | Use validation-based early stopping with 50 rounds for baseline XGBoost and LightGBM. | Limits unnecessary boosting rounds. This is distinct from FLAML's search budget. | Implemented |
| D33 | Use random seed 42 for configured model randomness and sampling. | Supports repeatable comparisons, without guaranteeing identical results across environments or timed searches. | Implemented |
| D34 | Run FLAML over XGBoost, LightGBM, and random forest with a 300-second budget and MAE objective. | Uses the same explicit chronological validation set with holdout evaluation for comparable tuning. | Implemented |
| D35 | Replace the baseline artifact only if FLAML has lower validation MAE. | Avoids replacing a stronger baseline. This is a notebook overwrite rule, not a production promotion/rollback system. | Implemented |
| D36 | Skip a separate Optuna tuning stage for now. | The design record treats FLAML as sufficient initial hyperparameter optimization and prioritizes feature/data improvements. Older notebook next-step text mentioning Optuna is superseded. | Recorded decision |
| D37 | Save the selected estimator as `models/best_model.pkl` and metadata as `models/model_metadata.json`. | Metadata records feature order, target transform, split dates, scores, hyperparameters, and seed for downstream inference. Models are local artifacts. | Implemented |

Recorded current winner (existing saved results; not rerun for this document):

| Item | Value |
|---|---|
| Model | `flaml_xgboost` |
| Training month | January 2023 |
| Target transform | None |
| Validation MAE / RMSE / R² | 2.4651 min / 3.9683 min / 0.8661 |
| Test MAE / RMSE / R² | 2.5740 min / 4.2092 min / 0.8467 |
| Baseline XGBoost validation MAE | 2.5457 min |
| Relative validation MAE improvement | Approximately 3.17% |

The notebooks label the split approximately Jan 1–17 / Jan 18–24 / Jan 25–31. Actual boundaries retain the time of day from the maximum pickup timestamp; they are not explicitly normalized to midnight.

## 5. Experiment tracking and explainability

| ID | Decision | Reason and limitations | Status |
|---|---|---|---|
| D38 | Track experiments in MLflow under `nyc-taxi-duration`, using a local SQLite database. | Provides persistent local experiment tracking without a separate database server. A shared multi-user backend is deferred. | Implemented in notebooks |
| D39 | Log model parameters, validation/test metrics, model artifacts, and applicable best-iteration information. | Supports model comparison and inspection of how artifacts were produced. | Implemented |
| D40 | Use SHAP TreeExplainer on a 5,000-row test sample before serving work. | Examines global, feature-level, individual-trip, and segment behavior. Results are limited to the sampled January distribution. | Implemented |
| D41 | Save SHAP importance, beeswarm, dependence, waterfall, and segment plots under `reports/`. | Preserves inspectable explanations. The design record reports six checks passed and ten SHAP plots were produced. | Implemented |

SHAP consistency checks support interpretation of the model; they do not prove causality, eliminate leakage, or establish robustness to future drift.

## 6. Serving and self-healing architecture

These choices are documented intentions. At consolidation time, `src/` contains only `config.py` and `__init__.py`; `tests/` contains only `__init__.py`.

| ID | Decision | Reason and remaining work | Status |
|---|---|---|---|
| D42 | Organize reusable code into data loading, feature engineering, training, evaluation, prediction, monitoring, and API modules. | Separates pipeline responsibilities and enables execution outside notebooks. These modules still need implementation. | Planned |
| D43 | Serve predictions through FastAPI with `POST /predict`, `GET /health`, and `GET /model-info`. | Provides an inference interface and basic service/model inspection. Configured address is `0.0.0.0:8000`; API code is absent. | Planned |
| D44 | Use Evidently for drift detection. | Compare incoming feature distributions with reference data. Reference windows, statistical settings, and logging still need implementation. | Planned |
| D45 | Trigger retraining when MAE degrades by more than 15% or more than 30% of features drift. | Records the configured initial thresholds. Requires a defined performance baseline, actual-duration labels, and drift calculation. | Planned; thresholds configured |
| D46 | Check monitoring conditions every 6 hours. | Establishes the configured default cadence. No scheduler/monitor loop is implemented. | Planned; interval configured |
| D47 | Use pytest for automated tests. | The dependency and test command are documented, but no actual tests have been added. | Planned |

## 7. Deferred choices and unresolved follow-ups

These are not finalized implementation decisions.

| Topic | Current position / unresolved question |
|---|---|
| Zone target encoding | Highest-priority feature enhancement in the design record; requires leakage-safe fitting and evaluation. Raw zone IDs remain the baseline. |
| Learned zone embeddings | Mentioned as a possible alternative; no architecture selected. |
| Holiday flag | Deferred as low priority for the baseline. |
| Multi-month training | January-only baseline retained; duration and months of expanded training remain open. |
| Outlier policy | Fixed 120-minute and 50-mile limits retained; percentile-based alternatives remain open. |
| Flex Fare coverage | Separate segment modeling deferred pending usable metadata. |
| Geographic lookup | Local caching, stable persisted encoding, unknown-zone handling, and production fallback need a final contract. |
| Distance available at prediction time | Decide how a pre-trip distance estimate is obtained and evaluate the model with that input. |
| Removing passenger count | Low signal documented; retained in the current schema. |
| Shared MLflow backend | PostgreSQL is mentioned for future concurrent use; local SQLite remains the current choice. |
| Production model replacement | Promotion criteria, versioning, rollback, and safe reload are unresolved; notebook artifact replacement is the only implemented selection mechanism. |
| Deployment and CI | No hosting platform, container strategy, or CI workflow has been selected in the inspected sources. |

## 8. Documentation discrepancies to retain in context

- The README shows planned modules and commands as if available; the module inventory above reflects the actual implementation state.
- Older summaries describe the final seven days as validation. Notebook code uses the final seven days for test and the preceding seven days for validation.
- The AutoML notebook still calls itself Notebook 4 and proposes later Optuna/SHAP work. The actual SHAP file is `4. shap_explainability.ipynb`; the design record skips separate Optuna tuning.
- Notebook MLflow configuration builds a `sqlite:///...` URI, while `src/config.py` currently exposes a filesystem path as `MLFLOW_TRACKING_URI`. Reusable training code must reconcile this before relying on that constant.
- Claims that every feature is available before departure require resolving the recorded-distance assumption in D25.

When a choice changes, record its new status, rationale, and implementation evidence here rather than silently treating a proposal as completed work.
