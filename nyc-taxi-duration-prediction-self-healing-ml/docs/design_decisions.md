# Design Decisions

NYC Taxi Trip Duration Prediction — Self-Healing ML System

Single source of truth for project decisions, implementation status, rationale, and open follow-ups. Consolidated on 2026-09-18 from the original design record, notebook code, configuration, and saved model metadata. This records existing choices; it does not approve new implementation proposals. Original decision dates were not consistently recorded.

Use sections 1–6 for the decision register, section 7 for open choices, section 8 for documentation discrepancies, and [Detailed rationale](#detailed-rationale) for supporting analysis. D01–D47 are the stable decision identifiers.

## Status and evidence

- **Implemented:** present in the notebooks, configuration, or saved artifacts. This does not imply a production service exists.
- **Planned:** selected in project documentation or configuration, but not implemented as a working service.
- **Deferred/open:** an alternative or follow-up without a final implementation decision.

Sources:

- [Project overview and intended architecture](../README.md)
- [Configuration](../configs/config.yaml) and [configuration loader](../src/config.py)
- [Data loading notebook](../notebooks/1.%20load_validate_raw_data.ipynb)
- [EDA and feature engineering notebook](../notebooks/2.%20eda_feature_engineering.ipynb)
- [Baseline training notebook](../notebooks/3.%20baseline_model_training.ipynb)
- [FLAML comparison notebook](../notebooks/Side%20Quest%3A%20automl_flaml_comparison.ipynb)
- [SHAP notebook](../notebooks/4.%20shap_explainability.ipynb)
- [Saved model metadata](../models/model_metadata.json) (local, git-ignored artifact)

Where prose and code disagree, the implementation details below follow the notebook code and saved metadata. The detailed rationale below preserves the original analysis, with clarifications where its conclusions exceeded the evidence.

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

## Detailed rationale

The following sections retain the original numbered rationale and EDA observations. Dataset-specific counts and interpretations reflect the original January analysis, not a new validation run. Filtering thresholds are modeling heuristics; they do not prove that all excluded trips are invalid. Current implementation status and qualifications are recorded in D01–D47 above. Open decisions are maintained only in section 7.

### 1. Data Source

**Register entries:** D02.

**Decision:** Use the NYC TLC Yellow Taxi trip records (Parquet format) from the official TLC S3-compatible endpoint.

**Rationale:**
- Authoritative source maintained by the NYC Taxi & Limousine Commission
- Parquet format is efficient for columnar reads and reduces I/O compared to CSV
- Data is publicly available and versioned by year/month

**Trade-offs:**
- Data lags by ~2 months (no real-time feed)
- Schema has changed across years; scripts must handle version differences

---

### 2. Geographic Scope ✅ Resolved

**Register entries:** D03, D18–D19.

**Decision:** Keep **all boroughs**. Encode pickup and dropoff borough as model features using the TLC zone lookup table.

**Rationale (from EDA):**  
Borough-level EDA revealed large, statistically significant differences in trip duration:
- Newark/EWR trips: median ~55–70 min
- Airport (Queens) trips: median ~35–50 min
- Manhattan trips: median ~12–14 min
- Outer boroughs: median ~18–25 min

Restricting to Manhattan would exclude ~15 % of trips and would cause the model to fail silently on all out-of-borough predictions. Encoding borough as a feature is strictly better than dropping it.

**Implementation:**  
Join the TLC zone lookup CSV to add `PU_borough_id` and `DO_borough_id` (label-encoded integers, suitable for tree-based models).

**Trade-offs:**
- Borough join introduces a network dependency (TLC CSV endpoint) — should be cached locally for offline runs
- Label encoding (arbitrary integers) is suboptimal for linear models; target encoding is a future enhancement

---

### 3. Target Variable

**Register entries:** D01.

**Decision:** Predict `trip_duration_min` — the elapsed time from pickup to dropoff in **minutes**.

**Rationale:**
- Minutes are human-interpretable (vs raw seconds)
- Regression on minutes aligns with standard taxi ETA benchmarks
- Derived as: `(dropoff_datetime - pickup_datetime).total_seconds() / 60`

**Trade-offs:**
- Does not account for waiting time at pickup (meter starts on dispatch in some cases)
- Ignores traffic conditions not captured in the structured features

---

### 4. Outlier Filtering — Duration

**Register entries:** D09.

**Decision:** Remove trips with `trip_duration_min < 1` or `trip_duration_min > 120`.

**Rationale:**
- Trips under 1 minute are almost certainly GPS errors, meter test runs, or cancelled rides — not physically completable taxi trips
- Trips over 120 minutes are rare edge cases (forgotten meters, disputes, system errors) that would skew model training and inflate error metrics
- The 120-minute cap covers JFK and LaGuardia airport runs comfortably (~45–75 min in traffic)
- This removes ~1.09 % (< 1 min) + ~0.11 % (> 120 min) = ~1.2 % of data — acceptable loss

**Percentile reference (post null-block drop):**
- 0.1th percentile: ~4 seconds → confirms < 60s trips are errors
- 99.9th percentile: ~175 minutes → extreme tail; 120-min cap is conservative

**Trade-offs:**
- A percentile-based filter (e.g. drop top 0.5%) would be more data-driven; the 120-min cap is a domain heuristic
- Revisit if the project scope expands beyond NYC (e.g., longer suburban trips)

---

### 5. Outlier Filtering — Negative Durations

**Register entries:** D08.

**Decision:** Remove trips where `trip_duration < 0` (dropoff recorded before pickup).

**Rationale:**
- 3 such records were found in Jan 2023 data; all had `RatecodeID = 99` (unknown/disputed)
- These are data entry errors and cannot represent real trips

**Trade-offs:**
- Negligible impact (3 out of ~3M rows)

---

### 6. Date Range Filtering

**Register entries:** D07.

**Decision:** Keep only rides where `pickup_datetime` falls strictly within the downloaded month (Jan 2023).

**Rationale:**
- The raw parquet occasionally contains records from adjacent months due to late submission by vendors
- Mixing months would introduce temporal leakage when the model is evaluated month-by-month

---

### 7. Flex Fare Block — Rows with payment_type = 0

**Register entries:** D10.

**Decision:** Drop all 71,742 rows where `payment_type = 0`.

**Rationale (from EDA + data dictionary):**  
The TLC data dictionary (updated March 2025) defines `payment_type = 0` as **"Flex Fare trip"** — a dynamic/surge-priced ride that operates outside standard metered rates.

These rows also form a **coherent metadata-null block**: five columns are simultaneously null in all 71,742 rows:
`passenger_count`, `RatecodeID`, `store_and_fwd_flag`, `congestion_surcharge`, `airport_fee`.

This is not a vendor submission error — it reflects how Flex Fare trips are recorded:
- No `RatecodeID` because Flex Fare pricing is not governed by the standard rate code system
- No `congestion_surcharge` because the surcharge calculation differs for dynamic fares
- `passenger_count` and `store_and_fwd_flag` are absent, likely because the Flex Fare dispatch system does not populate these fields

**Why not keep them?**
- `RatecodeID` is a key model feature — null rate code cannot be used or imputed without fabricating trip type
- `passenger_count` cannot be reliably imputed (Flex Fare trips may have a different passenger distribution)
- Flex Fare trips operate under dynamic pricing rules that make their duration distribution structurally distinct from standard metered trips — mixing them into the same training set would introduce noise
- They represent only ~2.3 % of the dataset — the loss is acceptable

**Future option:** Model Flex Fare trips as a separate segment once the full metadata fields are available in the dataset.

**Trade-offs:**
- Loss of 71,742 rows (~2.3 % of data)
- If Flex Fare adoption grows significantly in future months, this exclusion becomes a larger gap

---

### 8. Trip Distance Filtering

**Register entries:** D11, D25.

**Decision:** Remove trips where `trip_distance = 0` or `trip_distance > 50` miles.

**Rationale (from EDA):**

**Zero-distance trips (~45,856 rows):**
- These trips have real, nonzero durations (often 5–30 minutes) despite zero GPS-recorded distance
- Possible causes: GPS not acquired (meter started inside a building), odometer failure, waiting-time records
- No model can learn a meaningful relationship between 0 miles and a duration — 0 is a data gap, not a real measurement
- Including them teaches a spurious "zero distance → random duration" pattern

**Extreme distances (> 50 miles):**
- NYC's farthest legitimate trip (JFK to Bronx/Upper Manhattan) is ~25 miles
- Values > 50 miles are GPS or odometer malfunctions
- 99.99th percentile of trip_distance is ~30 miles; anything beyond is an error

**Trade-offs:**
- The 50-mile cap is generous; a tighter cap (30 miles) would remove a few hundred additional suspicious records
- Very rare legitimate long trips (e.g., Manhattan to eastern Suffolk County) are excluded

---

### 9. Passenger Count Filtering

**Register entries:** D12, D23.

**Decision:** Remove trips where `passenger_count = 0` or `passenger_count > 6`.

**Rationale (from EDA):**

**passenger_count = 0 (~51,164 rows after other filters):**
- A completed, fare-generating taxi trip must have at least one passenger
- These are most likely meter-on events without a passenger (driver testing, GPS initialization)
- Imputing with 1 would be wrong: these rows may be qualitatively different trips (no-shows, test runs)

**passenger_count > 6:**
- NYC yellow taxi maximum legal capacity is 4 passengers (standard cab) or 5 passengers (larger vehicles)
- Counts of 7, 8, 9 are data entry errors

**Trade-offs:**
- passenger_count has very low correlation with trip_duration (~r = 0.01); dropping these rows primarily improves data quality rather than model accuracy

---

### 10. Feature Engineering

**Register entries:** D17–D26.

#### 10a. Time-based features

| Feature | Derivation | Rationale |
|---|---|---|
| `hour` | `pickup_datetime.dt.hour` | Captures intra-day demand and traffic patterns (strongest time signal) |
| `dayofweek` | `pickup_datetime.dt.dayofweek` (0=Mon) | Captures weekly seasonality |
| `is_weekend` | `dayofweek >= 5` | Binary flag; weekend trips differ structurally from weekdays (leisure vs commute) |
| `is_rush_hour` | `hour in {7,8,9,17,18,19}` | Morning and evening peak hours drive 30–40% longer median durations |

**Empirical validation (from EDA):**
- Median duration at 8 AM is ~40–50 % higher than at 3 AM
- Friday evening (17–19 hr) is the single worst congestion window of the week
- Weekend midday trips are longer than weekday midday — leisure travel patterns differ from commuting

**Trade-offs:**
- Rush hour window is fixed (data-driven from EDA); a learned threshold might fit better in different months
- No holiday flag — NYC holidays significantly reduce traffic. A holiday calendar is a low-priority future enhancement

#### 10b. Location features

**Decision:** Keep `PULocationID` and `DOLocationID` as raw integer zone IDs **and** add `PU_borough_id` / `DO_borough_id` from the TLC zone lookup.

**Rationale:**
- Zone IDs encode fine-grained spatial information; tree-based models handle high-cardinality categorical integers well
- Borough IDs add coarser but more interpretable geographic context
- Providing both lets the model learn at two spatial resolutions

**Known gap:** High cardinality (~260 zones). Future options:
- Target-encode zones using mean trip duration (requires careful cross-validation to avoid leakage)
- Embed zones using a learned representation (e.g., entity embeddings in a neural net)

#### 10c. Airport trip flag

**Decision:** Add `is_airport_trip` binary feature (1 if PU or DO service_zone = 'Airports').

**Rationale (from EDA):**
- Airport trips have median duration 2–3× higher than standard city trips
- Their duration is dominated by highway congestion, not city street density — a structurally different regime
- The flag is a known-at-dispatch value (driver and dispatcher know the destination)
- `RatecodeID = 2` (JFK flat-rate) already partially captures this, but LaGuardia trips use standard metered rate

**Trade-offs:**
- Requires the zone lookup join to be successful at runtime

#### 10d. Excluded columns (leakage prevention)

The following columns are **excluded** from the model feature set because they are only known *after* the trip ends (data leakage) or are irrelevant to duration:

**Post-trip financial values:**
- `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `total_amount`, `congestion_surcharge`, `airport_fee`

**Post-trip or non-causal:**
- `payment_type` — selected by passenger at trip end
- `store_and_fwd_flag` — transmission mode indicator, not a trip attribute
- `VendorID` — no meaningful duration difference between vendors (confirmed in EDA). Vendor 2 is listed as "Curb Mobility, LLC" in the 2025 data dictionary (formerly VeriFone — same fleet, rebranded)

**Redundant datetime columns:**
- `pickup_datetime` — retained in the processed parquet for time-based train/val/test splits in downstream notebooks; excluded from the model feature matrix at training time
- `dropoff_datetime` — excluded (post-trip value, causes leakage)
- `trip_duration` (seconds) — replaced by `trip_duration_min` (target)

---

### 11. RatecodeID — Retention and Treatment

**Register entries:** D15, D22.

**Decision:** Retain `RatecodeID` as a model feature (cast to integer).

**Rationale (from EDA):**
- RatecodeID encodes the trip type selected at dispatch — it is a known-at-pickup value
- JFK flat-rate trips (code 2) have a median duration ~3–4× higher than standard metered trips (code 1)
- Newark (code 3) and Nassau/Westchester (code 4) also represent structurally longer trips
- The feature adds direct signal that is not fully captured by location IDs alone

**RatecodeID reference:**
| Code | Meaning | Median duration (approx.) |
|---|---|---|
| 1 | Standard metered rate | ~12 min |
| 2 | JFK flat rate | ~50 min |
| 3 | Newark | ~60 min |
| 4 | Nassau/Westchester | ~45 min |
| 5 | Negotiated fare | ~20 min |
| 6 | Group ride | ~15 min |
| 99 | Unknown/Disputed | ~13 min |

**Trade-offs:**
- Code 99 (unknown/disputed) trips have wider duration variance; could be flagged as a separate quality segment

---

### 12. Data Storage Format

**Register entries:** D04–D05.

**Decision:** Use Parquet at every pipeline stage.

| Stage | Path | Contents |
|---|---|---|
| Raw | `data/raw/rides_YYYY-MM.parquet` | Unmodified TLC download |
| Interim | `data/interim/intrim_rides_YYYY-MM.parquet` | Date-filtered, negative durations removed, `trip_duration` added |
| Processed | `data/processed/processed_rides_YYYY-MM.parquet` | All filters applied, features engineered, model-ready |

**Rationale:**
- Parquet preserves dtypes (avoids datetime parsing on reload)
- Columnar format is efficient when only a subset of columns is read downstream
- Separating raw/interim/processed makes each transformation stage independently inspectable and re-runnable

**Known pipeline constraint — `pickup_datetime` in processed parquet:**
- The processed parquet must include `pickup_datetime` as its first column even though it is not a model feature
- It is required for the time-based train/val/test split in all downstream notebooks (3, 4, Side Quest)
- If Notebook 2 is re-run from a stale kernel state, `pickup_datetime` can be dropped silently — downstream notebooks now raise a `ValueError` with a clear fix message rather than a `KeyError`

---

### 13. MLflow Backend

**Register entries:** D38–D39.

**Decision:** Use SQLite (`mlflow.db`) as the MLflow tracking backend instead of the default filesystem store (`mlruns/`).

**Rationale:**
- MLflow 3.x deprecated the filesystem backend (February 2026) — it will be removed in a future release
- SQLite enables the full MLflow feature set: model registry, run comparison, job execution support
- Zero additional infrastructure — SQLite is a single file, no separate server needed
- One-time migration: `mlflow db upgrade sqlite:///mlflow.db`

**Configuration:**
- `config.yaml`: `mlflow.tracking_uri: mlflow.db`
- Notebooks construct the full URI at runtime: `sqlite:///{PROJECT_ROOT}/mlflow.db`
- UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db --dev`

**Trade-offs:**
- `--dev` flag required in MLflow 3.x to disable security middleware for local single-user use
- SQLite is not suitable for multi-user concurrent writes — migrate to PostgreSQL if the project moves to a shared server

---

### 14. Model Selection — Baseline + AutoML

**Register entries:** D27–D37.

**Decision:** Use a two-stage selection process: manual baseline (Notebook 3) followed by FLAML AutoML search (`Side Quest: automl_flaml_comparison.ipynb`). The better model by val MAE overwrites `models/best_model.pkl`.

**Rationale:**
- Notebook 3 establishes interpretable baselines across six configurations from five model families with fixed hyperparameters
- FLAML searches hundreds of XGBoost/LightGBM/RF configurations within a time budget (300s default), using the same time-based val set for fair comparison
- FLAML found a better XGBoost config: **val MAE 2.4651 vs baseline 2.5457 (3.17% improvement)**

**Hyperparameter tuning (Optuna) — skipped:**
- FLAML's search already constitutes automated hyperparameter optimisation
- The observed 3.17% gain motivated prioritizing feature and data improvements; it does not establish a performance ceiling
- Future gains are more likely from feature engineering (zone target encoding, multi-month training) than further hyperparameter search

**Best model as of Jan 2023 baseline:**
| | Value |
|---|---|
| Model | `flaml_xgboost` |
| Val MAE | 2.4651 min |
| Val R² | 0.8661 |
| Saved to | `models/best_model.pkl` + `models/model_metadata.json` |

---

### 15. Model Explainability — SHAP TreeExplainer

**Register entries:** D40–D41.

**Decision:** Use SHAP TreeExplainer (exact Shapley values) to validate the FLAML XGBoost model before moving to serving. Applied to a 5,000-row sample of the Jan 2023 test set (Jan 25–31).

**Rationale:**
- TreeExplainer computes exact (not approximate) Shapley values by traversing the XGBoost tree structure directly — the correct tool for any gradient-boosted tree model
- Explainability is a pre-deployment gate: a model making good predictions for wrong reasons will fail silently under data shift, which is the exact scenario the self-healing system monitors for
- A 5,000-row sample limits computation while supporting inspection of common patterns; rare segments may be underrepresented

**Analysis produced (Notebook 4):**

| Analysis | What it shows |
|---|---|
| Global importance bar + beeswarm | Which features move predictions most, and in which direction |
| Dependence plots (top 4 features) | How each feature's SHAP value changes as the feature value increases |
| Waterfall plots (2 trips) | Full per-feature breakdown of one short city trip and one JFK airport trip |
| Segment analysis | Mean SHAP contributions grouped by rate code, hour of day, and airport vs standard |

**Validation results — all 6 checks passed:**

| Signal | Expected | Result |
|---|---|---|
| `trip_distance` is #1 feature | Yes | Largest mean \|SHAP\| by wide margin |
| `hour` captures rush-hour congestion | Yes | SHAP peaks at 8–9 AM and 17–19 PM |
| `RatecodeID=2` (JFK) adds large positive SHAP | Yes | JFK mean SHAP >> standard rate SHAP |
| `is_airport_trip` contributes positively | Yes | Meaningful positive effect |
| `passenger_count` near zero | Yes | Near-zero mean \|SHAP\| — low-signal feature |
| Zone IDs (`PULocationID`, `DOLocationID`) contribute | Yes | Fine-grained spatial signal confirmed |

**Key takeaway:** The sampled explanations are consistent with the expected distance, time, and geographic signals. They do not establish causality, rule out spurious correlations or leakage, or determine why future performance might degrade. In particular, the pre-trip distance assumption in D25 remains open.

**Output:** 10 plot files saved to `reports/shap_*.png`. Automated pass/fail checklist printed at end of Notebook 4.

**Trade-offs:**
- SHAP is computed on the Jan 2023 test-set sample only — distributions may shift for future months
- `passenger_count` confirmed low-signal; could be dropped without meaningful accuracy loss (kept for now to preserve the model feature schema; the API is still planned)

---
