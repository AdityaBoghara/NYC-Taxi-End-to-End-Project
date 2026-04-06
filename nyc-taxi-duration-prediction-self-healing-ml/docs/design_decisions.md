# Design Decisions

This document records the key design choices made throughout the NYC Taxi Duration Prediction project, along with the rationale and known trade-offs for each.

Decisions are numbered sequentially. Each entry records: **what was decided**, **why**, and **known trade-offs or open follow-ups**.

---

## 1. Data Source

**Decision:** Use the NYC TLC Yellow Taxi trip records (Parquet format) from the official TLC S3-compatible endpoint.

**Rationale:**
- Authoritative source maintained by the NYC Taxi & Limousine Commission
- Parquet format is efficient for columnar reads and reduces I/O compared to CSV
- Data is publicly available and versioned by year/month

**Trade-offs:**
- Data lags by ~2 months (no real-time feed)
- Schema has changed across years; scripts must handle version differences

---

## 2. Geographic Scope ✅ Resolved

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

## 3. Target Variable

**Decision:** Predict `trip_duration_min` — the elapsed time from pickup to dropoff in **minutes**.

**Rationale:**
- Minutes are human-interpretable (vs raw seconds)
- Regression on minutes aligns with standard taxi ETA benchmarks
- Derived as: `(dropoff_datetime - pickup_datetime).total_seconds() / 60`

**Trade-offs:**
- Does not account for waiting time at pickup (meter starts on dispatch in some cases)
- Ignores traffic conditions not captured in the structured features

---

## 4. Outlier Filtering — Duration

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

## 5. Outlier Filtering — Negative Durations

**Decision:** Remove trips where `trip_duration < 0` (dropoff recorded before pickup).

**Rationale:**
- 3 such records were found in Jan 2023 data; all had `RatecodeID = 99` (unknown/disputed)
- These are data entry errors and cannot represent real trips

**Trade-offs:**
- Negligible impact (3 out of ~3M rows)

---

## 6. Date Range Filtering

**Decision:** Keep only rides where `pickup_datetime` falls strictly within the downloaded month (Jan 2023).

**Rationale:**
- The raw parquet occasionally contains records from adjacent months due to late submission by vendors
- Mixing months would introduce temporal leakage when the model is evaluated month-by-month

---

## 7. Flex Fare Block — Rows with payment_type = 0

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

## 8. Trip Distance Filtering

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

## 9. Passenger Count Filtering

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

## 10. Feature Engineering

### 10a. Time-based features

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

### 10b. Location features

**Decision:** Keep `PULocationID` and `DOLocationID` as raw integer zone IDs **and** add `PU_borough_id` / `DO_borough_id` from the TLC zone lookup.

**Rationale:**
- Zone IDs encode fine-grained spatial information; tree-based models handle high-cardinality categorical integers well
- Borough IDs add coarser but more interpretable geographic context
- Providing both lets the model learn at two spatial resolutions

**Known gap:** High cardinality (~260 zones). Future options:
- Target-encode zones using mean trip duration (requires careful cross-validation to avoid leakage)
- Embed zones using a learned representation (e.g., entity embeddings in a neural net)

### 10c. Airport trip flag

**Decision:** Add `is_airport_trip` binary feature (1 if PU or DO service_zone = 'Airports').

**Rationale (from EDA):**
- Airport trips have median duration 2–3× higher than standard city trips
- Their duration is dominated by highway congestion, not city street density — a structurally different regime
- The flag is a known-at-dispatch value (driver and dispatcher know the destination)
- `RatecodeID = 2` (JFK flat-rate) already partially captures this, but LaGuardia trips use standard metered rate

**Trade-offs:**
- Requires the zone lookup join to be successful at runtime

### 10d. Excluded columns (leakage prevention)

The following columns are **excluded** from the model feature set because they are only known *after* the trip ends (data leakage) or are irrelevant to duration:

**Post-trip financial values:**
- `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `total_amount`, `congestion_surcharge`, `airport_fee`

**Post-trip or non-causal:**
- `payment_type` — selected by passenger at trip end
- `store_and_fwd_flag` — transmission mode indicator, not a trip attribute
- `VendorID` — no meaningful duration difference between vendors (confirmed in EDA). Vendor 2 is listed as "Curb Mobility, LLC" in the 2025 data dictionary (formerly VeriFone — same fleet, rebranded)

**Redundant datetime columns:**
- `pickup_datetime`, `dropoff_datetime` — replaced by derived features
- `trip_duration` (seconds) — replaced by `trip_duration_min` (target)

---

## 11. RatecodeID — Retention and Treatment

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

## 12. Data Storage Format

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

---

## 13. Open Decisions (To Be Resolved)

| # | Decision | Options | Blocking? | Status |
|---|---|---|---|---|
| 1 | Geographic scope | Manhattan only vs all boroughs vs borough-as-feature | Before model training | **Resolved** — all boroughs retained, borough encoded |
| 2 | Zone ID encoding | Raw integer vs target encoding vs borough lookup | Before model training | **Partially resolved** — raw integer + borough for baseline |
| 3 | Holiday flag | Add or skip | Low priority | Open |
| 4 | Duration outlier threshold | Fixed 120 min vs percentile-based | Low priority | Open |
| 5 | Multi-month training | Jan 2023 only vs full year | Before evaluation | Open |
| 6 | Distance outlier threshold | Fixed 50 mi vs percentile-based | Low priority | Open |
| 7 | Zone target encoding | Raw integer vs target-encoded mean duration | Before advanced modelling | Open |
