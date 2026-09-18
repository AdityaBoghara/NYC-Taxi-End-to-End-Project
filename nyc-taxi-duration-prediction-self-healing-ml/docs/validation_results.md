# Validation results and safeguards

Executed on 2026-09-18. This is a historical experiment and replay, not a deployed service.

Run: `20260918T223258Z-cd6d2cf5`. Full local report: `models/validation_runs/20260918T223258Z-cd6d2cf5/report.json`. Large artifacts and downloaded data are git-ignored; this summary is versioned.

## Scope

- Training: January 2023, 300,000 uniformly sampled rows before the final seven days. The last seven days provide 100,000 sampled early-stopping rows.
- Baselines use only the same training sample. Hierarchical medians require at least 30 observations per route/time, route, or pickup estimate; otherwise they fall back to the global median.
- Evaluation: 100,000 uniformly sampled rows in each later month, seed 42. No tuning on July or October.
- The new raw-data cohort requires valid pickup/dropoff times, pickup in the requested month, known geographic zones, and duration from 1 through 120 minutes.
- Actual distance, rate code, payment type, and passenger count do not decide inclusion. The airport flag now includes EWR; unknown zone placeholders are excluded.
- There is no claim of current NYC performance. The duration restriction also excludes some legitimate trips and limits generalization.

## Accuracy

MAE is average absolute error in minutes, not a per-trip error guarantee.

| Month | Role | XGBoost | Route median | Global median |
|---|---|---:|---:|---:|
| 2023-02 | diagnostic | 3.8224 | 4.6618 | 7.5738 |
| 2023-04 | promotion_gate | 4.4019 | 5.3130 | 8.5265 |
| 2023-07 | holdout | 4.3618 | 5.2079 | 8.4869 |
| 2023-10 | holdout | 5.1706 | 6.2550 | 9.4378 |

The candidate beats both baselines in every evaluated month, but October error is higher than winter error. The report also includes 90th-percentile absolute error and borough, airport, and rush-hour metrics. These runs use a broader cohort and smaller training sample than the earlier 3.96-minute notebook experiment, so they are not a controlled feature comparison.

## Promotion outcome: blocked

The proposed incumbent for this initial replay is the training-only route-median baseline; there is no deployed incumbent. The gate uses April only and requires:

- More than 2% overall MAE improvement.
- No required segment with more than 5% MAE regression.
- At least 1,000 paired labeled rows overall and 100 in each required pickup-borough, airport/nonairport, and rush/offpeak segment.

The April sample contains only **2 EWR pickups** and **7 Staten Island pickups**. Approval is blocked for insufficient evidence even though overall accuracy improves. Missing required segments also block approval. No model file or active deployment pointer was replaced.

The thresholds are initial engineering policy, not statistical significance guarantees. Before automatic promotion, strengthen evidence with larger/targeted segment evaluation and uncertainty estimates. Do not reduce the minimum simply to make this run pass.

## Outcome availability and temporal leakage

The replay assumes availability on the first day three months after pickup-month start (roughly two months after month end). This is an explicit simulation assumption; actual TLC release dates are not reconstructed.

| Event | Simulated date | Consequence |
|---|---|---|
| January training labels available | April 1, 2023 | Candidate becomes eligible to make simulated predictions |
| February labels available | May 1, 2023 | February is an offline diagnostic, not a prospective replay: the model did not exist in February |
| April outcomes available | July 1, 2023 | April can first support a promotion recommendation |
| July outcomes available | October 1, 2023 | Holdout reporting only |
| October outcomes available | January 1, 2024 | Holdout reporting only |

For April, July, and October, separate Parquet prediction and outcome ledgers contain stable IDs, prediction/model timestamps, actual durations, completion times, and simulated availability. The join returned **zero eligible outcomes before each release and 100,000 after release**. Predictions are frozen-candidate shadow predictions; the replay does not pretend that a blocked model was deployed.

The join rejects duplicate IDs, invalid timing, missing values, and models that would have required training labels unavailable at prediction time. IDs are based on a raw-file fingerprint and row number; they are replay IDs, not real ride IDs. A live service will need its own request/trip ID and outcome ingestion integration.

## Legacy score audit

The saved notebook recorded **2,883,614 rows**, versus **2,892,835** in the current processed file. This proves the cohort changed. Booster feature order matches the saved metadata, and the recorded raw-target transformation is respected.

The recorded test MAE of 2.5740 is not reproduced: reevaluation gives 3.5904. The row-count change does not prove the entire cause. Original dataset hashes and geographic encoding snapshots are missing, so the old score is quarantined from performance claims. The source model and its metadata are preserved.

Run `python -m src.audit_legacy` to regenerate `reports/legacy_audit.json` with artifact hashes, cutoffs, recomputed metrics, and evidence.

## Runbook and verification

```bash
python -m src.data_pipeline 2023-01 2023-02 2023-04 2023-07 2023-10
python -m src.validation
python -m src.audit_legacy
python -m pytest tests/ -q
```

**20 tests passed.** The full seasonal workflow also completed successfully. Tests cover filtering independence, chronological boundaries, training-only baseline fallback, segment regression and evidence requirements, drift response, label delay, duplicate IDs, and model readiness.

Each validation run has a unique directory with candidate/baseline bundles, training row IDs, ledgers, and the full report. The report captures configuration, package versions, raw-data hashes, code hashes, zone lookup hash, and model hashes. `reports/validation_latest.json` is atomically published only after successful completion and is an inspection pointer, not a production-model pointer.

## Remaining work

1. Increase evaluation evidence for EWR and Staten Island; if necessary evaluate multiple later labeled months while keeping a new future holdout untouched.
2. Validate additional months and recent data. This four-month evaluation reduces January-only uncertainty but does not eliminate it.
3. Review the 1–120 minute scope and rare/unseen-route behavior with product requirements; duration-based filtering cannot be applied before a trip.
4. Connect runs to MLflow, implement serving and live outcome ingestion, and add an operational drift monitor and retraining scheduler.
5. Add version promotion/rollback execution only after the evidence gate and deployment checks are met. Current code produces recommendations and never auto-deploys.
