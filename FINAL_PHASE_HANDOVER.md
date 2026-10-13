# Final-phase handover → teammate
PRC Data Challenge 2026 · from Oscar · 13 Oct 2026

Everything below is what changed since `HANDOVER.md` (the original enrichment handover) to
support the final phase, which scores on **four months**: Jan + Jul 2026 (original) plus
Feb + Jun 2026 (added). This doc is meant to give you what you need to update the repo and
write the paper without re-deriving any of it.

## 1. What's ready to submit

`submission_final_phase.parquet` — 670,790 rows, columns `MVT_ID_mvt` (float64),
`TAXITIME_SEC_mvt` (float64, the prediction, seconds). Same row set and row order as the
organizers' `final_submitting.parquet` template, every row filled, no NaNs, all positive.
This is the file to hand in.

It was built by: taking the already-trained production model (`taxi_out_xgb.joblib`, trained
on all 2025 months, see `HANDOVER.md` / the "adapted" notebook), running it on the new
4-month ranking set, then reshaping to the template's exact schema and row order.
`ranking_predictions.parquet` is the same predictions in the pipeline's own working format
(`MVT_ID_mvt` + `predicted_taxi_sec`) if you need them for analysis rather than submission.

## 2. What changed in the data

The organizers released `final_ranking.parquet` / `final_submitting.parquet` (670,790 DEP
rows total: Jan 152,724 / Feb 144,149 / Jun 181,789 / Jul 192,118, plus ~10 stray
boundary rows). These replaced the old 2-month `ranking.parquet` / had no `submitting.parquet`
equivalent before. In this repo they're now just `ranking.parquet` / `submitting.parquet` (so
`enrichment.py` and the notebook need no code changes); the old 2-month versions are kept as
`ranking_jan_jul2026.parquet` / `enriched_features_ranking_jan_jul2026.parquet` /
`ranking_predictions_jan_jul2026.parquet` for a before/after comparison in the paper if useful,
but are gitignored (superseded, no need to carry them in the repo).

`enrichment.py` was rerun unchanged against the new ranking file:
`enriched_features_ranking.parquet` now has 670,790 rows / 20 cols (was 344,841 / 20 for
Jan+Jul only). `enriched_features_train.parquet` is untouched (2,085,047 rows) — only the
ranking side changed.

**Worth noting for the paper's data section:** the organizers' final release also quietly
revised arrival timestamps for the *original* Jan/Jul flights, not just added Feb/Jun. Comparing
old vs. new raw files on the same flight IDs: 317,088 of 344,693 shared ARR records have a
slightly different `MVT_TIME_UTC_mvt` (jitter up to ±12s, mean ~0). Departure-side timestamps
(`AOBT_3_flt`, schedule) are untouched. That jitter is enough to flip the `arrivals_last_30min`
count by ±1 for about 23% of Jan/Jul flights (arrivals landing near the 30-minute window edge),
which moves individual predictions by up to ~370s even though nothing about the model or
pipeline changed. If you're comparing this phase's scores to any earlier Jan/Jul-only numbers,
this is part of why they won't match exactly.

Weather (`weather_cache.parquet`) and the static runway/stand files already covered the full
Jan–Jul 2026 span before any of this, so they needed no changes.

## 3. Generalization check: does the model actually work on Feb/Jun, or just Jan/Jul?

This was the main open question for the final phase (the organizers explicitly said they're
checking for generalization beyond the originally-scored months). Trained a diagnostic model —
same tuned hyperparameters as the production model
(`max_depth=9, min_child_weight=1.9, subsample=0.72, colsample_bytree=0.68, reg_lambda=1.7`),
but holding out **all four** final-phase months from training instead of just Jan+Jul, to get a
fair read on each pair:

| Test set | RMSE (s) | Baseline RMSE (airport+runway median) |
|---|---|---|
| Jan + Jul | 312.9 | 408.5 |
| Feb + Jun | 304.0 | 398.6 |
| All four combined | 308.6 | 403.7 |

Per month: Jan 308.3 / Feb 343.1 / Jun 269.4 / Jul 316.5 (n ≈ 143k–191k each).

**Takeaway: Feb+Jun generalizes at least as well as Jan+Jul (304.0 vs 312.9) — no sign the
model overfit to the originally-scored months.** Dropping two extra months from training barely
moved the Jan+Jul number versus the original 2-month-holdout run (310.7 → 312.9), which is
further evidence the gains come from real structure (`typical_taxi_sec`, weather, delay), not
memorizing those two calendar months. Feb is individually the weakest month, but see the weather
check below — it isn't a weather-anomaly effect for 2026, so more likely just a messier month in
general across years.

This diagnostic model is saved as `_diag_4month_holdout_model.joblib` (gitignored, same as the
production model) — it's evidence for the paper, not meant to generate predictions; the actual
submission still comes from the original production model trained on all of 2025.

## 4. Weather-anomaly check, extended to all four months

`HANDOVER.md` section 6.2 already flagged that Jan 2026 has ~3x the snow of Jan 2025 (1.6% →
4.6% of departures), a real risk for the Jan score. Checked the same thing for Feb/Jun/Jul:

| Month | 2025 snow share | 2026 snow share | 2025 fog share | 2026 fog share |
|---|---|---|---|---|
| Jan | 1.6% | 4.6% | 3.9% | 3.2% |
| Feb | 3.0% | 1.9% | 2.0% | 2.9% |
| Jun | 0.0% | 0.0% | 0.2% | 0.1% |
| Jul | 0.0% | 0.0% | 0.3% | 0.1% |

**Jan 2026 is the only month with a real adverse weather shift versus training.** Feb 2026 is
actually calmer than Feb 2025 on snow; Jun/Jul are flat both years. Worth a line in the paper:
expect the Jan score to be pulled down a bit by snow it hasn't seen as much of in training, but
no equivalent risk for the other three months.

## 5. Decided: take-off-time-minus-`AOBT_3_flt` is excluded as leakage

**Take-off time minus `AOBT_3_flt`** (`HANDOVER.md` §6.1) is **not used** in the submission.
This was open pending rules clarification; decided to leave it out rather than wait on that,
because tracing through the actual columns shows it's leakage, not just a risk of it:

- `TAXITIME_SEC_mvt` (the target) = `MVT_TIME_UTC_mvt` − `BLOCK_TIME_UTC_mvt`, exactly (checked:
  100% match in training data, to the second).
- `BLOCK_TIME_UTC_mvt` (the true off-block anchor) is deliberately blanked out in the ranking
  file — that's the entire reason the task isn't trivial.
- `MVT_TIME_UTC_mvt` (actual take-off time) is **not** blanked out (0% missing in ranking),
  even though it's the event marking the *end* of the taxi-out period being predicted.
- `AOBT_3_flt` is a flight-plan-side estimate of off-block time, used elsewhere as the anchor
  for every legitimate feature. It's correlated with the true `BLOCK_TIME_UTC_mvt` but not
  identical to it.
- So "take-off minus `AOBT_3`" = `MVT_TIME_UTC_mvt` − `AOBT_3_flt` is a noisy reconstruction of
  the answer: r = 0.64 against the true target in training data (not 1.0, because `AOBT_3_flt`
  isn't exactly `BLOCK_TIME_UTC_mvt`), which is why it drops Harsh's RMSE by ~12% (315→276) —
  far more than any legitimate feature contributes individually.

At prediction time (push-back), a real deployment of this model would not know the flight's own
take-off time — if it did, it could just subtract and get the real taxi time with no model
needed. Using it would inflate the measured score without reflecting anything the model could
actually do operationally, which cuts directly against the final phase's stated goal of
rewarding models that generalize beyond the competition's specifics. Worth a line in the paper:
this feature was identified, evaluated, and deliberately excluded as leakage, not overlooked.

## 6. Repo hygiene done this session

- `requirements.txt` updated to the versions actually used to run everything
  (`pandas==3.0.3`, `numpy==2.4.5`, `pyarrow==24.0.0`, plus `xgboost==3.2.0`,
  `scikit-learn==1.8.0`, `joblib==1.5.2` for the modeling side, which wasn't listed before).
  `HANDOVER.md` already confirmed `enrichment.py`'s output is identical on old and new pandas,
  so this is a safe bump, not a behavior change.
- `.gitignore` updated: `submitting.parquet` is now excluded alongside `ranking.parquet` (same
  redistribution restriction), and the three superseded Jan/Jul-only files are excluded too.

## 7. File manifest (final-phase-relevant files only)

| File | What it is |
|---|---|
| `submission_final_phase.parquet` | **The file to submit.** |
| `ranking_predictions.parquet` | Same predictions, working format |
| `ranking.parquet` / `submitting.parquet` | Final-phase raw files from the organizers (4 months) |
| `enriched_features_ranking.parquet` | Enriched features for the 4-month ranking set |
| `enriched_features_train.parquet` | Unchanged from the original handover |
| `taxi_out_xgb.joblib` / `.ubj` | Production model (trained on all 2025, used for the submission) |
| `_diag_4month_holdout_model.joblib` | Diagnostic only — generalization check in §3, not used for predictions |
| `ranking_jan_jul2026.parquet` and 2 related files | Superseded 2-month-only versions, kept for before/after comparison, gitignored |
