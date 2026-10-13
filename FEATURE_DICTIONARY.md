# Feature Dictionary

Explains the two feature files, `enriched_features_train.parquet` and `enriched_features_ranking.parquet`, produced by `enrichment.py`.

Prepared by Harsh (data enrichment); updated for the final phase.

Each row is one departing flight, keyed by `MVT_ID_mvt` (float64, unique). Join it to the challenge data on that column.

## Anchor time

Every feature is computed relative to an anchor time known at off-block: `AOBT_3_flt`, the planned off-block time from the flight data. When it is missing (about 1–1.5% of flights) the scheduled time `SCHED_TIME_UTC_mvt` is used. `BLOCK_TIME_UTC_mvt` (actual off-block) is never used, because it is blank in the ranking file.

## Columns

| Column | Meaning | Source |
|---|---|---|
| `MVT_ID_mvt` | Flight ID, join key | Challenge data |
| `pushback_hour` | Hour of the anchor time, UTC (0–23) | Challenge data |
| `is_weekend` | 1 if Saturday or Sunday | Challenge data |
| `month` | Month of the anchor time (1–12); used only in the final model | Challenge data |
| `planes_pushed_last_15min` / `_30min` / `_60min` | Departures at the same airport in the window before this flight, strictly earlier (same minute and the flight itself not counted) | Challenge data |
| `arrivals_last_30min` | Landings at the airport in the 30 minutes before | Challenge data (ARR rows) |
| `dep_delay_min` | Planned off-block minus schedule, minutes, clipped to −120…600; 0 if AOBT is missing | Challenge data |
| `aobt_missing` | 1 if `AOBT_3_flt` was missing and the schedule was used | Challenge data |
| `visibility_miles` | Visibility, statute miles | IEM METAR |
| `wind_kt` | Wind speed, knots | IEM METAR |
| `temp_c` | Air temperature, °C | IEM METAR |
| `is_fog` | 1 if fog reported | IEM METAR |
| `is_snow_ice` | 1 if snow or freezing conditions reported (de-icing proxy) | IEM METAR |
| `stand_area` | First character of the stand code; groups under 1% of training flights → `OTHER` | Challenge data |
| `aircraft_size` | Wake category L/M/H/J, `UNK` if missing | Challenge data |
| `typical_taxi_sec` | Median taxi-out per airport + stand + runway (≥10 flights; falls back to airport + runway, then airport) | Challenge data |
| `unimpeded_taxi_sec` | 10th percentile of the same ("quiet day" taxi time) | Challenge data |
| `stand_runway_km` | Straight-line km from stand to departure runway threshold; NaN where the stand is not mapped (~6%) | OpenStreetMap + OurAirports |

## Things to know

1. Nothing uses `BLOCK_TIME_UTC_mvt` or the take-off time `MVT_TIME_UTC_mvt` of the flight itself. Take-off minus `AOBT_3_flt` reconstructs the target and is excluded as leakage.
2. Weather is the latest report at or before the anchor time, never a later one. A few reports missing a value take the previous report's value.
3. `typical_taxi_sec` and `unimpeded_taxi_sec` use leave-one-month-out on training rows (a January flight's value comes from the other 11 months) and all of 2025 on ranking rows. Only taxi times between 1 min and 2 h feed the statistics.
4. Read `stand_area` together with the airport: the same letter means a different place at different airports.
5. Rain and snow amounts are left out because the weather source does not report them reliably outside the US.

## Rebuild

Run `python enrichment.py` with the challenge files, `weather_cache.parquet`, `runways_10_airports.csv` and `osm_stands.csv` in the same folder. Needs pandas and pyarrow.
