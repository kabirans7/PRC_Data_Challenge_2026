# Enrichment handover → Oscar
PRC Data Challenge 2026 · from Harsh (Enrichment) · 30 Sep 2026

## 1. What's in this folder

| File | What it is |
|---|---|
| `enriched_features_train.parquet` | 2,085,047 rows: one per 2025 training departure |
| `enriched_features_ranking.parquet` | 344,841 rows: one per Jan + Jul 2026 test departure |
| `enrichment.py` | The script that makes both files. Run `python enrichment.py` next to the challenge files |
| `weather_cache.parquet` | IEM airport weather, 10 airports, 1 Jan 2025 → 31 Jul 2026 (input to the script) |
| `runways_10_airports.csv` | OurAirports runway ends (input for the distance feature) |
| `osm_stands.csv` | OpenStreetMap stand/gate positions, 10 airports, downloaded 30 Sep 2026 (input for the distance feature) |
| `fetch_osm_stands.py` | Re-downloads `osm_stands.csv` (the free servers are often busy; it retries and skips airports already saved) |
| `requirements.txt` | Package versions |

Join key: `MVT_ID_mvt` (float64, same as `submitting.parquet`).

## 2. Features (19 + ID)

Every feature uses only information available at off-block. Anchor time = `AOBT_3_flt`, or `SCHED_TIME_UTC_mvt` when it is missing (~1–1.5%). `BLOCK_TIME_UTC_mvt` is never used (it is blank in the ranking file).

| Feature | Meaning | Source |
|---|---|---|
| `pushback_hour` | Hour of anchor time (UTC) | Challenge data |
| `is_weekend` | 1 = Sat/Sun | Challenge data |
| `month` | 1–12 | Challenge data |
| `planes_pushed_last_15min` / `30min` / `60min` | Departures at the same airport in the window before this flight (same-minute flights not counted) | Challenge data |
| `arrivals_last_30min` **(new)** | Planes that landed at the airport in the 30 min before | Challenge data (ARR rows) |
| `dep_delay_min` **(new)** | Minutes off-block is late vs schedule (clipped −120…600; 0 if AOBT missing) | Challenge data |
| `aobt_missing` **(new)** | 1 = AOBT_3 was missing, anchor fell back to schedule | Challenge data |
| `visibility_miles`, `wind_kt`, `temp_c`, `is_fog`, `is_snow_ice` | Latest weather report at or before anchor time | IEM |
| `stand_area` | First character of stand; groups <1% of training flights → `OTHER` | Challenge data |
| `aircraft_size` | Wake category L/M/H/J, `UNK` if missing | Challenge data |
| `typical_taxi_sec` **(new)** | Median taxi time for this airport + stand + runway (≥10 flights; falls back to airport + runway, then airport) | Challenge data |
| `unimpeded_taxi_sec` **(new)** | 10th percentile of the same: the "quiet day" taxi time, EUROCONTROL's method | Challenge data |
| `stand_runway_km` **(new)** | Straight-line km from the stand to the start of the departure runway. Empty (NaN) where the stand isn't on the map: ~6% of flights, mostly Munich (57% found) | OpenStreetMap + OurAirports |

**How the two typical-time features avoid leakage:** training rows use *leave-one-month-out* (a January flight's value comes only from the other 11 months). Test rows use all of 2025. Only taxi times between 1 min and 2 h feed the statistics.

Also give the model the raw columns: `ADEP_mvt`, `RUNWAY_mvt`, `AIRCRAFT_TYPE_mvt` (and `AIRCRAFT_OPERATOR_flt` if you like).

## 3. Checks done

- Train IDs = exactly the DEP rows of the 12 training files. Ranking IDs = exactly `submitting.parquet`. 0 duplicates in both.
- Missing values: ranking 0 (except `stand_runway_km`). Train: 5 flights have no weather (off-block before the first report on 1 Jan 2025 00:00).
- `stand_runway_km` found for 94% of flights (96–99% at 8 airports, LEBL 90%, EDDM 57%). Median distances 1–3 km. Within each airport, longer distance = longer taxi (correlation 0.16–0.56).
- Categories in ranking all exist in train.
- Traffic counts spot-checked by hand on 300 random flights: 0 mismatches.
- Same output with pandas 1.5.3 and pandas 3.0 (reproducible).

## 4. Quick model test (LightGBM, train on other months, test on Jan + Jul 2025)

| Features | RMSE (s) |
|---|---|
| Predict the average | 452 |
| Previous features + raw airport/runway/aircraft type | 339 |
| + new features | 315.9 |
| **+ distance (this handover, all features)** | **315.4** |
| + take-off minus AOBT_3 (see below) | 276 |

The 315 holds in a stricter test where the typical-time values never see January or July at all. Most useful features by gain: `typical_taxi_sec`, then `dep_delay_min`. The distance adds little on top (`typical_taxi_sec` already knows each stand+runway pair), but it helps stands with too few flights for a typical time. Keep it or drop it based on your own tests.

## 5. Changes vs the files I had before

1. **Traffic counts fixed.** The old counts included flights pushing back in the *same minute*, which made about 21% of values 1 higher. The result also depended on the pandas version. The new version counts strictly before t, and gives the same result on every version.
2. **Traffic counts across month boundaries.** Built on the whole year at once, so the first hour of each month is no longer undercounted.
3. **`stand_area` groups are decided once, on training data**, and reused for ranking. Before, they were decided month by month.
4. **Weather:** duplicate reports removed. The few reports missing a value take the previous report's value (past only).

## 6. Things to decide

1. **Take-off time is in the ranking file** (`MVT_TIME_UTC_mvt`, 0% missing). Take-off minus AOBT_3 cuts RMSE from 315 to 276 in the test above. It is in the official test file, but confirm with Luis/the rules before using it.
2. **January 2026 had about 3× more snow than January 2025** (4.6% vs 1.6% of departures). Check errors on snowy flights separately.

## 7. Sources to cite

- PRC Data Challenge 2026 data: https://prc-data-challenge-2026.netlify.app/data.html
- Iowa Environmental Mesonet (IEM) ASOS/METAR: https://mesonet.agron.iastate.edu/request/download.phtml
- OurAirports: https://ourairports.com/data/
- OpenStreetMap (© OpenStreetMap contributors, ODbL) via Overpass API: https://www.openstreetmap.org/copyright
