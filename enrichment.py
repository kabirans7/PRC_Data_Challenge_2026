"""
PRC Data Challenge 2026 - Data Enrichment (final version)
Author: Harsh (External Data & Feature Enrichment Lead)

Builds leakage-safe features for every departing flight, for both the
training months (2025) and the ranking set (Jan + Jul 2026).

Anchor time = AOBT_3_flt (flight-plan off-block), falling back to
SCHED_TIME_UTC_mvt when missing. BLOCK_TIME_UTC_mvt is NOT used because it is
blank in the ranking file.

Run:
    python enrichment.py
Inputs (same folder):
    training_2025-*.parquet, ranking.parquet, weather_cache.parquet,
    runways_10_airports.csv, osm_stands.csv (optional, for distance)
Outputs:
    enriched_features_train.parquet, enriched_features_ranking.parquet
"""

import glob
import os

import numpy as np
import pandas as pd

WEATHER_CACHE = "weather_cache.parquet"
RUNWAYS = "runways_10_airports.csv"
OSM_STANDS = "osm_stands.csv"

WEATHER_COLS = ["visibility_miles", "wind_kt", "temp_c", "is_fog", "is_snow_ice"]
RAW_COLS = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "STAND_mvt", "RUNWAY_mvt",
            "AOBT_3_flt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "WK_TBL_CAT_flt",
            "TAXITIME_SEC_mvt"]
MIN_FLIGHTS = 10  # minimum flights for a stand+runway typical-time value


# ---------------------------------------------------------------- loading
def load_raw(files):
    parts = []
    for f in files:
        df = pd.read_parquet(f)
        parts.append(df[[c for c in RAW_COLS if c in df.columns]])
    d = pd.concat(parts, ignore_index=True)
    for c in ["AOBT_3_flt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"]:
        d[c] = pd.to_datetime(d[c], utc=True, errors="coerce")
    return d


def load_weather(path=WEATHER_CACHE):
    wx = pd.read_parquet(path)
    wx["valid"] = pd.to_datetime(wx["valid"], errors="coerce")
    wx["valid"] = (wx["valid"].dt.tz_localize("UTC") if wx["valid"].dt.tz is None
                   else wx["valid"].dt.tz_convert("UTC"))
    wx = wx.dropna(subset=["valid", "airport"])
    # deterministic: one report per airport+time (keep the last)
    wx = wx.sort_values(["airport", "valid"]).drop_duplicates(["airport", "valid"], keep="last")
    # a few reports lack a value: carry the previous report forward (past only)
    wx[WEATHER_COLS] = wx.groupby("airport")[WEATHER_COLS].ffill()
    return wx.sort_values("valid").reset_index(drop=True)


# ---------------------------------------------------------------- helpers
def count_before(event_times, query_times, minutes):
    """For each query time t, count events with t - minutes <= event < t.
    event_times must be sorted. Ties at exactly t are NOT counted (past only)."""
    ev = np.asarray(event_times).astype("datetime64[ns]").astype("int64")
    q = np.asarray(query_times).astype("datetime64[ns]").astype("int64")
    w = np.int64(minutes * 60 * 1_000_000_000)
    return np.searchsorted(ev, q, "left") - np.searchsorted(ev, q - w, "left")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def norm_stand(s):
    """Normalise stand codes so 'A 12', 'a12', 'A012' all become 'A12'."""
    s = s.astype("string").str.upper().str.replace(r"[\s\-_/]", "", regex=True)
    return s.str.replace(r"(?<=[A-Z])0+(?=\d)", "", regex=True).str.replace(r"^0+(?=\d)", "", regex=True)


# ---------------------------------------------------------------- features
def build_features(raw, weather, stand_area_keep=None):
    """Features for every DEP row of `raw` (one row per MVT_ID_mvt)."""
    dep = raw[raw["PHASE_mvt"] == "DEP"].copy()
    arr = raw[raw["PHASE_mvt"] == "ARR"]

    dep["aobt_missing"] = dep["AOBT_3_flt"].isna().astype(int)
    dep["anchor_time"] = dep["AOBT_3_flt"].fillna(dep["SCHED_TIME_UTC_mvt"])

    # time
    dep["pushback_hour"] = dep["anchor_time"].dt.hour
    dep["is_weekend"] = (dep["anchor_time"].dt.dayofweek >= 5).astype(int)
    dep["month"] = dep["anchor_time"].dt.month

    # departure delay (minutes late leaving the stand vs schedule)
    delay = (dep["AOBT_3_flt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds() / 60
    dep["dep_delay_min"] = delay.clip(-120, 600).fillna(0).round(1)

    # traffic: departures pushed back / arrivals landed at the same airport before t
    for c in ["planes_pushed_last_15min", "planes_pushed_last_30min",
              "planes_pushed_last_60min", "arrivals_last_30min"]:
        dep[c] = 0
    for ap, idx in dep.groupby("ADEP_mvt").groups.items():
        t = dep.loc[idx, "anchor_time"]
        ev = np.sort(t.dropna().values)
        ok = t.notna().values
        for m in (15, 30, 60):
            col = f"planes_pushed_last_{m}min"
            vals = np.zeros(len(t), dtype=int)
            vals[ok] = count_before(ev, t.values[ok], m)
            dep.loc[idx, col] = vals
        landed = np.sort(arr.loc[arr["ADES_mvt"] == ap, "MVT_TIME_UTC_mvt"].dropna().values)
        vals = np.zeros(len(t), dtype=int)
        vals[ok] = count_before(landed, t.values[ok], 30)
        dep.loc[idx, "arrivals_last_30min"] = vals

    # weather: latest report at or before anchor time
    dep = dep.sort_values("anchor_time")
    has_t = dep["anchor_time"].notna()
    merged = pd.merge_asof(
        dep.loc[has_t, ["MVT_ID_mvt", "ADEP_mvt", "anchor_time"]],
        weather[["airport", "valid"] + WEATHER_COLS],
        left_on="anchor_time", right_on="valid",
        left_by="ADEP_mvt", right_by="airport", direction="backward")
    dep = dep.merge(merged[["MVT_ID_mvt"] + WEATHER_COLS], on="MVT_ID_mvt", how="left")

    # categories
    dep["stand_area"] = dep["STAND_mvt"].astype("string").str.strip().str[0].fillna("UNK")
    if stand_area_keep is None:  # decided on training data only, reused for ranking
        share = dep["stand_area"].value_counts(normalize=True)
        stand_area_keep = set(share[share >= 0.01].index)
    dep.loc[~dep["stand_area"].isin(stand_area_keep), "stand_area"] = "OTHER"
    dep["aircraft_size"] = (dep["WK_TBL_CAT_flt"].astype("string").str.strip().str.upper()
                            .replace("", pd.NA).fillna("UNK"))
    return dep, stand_area_keep


# -------------------------------------------- typical stand+runway taxi time
def add_typical_taxi(train_dep, rank_dep):
    """Median and 10th-percentile taxi time per (airport, stand, runway).
    Training rows: leave-one-month-out (a flight never sees its own month).
    Ranking rows: from all 2025 months. Falls back to (airport, runway), then airport."""
    # only plausible taxi times feed the statistics (1 min to 2 h)
    ok = train_dep[train_dep["TAXITIME_SEC_mvt"].between(60, 7200)]
    levels = [["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], ["ADEP_mvt", "RUNWAY_mvt"], ["ADEP_mvt"]]

    def stats(src):
        out = []
        for keys in levels:
            g = src.groupby(keys)["TAXITIME_SEC_mvt"].agg(
                n="size", med="median", p10=lambda x: x.quantile(0.10))
            out.append((keys, g[g["n"] >= MIN_FLIGHTS]))
        return out

    def apply(target, st):
        med = pd.Series(np.nan, index=target.index)
        p10 = pd.Series(np.nan, index=target.index)
        for keys, g in st:
            m = target[keys].merge(g[["med", "p10"]], left_on=keys, right_index=True, how="left")
            fill = med.isna() & m["med"].notna()
            med[fill] = m.loc[fill, "med"]
            p10[fill] = m.loc[fill, "p10"]
        return med.round(0), p10.round(0)

    train_dep = train_dep.copy()
    train_dep["typical_taxi_sec"] = np.nan
    train_dep["unimpeded_taxi_sec"] = np.nan
    for mon in sorted(train_dep["month"].dropna().unique()):
        rows = train_dep["month"] == mon
        med, p10 = apply(train_dep[rows], stats(ok[ok["month"] != mon]))
        train_dep.loc[rows, "typical_taxi_sec"] = med
        train_dep.loc[rows, "unimpeded_taxi_sec"] = p10

    rank_dep = rank_dep.copy()
    rank_dep["typical_taxi_sec"], rank_dep["unimpeded_taxi_sec"] = apply(rank_dep, stats(ok))
    return train_dep, rank_dep


# ------------------------------------------------ stand -> runway distance
def add_distance(dep, runways_path=RUNWAYS, stands_path=OSM_STANDS):
    """Straight-line km from the stand (OpenStreetMap) to the start of the
    departure runway (OurAirports threshold). NaN where the stand is not on the map."""
    dep = dep.copy()
    dep["stand_runway_km"] = np.nan
    if not os.path.exists(stands_path):
        print(f"  {stands_path} not found - distance feature skipped")
        return dep

    rw = pd.read_csv(runways_path)
    ends = pd.concat([
        rw[["airport_ident", "le_ident", "le_latitude_deg", "le_longitude_deg"]].set_axis(
            ["ADEP_mvt", "RUNWAY_mvt", "rw_lat", "rw_lon"], axis=1),
        rw[["airport_ident", "he_ident", "he_latitude_deg", "he_longitude_deg"]].set_axis(
            ["ADEP_mvt", "RUNWAY_mvt", "rw_lat", "rw_lon"], axis=1)])
    ends["RUNWAY_mvt"] = ends["RUNWAY_mvt"].astype(str)

    st = pd.read_csv(stands_path)
    st["key"] = norm_stand(st["ref"].fillna(st["name"]))
    st = st.dropna(subset=["key"])
    # where a code exists as both a parking position and a gate, use the parking position
    st["prio"] = (st["aeroway"] != "parking_position").astype(int)
    st = st[st["prio"] == st.groupby(["airport", "key"])["prio"].transform("min")]
    st = st.groupby(["airport", "key"], as_index=False)[["lat", "lon"]].mean()
    st = st.rename(columns={"airport": "ADEP_mvt", "lat": "st_lat", "lon": "st_lon"})

    tmp = dep[["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]].copy()
    tmp["key"] = norm_stand(tmp["STAND_mvt"])
    tmp["RUNWAY_mvt"] = tmp["RUNWAY_mvt"].astype(str)
    tmp = tmp.merge(st, on=["ADEP_mvt", "key"], how="left").merge(
        ends, on=["ADEP_mvt", "RUNWAY_mvt"], how="left")
    dep["stand_runway_km"] = haversine_km(tmp["st_lat"].values, tmp["st_lon"].values,
                                          tmp["rw_lat"].values, tmp["rw_lon"].values).round(3)
    # a stand more than 8 km from its runway is a wrong match, not a real distance
    dep.loc[dep["stand_runway_km"] > 8, "stand_runway_km"] = np.nan
    return dep


FINAL_COLS = [
    "MVT_ID_mvt",
    "pushback_hour", "is_weekend", "month",
    "planes_pushed_last_15min", "planes_pushed_last_30min", "planes_pushed_last_60min",
    "arrivals_last_30min",
    "dep_delay_min", "aobt_missing",
    "visibility_miles", "wind_kt", "temp_c", "is_fog", "is_snow_ice",
    "stand_area", "aircraft_size",
    "typical_taxi_sec", "unimpeded_taxi_sec", "stand_runway_km",
]


def main():
    weather = load_weather()
    train_raw = load_raw(sorted(glob.glob("training_2025-*.parquet")))
    rank_raw = load_raw(["ranking.parquet"])

    train_dep, keep = build_features(train_raw, weather)
    rank_dep, _ = build_features(rank_raw, weather, stand_area_keep=keep)
    train_dep, rank_dep = add_typical_taxi(train_dep, rank_dep)
    train_dep = add_distance(train_dep)
    rank_dep = add_distance(rank_dep)

    cols = FINAL_COLS
    if train_dep["stand_runway_km"].isna().all():
        cols = [c for c in FINAL_COLS if c != "stand_runway_km"]
    for name, d in [("train", train_dep), ("ranking", rank_dep)]:
        out = d[cols].sort_values("MVT_ID_mvt").reset_index(drop=True)
        out.to_parquet(f"enriched_features_{name}.parquet", index=False)
        print(f"saved enriched_features_{name}.parquet {out.shape}")


if __name__ == "__main__":
    main()
