"""
Downloads stand / gate positions for our 10 airports from OpenStreetMap (Overpass API).
Output: osm_stands.csv  ->  airport, osm_type, aeroway, ref, name, lat, lon
Run once:  python fetch_osm_stands.py
Data (c) OpenStreetMap contributors, ODbL.
"""
import time, requests, pandas as pd

AIRPORTS = {  # ICAO: (lat, lon) airport centre
    "EDDF": (50.033, 8.570), "EDDM": (48.354, 11.786), "EGLL": (51.471, -0.461),
    "EHAM": (52.309, 4.764), "LEBL": (41.297, 2.078), "LEMD": (40.494, -3.572),
    "LFPG": (49.009, 2.554), "LIRF": (41.805, 12.252), "LSZH": (47.458, 8.548),
    "LTFM": (41.275, 28.732),
}
RADIUS_M = 4500  # smaller area = less likely to time out
URLS = ["https://overpass-api.de/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
HEADERS = {"User-Agent": "prc-data-challenge-2026-student-project/1.0 (research; contact via GitHub)",
           "Accept": "application/json"}

import os
old = pd.read_csv("osm_stands.csv") if os.path.exists("osm_stands.csv") else pd.DataFrame()
have = set(old["airport"]) if len(old) else set()
todo = {k: v for k, v in AIRPORTS.items() if k not in have}
print("already have:", sorted(have)); print("fetching:", sorted(todo))
rows = []
for i, (icao, (lat, lon)) in enumerate(todo.items(), 1):
    q = f"""[out:json][timeout:120];
(
  node["aeroway"~"^(parking_position|gate)$"](around:{RADIUS_M},{lat},{lon});
  way["aeroway"="parking_position"](around:{RADIUS_M},{lat},{lon});
);
out center tags;"""
    print(f"[{i}/{len(todo)}] {icao} ...", end=" ", flush=True)
    data = None
    for attempt in range(6):
        url = URLS[attempt % len(URLS)]
        try:
            r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=240)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code} from {url.split('/')[2]}: {r.text[:120]!r}")
            data = r.json(); break
        except Exception as e:
            print(f"\n   retry: {e}", flush=True); time.sleep(15)
    if data is None:
        print("FAILED"); continue
    n = 0
    for el in data.get("elements", []):
        t = el.get("tags", {})
        la = el.get("lat") or el.get("center", {}).get("lat")
        lo = el.get("lon") or el.get("center", {}).get("lon")
        if la is None: continue
        rows.append({"airport": icao, "osm_type": el["type"], "aeroway": t.get("aeroway"),
                     "ref": t.get("ref"), "name": t.get("name"), "lat": la, "lon": lo})
        n += 1
    print(f"ok ({n} stands/gates)")
    time.sleep(5)  # be polite to the free server

df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
df.to_csv("osm_stands.csv", index=False)
print(f"\nSaved {len(df)} rows -> osm_stands.csv")
print(df.groupby(["airport", "aeroway"]).size().unstack(fill_value=0))
print("rows with no ref/name:", df[["ref", "name"]].isna().all(axis=1).sum())
