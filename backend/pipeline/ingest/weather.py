"""P1-4 — weather and track conditions, joined onto each lap.

Weather arrives as its own time series (about one row per minute), so each
lap takes the most recent reading at or before its start. `merge_asof` with
`direction="backward"` is the right join here: a reading taken after a lap
began did not describe the conditions that lap was driven in.

Track temperature is the single most load-bearing environment variable in the
simulator — it drives the tyre thermal model — so a lap that cannot be given
one is worth knowing about rather than silently defaulting.
"""
from __future__ import annotations

import pandas as pd

WEATHER_COLUMNS = ["AirTemp", "TrackTemp", "Humidity", "Pressure",
                   "Rainfall", "WindSpeed", "WindDirection"]

RENAME = {
    "AirTemp": "air_temp_c",
    "TrackTemp": "track_temp_c",
    "Humidity": "humidity_pct",
    "Pressure": "pressure_mbar",
    "Rainfall": "rainfall",
    "WindSpeed": "wind_speed_kph",
    "WindDirection": "wind_direction_deg",
}


def merge_onto_laps(laps: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = laps.copy()
    diag: dict = {"weather_rows": int(len(weather))}

    if weather is None or not len(weather) or "Time" not in weather or "LapStartTime" not in out:
        for c in RENAME.values():
            out[c] = pd.NA
        diag["joined"] = False
        diag["reason"] = "no weather series or no LapStartTime"
        return out, diag

    w = weather[["Time"] + [c for c in WEATHER_COLUMNS if c in weather]].copy()
    w["_t"] = pd.to_timedelta(w["Time"], errors="coerce")
    w = w.dropna(subset=["_t"]).sort_values("_t")

    left = out.copy()
    left["_t"] = pd.to_timedelta(left["LapStartTime"], errors="coerce")
    order = left["_t"].argsort(kind="stable")
    left = left.iloc[order]

    merged = pd.merge_asof(
        left.dropna(subset=["_t"]),
        w.drop(columns=["Time"]),
        on="_t", direction="backward",
    )
    merged = merged.rename(columns=RENAME).drop(columns=["_t"])

    diag["joined"] = True
    if "track_temp_c" in merged:
        tt = pd.to_numeric(merged["track_temp_c"], errors="coerce")
        diag["track_temp_c_range"] = [round(float(tt.min()), 1), round(float(tt.max()), 1)] \
            if tt.notna().any() else None
        diag["laps_without_track_temp"] = int(tt.isna().sum())
    if "rainfall" in merged:
        diag["laps_with_rainfall"] = int(merged["rainfall"].fillna(False).astype(bool).sum())

    return merged.reset_index(drop=True), diag
