"""P1-6 — bronze layer writer.

Layout:

    data/bronze/{season}/{event_slug}/{session}/
        laps.parquet          one row per surviving lap
        telemetry.parquet     one row per raw sample of those laps
        session.json          identity + the full filter report
    data/bronze/manifest.json aggregated index of every ingested session

Two rules the schema exists to enforce:

* Speed traps are written VERBATIM with an explicit `*_missing` flag beside
  each. Imputation needs segment boundaries, which do not exist until P2, so
  filling them here would mean inventing a value from the wrong scope. The
  flag travels with the row so P2-7 can fill it and the model can see that it
  was filled (docs/recon/DECISIONS.md D6).
* DRS is stored as the raw code only, under `drs_raw`. The simulator's
  `drs_open` is derived from the circuit's activation zones, because DRS
  availability is a race-situation variable rather than anything a setup can
  change (D5b). Writing `drs_open` here would bake a race situation into a
  setup dataset.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LAP_COLUMNS: dict[str, str] = {
    # identity
    "lap_uid": "lap_uid",
    "season": "season",
    "event": "event",
    "event_slug": "event_slug",
    "session": "session",
    "circuit": "circuit",
    "Driver": "driver",
    "DriverNumber": "driver_number",
    "Team": "team_raw",
    "team_canonical": "team",
    "chassis": "chassis",
    "power_unit": "power_unit",
    # lap
    "LapNumber": "lap_number",
    "Stint": "stint",
    "lap_time_s": "lap_time_s",
    "sector1_s": "sector1_s",
    "sector2_s": "sector2_s",
    "sector3_s": "sector3_s",
    "lap_start_s": "lap_start_s",
    "Position": "position",
    "IsPersonalBest": "is_personal_best",
    # speed traps (setup proxies)
    "SpeedI1": "speed_i1_kph",
    "SpeedI2": "speed_i2_kph",
    "SpeedFL": "speed_fl_kph",
    "SpeedST": "speed_st_kph",
    "speed_i1_missing": "speed_i1_missing",
    "speed_i2_missing": "speed_i2_missing",
    "speed_fl_missing": "speed_fl_missing",
    "speed_st_missing": "speed_st_missing",
    # tyre & conditions
    "Compound": "compound",
    "TyreLife": "tyre_life",
    "FreshTyre": "fresh_tyre",
    "condition": "condition",
    "TrackStatus": "track_status",
    "track_status_flag": "track_status_flag",
    "telemetry_quality": "telemetry_quality",
    # pace gate provenance
    "pace_reference_s": "pace_reference_s",
    "pace_ratio": "pace_ratio",
    "pace_ratio_session_best": "pace_ratio_session_best",
    # telemetry quality
    "n_samples": "n_samples",
    "lap_distance_m": "lap_distance_m",
    "max_gap_m": "telemetry_max_gap_m",
    "median_gap_m": "telemetry_median_gap_m",
    "p95_gap_m": "telemetry_p95_gap_m",
    "max_gap_s": "telemetry_max_gap_s",
    "median_gap_s": "telemetry_median_gap_s",
    "missed_samples_worst": "telemetry_missed_samples",
    "worst_gap_at_frac": "telemetry_worst_gap_at_frac",
    "implied_speed_kph": "telemetry_gap_implied_speed_kph",
    # weather
    "air_temp_c": "air_temp_c",
    "track_temp_c": "track_temp_c",
    "humidity_pct": "humidity_pct",
    "pressure_mbar": "pressure_mbar",
    "rainfall": "rainfall",
    "wind_speed_kph": "wind_speed_kph",
    "wind_direction_deg": "wind_direction_deg",
}

TELEMETRY_COLUMNS: dict[str, str] = {
    "lap_uid": "lap_uid",
    "Distance": "distance_m",
    "Speed": "speed_kph",
    "Throttle": "throttle_pct",
    "Brake": "brake_on",          # boolean upstream — not a pressure
    "nGear": "gear",
    "RPM": "rpm",
    "DRS": "drs_raw",             # raw code; drs_open is derived in P2
    "X": "pos_x",
    "Y": "pos_y",
    "Z": "pos_z",
    "session_time_s": "session_time_s",
}


def _project(df: pd.DataFrame, spec: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for src, dst in spec.items():
        out[dst] = df[src] if src in df.columns else pd.NA
    return out


def write_session(
    out_dir: Path,
    laps: pd.DataFrame,
    telemetry_frames: list[pd.DataFrame],
    meta: dict,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    lap_out = _project(laps, LAP_COLUMNS)
    lap_path = out_dir / "laps.parquet"
    lap_out.to_parquet(lap_path, index=False, compression="zstd")

    tel_path = out_dir / "telemetry.parquet"
    n_samples = 0
    if telemetry_frames:
        tel = pd.concat(telemetry_frames, ignore_index=True)
        tel_out = _project(tel, TELEMETRY_COLUMNS)
        # Keep bronze small: these are float64 by default and we do not need it.
        for c in ("distance_m", "speed_kph", "throttle_pct", "rpm",
                  "pos_x", "pos_y", "pos_z", "session_time_s"):
            if c in tel_out:
                tel_out[c] = pd.to_numeric(tel_out[c], errors="coerce").astype("float32")
        if "gear" in tel_out:
            tel_out["gear"] = pd.to_numeric(tel_out["gear"], errors="coerce").astype("Int8")
        tel_out.to_parquet(tel_path, index=False, compression="zstd")
        n_samples = int(len(tel_out))
    else:
        pd.DataFrame(columns=list(TELEMETRY_COLUMNS.values())).to_parquet(
            tel_path, index=False, compression="zstd")

    meta = {**meta,
            "laps_written": int(len(lap_out)),
            "telemetry_rows": n_samples,
            "laps_bytes": lap_path.stat().st_size,
            "telemetry_bytes": tel_path.stat().st_size,
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out_dir / "session.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def update_manifest(bronze: Path, entries: list[dict]) -> Path:
    """Merge new session entries into the bronze manifest, keyed by session."""
    path = bronze / "manifest.json"
    existing: dict[str, dict] = {}
    if path.exists():
        try:
            existing = {e["key"]: e for e in json.loads(path.read_text()).get("sessions", [])}
        except (json.JSONDecodeError, OSError, KeyError):
            existing = {}

    for e in entries:
        existing[e["key"]] = e

    rows = sorted(existing.values(), key=lambda e: e["key"])
    doc = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sessions": rows,
        "totals": {
            "sessions": len(rows),
            "laps": sum(int(r.get("laps_written", 0)) for r in rows),
            "telemetry_rows": sum(int(r.get("telemetry_rows", 0)) for r in rows),
            "bytes": sum(int(r.get("laps_bytes", 0)) + int(r.get("telemetry_bytes", 0))
                         for r in rows),
        },
    }
    bronze.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str))
    tmp.replace(path)
    return path
