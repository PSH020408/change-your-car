"""P6 — the baseline store: every driver's representative and fastest lap of
every ingested session, precomputed into one JSON per session.

Why precomputed: the API must not carry 160 MB of parquet or a pandas
groupby per request. One JSON per session (a few hundred KB) holds the two
laps the HUD offers per driver, the circuit's segments and SVG, and a list
of every other lap so the docking screen can show what exists.

    make baselines            # python -m app.services.baselines build
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.features import effort as effort_mod

TRACE_COLS = ("distance_m", "speed_kph", "throttle_pct", "brake_on", "gear", "drs_raw")
DRS_OPEN_CODES = {10, 12, 14}


def _f(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _i(x):
    v = _f(x)
    return None if v is None else int(v)


def _lap_meta(lap: pd.Series, effort: str | None) -> dict:
    return {
        "lap_uid": str(lap["lap_uid"]), "driver": str(lap["driver"]),
        "team": lap.get("team"), "chassis": lap.get("chassis"), "power_unit": lap.get("power_unit"),
        "lap_number": _i(lap.get("lap_number")), "lap_time_s": _f(lap.get("lap_time_s")),
        "compound": lap.get("compound"), "tyre_life": _i(lap.get("tyre_life")),
        "fresh_tyre": None if pd.isna(lap.get("fresh_tyre")) else bool(lap.get("fresh_tyre")),
        "track_temp_c": _f(lap.get("track_temp_c")), "air_temp_c": _f(lap.get("air_temp_c")),
        "telemetry_quality": lap.get("telemetry_quality"), "effort_class": effort,
        "gap_ahead_s": _f(lap.get("gap_ahead_s")),
        "sector_times_s": [_f(lap.get("sector1_s")), _f(lap.get("sector2_s")), _f(lap.get("sector3_s"))],
        "condition": lap.get("condition"),
    }


def _trace(tel: pd.DataFrame, lap_len: float) -> dict:
    t = tel.sort_values("distance_m").drop_duplicates("distance_m")
    d = t["distance_m"].to_numpy(float)
    scale = lap_len / float(d.max()) if d.max() > 0 else 1.0
    out = {"distance_m": [round(x * scale, 1) for x in d],
           "speed_kph": [round(float(x), 1) for x in t["speed_kph"].to_numpy(float)],
           "throttle_pct": [round(float(x), 1) if not pd.isna(x) else 0.0 for x in t.get("throttle_pct", pd.Series(0, index=t.index))],
           "brake_on": [bool(x) for x in t.get("brake_on", pd.Series(False, index=t.index)).fillna(False)],
           "gear": [int(x) if not pd.isna(x) else 0 for x in t.get("gear", pd.Series(0, index=t.index))],
           "drs_open": [int(x) in DRS_OPEN_CODES if not pd.isna(x) else False for x in t.get("drs_raw", pd.Series(0, index=t.index))]}
    return out


def pick_laps(laps: pd.DataFrame, tel: pd.DataFrame, driver: str) -> tuple[pd.Series | None, pd.Series | None, pd.DataFrame]:
    """(representative, fastest, all-laps-with-effort) for one driver.

    Representative = median lap time among clean-ish push/moderate laps in
    clean air (race) — the HUD default. Fastest = min lap time. Effort is
    classified from the lap's own pedals (pipeline.features.effort).
    """
    d = laps[laps["driver"].astype(str) == driver].dropna(subset=["lap_time_s"]).copy()
    if not len(d):
        return None, None, d
    eff = []
    for uid in d["lap_uid"]:
        lt = tel[tel["lap_uid"] == uid]
        e = effort_mod.lap_effort(lt) if len(lt) >= 20 else {"effort_index": float("nan")}
        eff.append(effort_mod.classify_effort(e["effort_index"]))
    d["effort_class"] = eff
    cand = d
    for name, mask in (("quality", d["telemetry_quality"].isin(["clean", "normal"]) if "telemetry_quality" in d else None),
                       ("effort", d["effort_class"].isin(["push", "moderate"])),
                       ("gap", (pd.to_numeric(d.get("gap_ahead_s"), errors="coerce").fillna(99) >= 2.5)
                        if str(d["session"].iloc[0]) == "R" and "gap_ahead_s" in d else None)):
        if mask is None:
            continue
        nxt = cand[mask.loc[cand.index]]
        if len(nxt):
            cand = nxt
    med = cand["lap_time_s"].median()
    rep = cand.iloc[(cand["lap_time_s"] - med).abs().argsort().iloc[0]]
    fast = d.loc[d["lap_time_s"].idxmin()]
    return rep, fast, d


def build_session(bronze_dir: Path, track_json: Path) -> dict:
    laps = pd.read_parquet(bronze_dir / "laps.parquet")
    tel = pd.read_parquet(bronze_dir / "telemetry.parquet")
    track = json.loads(track_json.read_text())
    meta = json.loads((bronze_dir / "session.json").read_text())
    lap_len = float(track["geometry"]["lap_length_m"])
    pub = track.get("published_reference") or {}
    doc = {
        "season": int(meta["season"]), "event": meta["event_slug"], "event_name": meta["event"],
        "session": meta["session"], "circuit": meta.get("circuit"),
        "track": {"view_box": track["track_map"]["view_box"], "path": track["track_map"]["path"],
                  "sector_boundaries_m": track.get("sector_boundaries_m") or [], "lap_length_m": lap_len,
                  "published_turns": pub.get("published_turns") if pub.get("applicable") else None,
                  "measured_turns": int(track["counts"]["turns"]) if "counts" in track and "turns" in track["counts"]
                  else int(sum(1 for s in track["segments"] if s["kind"] != "straight"))},
        "segments": [{"index": s["index"], "kind": s["kind"], "start_m": s["start_m"], "end_m": s["end_m"],
                      "length_m": s["length_m"], "sector": s.get("sector"),
                      "min_radius_m": s.get("interior_min_radius_m") or s.get("min_radius_m"),
                      "direction": s.get("direction"), "peak_curvature_1pm": s.get("peak_curvature_1pm"),
                      "wraps_start_finish": s.get("wraps_start_finish", False)} for s in track["segments"]],
        "session_temps": {"track_temp_c": _f(pd.to_numeric(laps.get("track_temp_c"), errors="coerce").median()),
                          "air_temp_c": _f(pd.to_numeric(laps.get("air_temp_c"), errors="coerce").median())},
        "drivers": {},
    }
    for drv in sorted(laps["driver"].astype(str).unique()):
        rep, fast, all_laps = pick_laps(laps, tel, drv)
        if rep is None:
            continue
        chosen = {}
        for key, lap in (("representative", rep), ("fastest", fast)):
            uid = str(lap["lap_uid"])
            if uid in {v["lap_uid"] for v in chosen.values()}:
                chosen[key] = {"alias_of": [k for k, v in chosen.items() if v["lap_uid"] == uid][0], "lap_uid": uid}
                continue
            lt = tel[tel["lap_uid"] == uid]
            if len(lt) < 20:
                continue
            m = _lap_meta(lap, str(all_laps.loc[lap.name, "effort_class"]))
            m["trace"] = _trace(lt, lap_len)
            chosen[key] = m
        if not chosen:
            continue
        doc["drivers"][drv] = {
            "team": rep.get("team"), "chassis": rep.get("chassis"), "power_unit": rep.get("power_unit"),
            "laps": chosen,
            "available": [{"lap_uid": str(r["lap_uid"]), "lap_number": _i(r.get("lap_number")),
                           "lap_time_s": _f(r.get("lap_time_s")), "compound": r.get("compound"),
                           "tyre_life": _i(r.get("tyre_life")), "effort_class": r.get("effort_class"),
                           "telemetry_quality": r.get("telemetry_quality"), "gap_ahead_s": _f(r.get("gap_ahead_s"))}
                          for _, r in all_laps.sort_values("lap_number").iterrows()],
        }
    return doc


def build_all(lake: Path, out: Path, verbose: bool = True) -> dict:
    bronze, silver = lake / "bronze", lake / "silver"
    out.mkdir(parents=True, exist_ok=True)
    index: dict = {"seasons": {}}
    n = 0
    for sj in sorted(bronze.glob("*/*/*/session.json")):
        sdir = sj.parent
        season, slug, ses = sdir.parts[-3], sdir.parts[-2], sdir.parts[-1]
        tj = silver / season / slug / "track.json"
        if not tj.exists() or not (sdir / "telemetry.parquet").exists():
            continue
        try:
            doc = build_session(sdir, tj)
        except Exception as exc:                   # noqa: BLE001
            print(f"  FAILED {season}/{slug}/{ses}: {type(exc).__name__}: {exc}")
            continue
        if not doc["drivers"]:
            # a session with no usable lap (e.g. every lap filtered) must not become an empty menu entry
            print(f"  skipped {season}/{slug}/{ses}: no drivers with a usable lap")
            continue
        d = out / season / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{ses}.json").write_text(json.dumps(doc, separators=(",", ":")))
        ev = index["seasons"].setdefault(season, {}).setdefault(slug, {"event_name": doc["event_name"],
                                                                        "circuit": doc["circuit"], "sessions": {}})
        ev["sessions"][ses] = {"drivers": sorted(doc["drivers"].keys()),
                               "chassis": sorted({v["chassis"] for v in doc["drivers"].values() if v.get("chassis")}),
                               "temps": doc["session_temps"]}
        n += 1
        if verbose:
            kb = (d / f"{ses}.json").stat().st_size // 1024
            print(f"  {season}/{slug:<32s} {ses:<3s} {len(doc['drivers']):>2d} drivers  {kb:>5d} KB")
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"\nbaselines -> {out}   {n} sessions")
    return index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--lake", type=Path, default=Path("../data"))
    ap.add_argument("--out", type=Path, default=Path("../data/artifacts/baselines"))
    a = ap.parse_args()
    build_all(a.lake, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
