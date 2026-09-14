"""Data reconnaissance — answer "what do we actually have?" before designing features.

This runs BEFORE the ingestion pipeline is written, on a handful of sessions,
and produces the document that the feature design (P2) is built from. It
deliberately asks the questions that are expensive to get wrong later:

  1. DATA DICTIONARY   every column FastF1 hands us, its dtype, its null rate,
                       and a real example value — not the docs' version, ours
  2. COVERAGE MATRIX   per season x session: do laps / car telemetry /
                       position data / weather actually exist? Coverage is not
                       uniform across 2022-2025 and the gaps decide the scope
  3. SAMPLE RATE       measured median dt of car and position channels. The
                       10 m resample grid is only defensible if the raw rate
                       supports it at the speeds we care about
  4. LAP YIELD         raw laps -> after each filter. If the 107% + accuracy +
                       track-status filters leave 30% of laps, the training
                       set is far smaller than the raw counts suggest
  5. SIZE              measured cache cost per session, which is the only
                       honest basis for projecting a full-era download
  6. ISSUES            concrete anomalies worth knowing now

Usage
-----
    python -m pipeline.recon.survey --scope configs/scope.yaml
    python -m pipeline.recon.survey --scope configs/scope.yaml --out ../docs/recon
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# --------------------------------------------------------------------- helpers
def human(n: float) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} TB"


def describe_frame(df: pd.DataFrame, sample_rows: int = 1) -> list[dict]:
    """Column-level profile: dtype, null rate, cardinality, an example value."""
    out = []
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        nulls = int(s.isna().sum())
        example = ""
        nn = s.dropna()
        if len(nn):
            example = str(nn.iloc[0])[:48]
        try:
            card = int(s.nunique(dropna=True))
        except TypeError:
            card = -1                     # unhashable (rare, e.g. list columns)
        out.append({
            "column": str(col),
            "dtype": str(s.dtype),
            "null_pct": round(100.0 * nulls / n, 2),
            "cardinality": card,
            "example": example,
        })
    return out


def median_dt_ms(series: pd.Series) -> float | None:
    """Median sampling interval of a time-like column, in milliseconds."""
    if series is None or len(series) < 3:
        return None
    d = pd.Series(series).diff().dropna()
    if not len(d):
        return None
    try:
        return float(d.dt.total_seconds().median() * 1000.0)
    except AttributeError:
        return None



def resolve_cache_dir(scope_path: Path, scope: dict) -> Path:
    """Cache location, read from config rather than walked from __file__.

    The previous version did `scope_path.parent.parent.parent`, which
    saturates at "." for a relative path and silently put the cache under
    backend/ instead of the repo root. The path is now explicit in
    scope.yaml (docs/recon/DECISIONS.md D9).
    """
    cfg = (scope.get("paths") or {}).get("cache_dir", "data/cache")
    base = scope_path.resolve().parent.parent          # -> backend/
    return (base / cfg).resolve() if not Path(cfg).is_absolute() else Path(cfg)

# ------------------------------------------------------------------ per session
def survey_session(ff1, season: int, event: str, ident: str) -> dict:
    rec: dict = {
        "season": season, "event": event, "session": ident,
        "ok": False, "error": "",
    }

    session = ff1.get_session(season, event, ident)
    session.load(laps=True, telemetry=True, weather=True, messages=True)

    laps = session.laps
    rec["official_name"] = str(getattr(session, "name", ident))
    rec["circuit"] = str(getattr(getattr(session, "event", None), "Location", "") or "")

    # ---- 1. laps ---------------------------------------------------------
    if laps is None or not len(laps):
        rec["error"] = "no laps in session"
        return rec

    rec["laps_raw"] = int(len(laps))
    rec["drivers"] = int(laps["Driver"].nunique())
    rec["lap_columns"] = describe_frame(laps)

    # ---- 2. lap yield through each filter --------------------------------
    # Applied cumulatively, in the order the real pipeline will apply them,
    # so the last number is the actual training-set size per session.
    yield_steps: list[dict] = [{"step": "raw", "laps": int(len(laps))}]
    cur = laps
    try:
        cur = cur.pick_wo_box()
        yield_steps.append({"step": "drop pit in/out", "laps": int(len(cur))})
        cur = cur.pick_not_deleted()
        yield_steps.append({"step": "drop deleted", "laps": int(len(cur))})
        cur = cur.pick_accurate()
        yield_steps.append({"step": "accurate only", "laps": int(len(cur))})
        cur = cur.pick_quicklaps(1.07)
        yield_steps.append({"step": "within 107%", "laps": int(len(cur))})
        cur = cur.pick_track_status("457", how="none")
        yield_steps.append({"step": "green flag only", "laps": int(len(cur))})
    except Exception as exc:                              # noqa: BLE001
        yield_steps.append({"step": f"FILTER ERROR: {type(exc).__name__}: {exc}"[:160], "laps": -1})

    rec["lap_yield"] = yield_steps
    rec["laps_usable"] = int(len(cur)) if len(yield_steps) and yield_steps[-1]["laps"] >= 0 else 0
    rec["yield_pct"] = round(100.0 * rec["laps_usable"] / rec["laps_raw"], 1)

    # ---- 3. telemetry on the fastest lap ---------------------------------
    try:
        fl = laps.pick_fastest()
        car = fl.get_car_data()
        pos = fl.get_pos_data()

        rec["car_columns"] = describe_frame(car)
        rec["pos_columns"] = describe_frame(pos)
        rec["car_samples"] = int(len(car))
        rec["pos_samples"] = int(len(pos))
        rec["car_dt_ms"] = median_dt_ms(car["Date"]) if "Date" in car else None
        rec["pos_dt_ms"] = median_dt_ms(pos["Date"]) if "Date" in pos else None

        # spatial resolution: how far does the car travel between samples?
        with_dist = car.add_distance()
        if "Distance" in with_dist and len(with_dist) > 2:
            step = pd.Series(with_dist["Distance"]).diff().dropna()
            rec["lap_distance_m"] = round(float(with_dist["Distance"].iloc[-1]), 1)
            rec["sample_spacing_m"] = {
                "median": round(float(step.median()), 2),
                "p95": round(float(step.quantile(0.95)), 2),
                "max": round(float(step.max()), 2),
            }
        if "Speed" in car and len(car):
            rec["top_speed_kph"] = round(float(car["Speed"].max()), 1)
    except Exception as exc:                              # noqa: BLE001
        rec["telemetry_error"] = f"{type(exc).__name__}: {exc}"[:200]

    # ---- 4. telemetry availability across ALL drivers --------------------
    # A driver who crashed out on lap 1 has no telemetry for a legitimate
    # reason. Lumping that together with a genuine data hole turns the issue
    # list into noise (docs/recon/DECISIONS.md D7), so classify it.
    missing, retired = [], []
    DNF_LAP_THRESHOLD = 3
    total_laps = int(laps["LapNumber"].max()) if "LapNumber" in laps else 0
    for drv in sorted(laps["Driver"].dropna().unique()):
        try:
            d_laps = laps.pick_drivers(drv)
            n = len(d_laps)
            if not n:
                retired.append({"driver": str(drv), "reason": "no laps recorded"})
                continue
            completed = int(d_laps["LapNumber"].max()) if "LapNumber" in d_laps else n
            t = d_laps.pick_fastest().get_car_data()
            if t is None or len(t) < 10:
                early_exit = (n <= DNF_LAP_THRESHOLD) or (
                    total_laps and completed < 0.25 * total_laps)
                (retired if early_exit else missing).append({
                    "driver": str(drv),
                    "laps": n,
                    "completed_to": completed,
                    "reason": "retired early — expected" if early_exit
                              else "laps exist but telemetry empty",
                })
        except Exception as exc:                          # noqa: BLE001
            missing.append({"driver": str(drv), "reason": f"{type(exc).__name__}"})
    rec["drivers_without_telemetry"] = missing      # genuine data holes
    rec["drivers_retired"] = retired                # legitimate absences

    # ---- 5. weather ------------------------------------------------------
    w = getattr(session, "weather_data", None)
    if w is None or not len(w):
        rec["weather"] = {"rows": 0}
    else:
        rec["weather"] = {
            "rows": int(len(w)),
            "dt_ms": median_dt_ms(w["Time"]) if "Time" in w else None,
            "track_temp_c": [round(float(w["TrackTemp"].min()), 1),
                             round(float(w["TrackTemp"].max()), 1)] if "TrackTemp" in w else None,
            "air_temp_c": [round(float(w["AirTemp"].min()), 1),
                           round(float(w["AirTemp"].max()), 1)] if "AirTemp" in w else None,
            "rainfall_any": bool(w["Rainfall"].any()) if "Rainfall" in w else None,
        }
        rec["weather_columns"] = describe_frame(w)

    # ---- 6. compounds present -------------------------------------------
    if "Compound" in laps:
        rec["compounds"] = {str(k): int(v) for k, v in
                            laps["Compound"].value_counts(dropna=True).items()}

    rec["ok"] = True
    return rec


# ---------------------------------------------------------------------- report
def render_markdown(results: list[dict], cache_info: tuple, scope: dict) -> str:
    ok = [r for r in results if r.get("ok")]
    L: list[str] = []
    A = L.append

    A("# Data Reconnaissance Report")
    A("")
    A(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    A("")
    A(f"Sessions surveyed: **{len(ok)} / {len(results)}** · "
      f"Era: **{scope['era']['name']}** {scope['era']['seasons']}")
    if cache_info and cache_info[1]:
        A(f"Cache: `{cache_info[0]}` — **{human(cache_info[1])}**")
    A("")

    # --- coverage ---------------------------------------------------------
    A("## 1. Coverage matrix")
    A("")
    A("| Season | Event | Ses | Laps raw | Usable | Yield | Drivers | No telem | Weather | Car dt |")
    A("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        if not r.get("ok"):
            A(f"| {r['season']} | {r['event']} | {r['session']} | "
              f"— | — | — | — | — | — | **{r.get('error','failed')[:40]}** |")
            continue
        A(f"| {r['season']} | {r['event']} | {r['session']} | "
          f"{r.get('laps_raw',0)} | {r.get('laps_usable',0)} | "
          f"{r.get('yield_pct',0)}% | {r.get('drivers',0)} | "
          f"{len(r.get('drivers_without_telemetry', []))} | "
          f"{r.get('weather',{}).get('rows',0)} | "
          f"{(str(round(r['car_dt_ms'])) + ' ms') if r.get('car_dt_ms') else '—'} |")
    A("")

    # --- sampling ---------------------------------------------------------
    A("## 2. Sampling resolution")
    A("")
    A("Does the raw sample rate support a 10 m resample grid? The grid is only")
    A("defensible where sample spacing stays below it — check the p95 column.")
    A("")
    A("| Season | Event | Lap length | Samples | Spacing median | p95 | max | Top speed |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in ok:
        sp = r.get("sample_spacing_m") or {}
        A(f"| {r['season']} | {r['event']} | "
          f"{r.get('lap_distance_m','—')} m | {r.get('car_samples','—')} | "
          f"{sp.get('median','—')} m | {sp.get('p95','—')} m | {sp.get('max','—')} m | "
          f"{r.get('top_speed_kph','—')} km/h |")
    A("")
    worst = max((r.get("sample_spacing_m", {}).get("p95", 0) or 0) for r in ok) if ok else 0
    if worst:
        verdict = "OK — grid is coarser than the raw spacing" if worst <= 10 else \
                  "**WARNING — raw spacing exceeds the 10 m grid at speed; interpolation will invent detail**"
        A(f"Worst p95 spacing: **{worst} m**. {verdict}")
        A("")

    # --- lap yield --------------------------------------------------------
    A("## 3. Lap yield through the filter chain")
    A("")
    A("Cumulative, in pipeline order. The last row is the real training-set size.")
    A("")
    steps: dict[str, list[int]] = defaultdict(list)
    for r in ok:
        for s in r.get("lap_yield", []):
            if s["laps"] >= 0:
                steps[s["step"]].append(s["laps"])
    if steps:
        first = next(iter(steps.values()))
        base = sum(first) if first else 1
        A("| Filter step | Laps remaining | % of raw |")
        A("|---|---:|---:|")
        for k, v in steps.items():
            tot = sum(v)
            A(f"| {k} | {tot} | {round(100.0*tot/max(base,1),1)}% |")
    A("")

    # --- dictionaries -----------------------------------------------------
    A("## 4. Data dictionary")
    A("")
    for title, key in (("Laps", "lap_columns"), ("Car telemetry", "car_columns"),
                       ("Position telemetry", "pos_columns"), ("Weather", "weather_columns")):
        ref = next((r for r in ok if r.get(key)), None)
        if not ref:
            continue
        A(f"### {title}")
        A("")
        A(f"_from {ref['season']} {ref['event']} {ref['session']}_")
        A("")
        A("| Column | dtype | null % | cardinality | example |")
        A("|---|---|---:|---:|---|")
        for c in ref[key]:
            A(f"| `{c['column']}` | {c['dtype']} | {c['null_pct']} | "
              f"{c['cardinality']} | {c['example']} |")
        A("")

    # --- compounds --------------------------------------------------------
    comp: dict[str, int] = defaultdict(int)
    for r in ok:
        for k, v in (r.get("compounds") or {}).items():
            comp[k] += v
    if comp:
        A("## 5. Tyre compounds observed")
        A("")
        A("| Compound | Laps |")
        A("|---|---:|")
        for k, v in sorted(comp.items(), key=lambda kv: -kv[1]):
            A(f"| {k} | {v} |")
        A("")

    # --- issues -----------------------------------------------------------
    A("## 6. Issues to resolve before P2 (feature design)")
    A("")
    issues: list[str] = []
    for r in results:
        if not r.get("ok"):
            issues.append(f"`{r['season']} {r['event']} {r['session']}` — failed: {r.get('error','')}")
            continue
        if r.get("telemetry_error"):
            issues.append(f"`{r['season']} {r['event']} {r['session']}` — telemetry: {r['telemetry_error']}")
        miss = r.get("drivers_without_telemetry", [])
        if miss:
            issues.append(f"`{r['season']} {r['event']} {r['session']}` — "
                          f"{len(miss)} driver(s) with laps but no telemetry: "
                          + ", ".join(m["driver"] for m in miss[:8]))
        if r.get("weather", {}).get("rows", 0) == 0:
            issues.append(f"`{r['season']} {r['event']} {r['session']}` — no weather data")
        # A qualifying session is mostly out / cool-down / in laps; 25-40%
        # is what a correct filter returns there, so only flag RACE sessions
        # (docs/recon/DECISIONS.md D3).
        yp = r.get("yield_pct", 100)
        if r["session"] in ("R", "S") and yp < 40:
            issues.append(f"`{r['season']} {r['event']} {r['session']}` — "
                          f"race lap yield only {yp}%; check whether the pace gate "
                          f"is being computed against a session best set in "
                          f"different conditions")
        elif r["session"] in ("Q", "SQ") and yp < 15:
            issues.append(f"`{r['season']} {r['event']} {r['session']}` — "
                          f"qualifying yield {yp}% is low even for a Q session")
    if issues:
        for i in issues:
            A(f"- {i}")
    else:
        A("- None found in the sampled sessions.")
    A("")

    A("## 7. Decisions this report should settle")
    A("")
    A("- [ ] 10 m resample grid — confirmed or revised to the measured spacing")
    A("- [ ] Which sessions enter training (Q only / Q+R / +FP)")
    A("- [ ] Filter chain final order and thresholds")
    A("- [ ] Setup-proxy feature list (which observable channels stand in for setup)")
    A("- [ ] Full-era download size projection and whether it fits the disk budget")
    A("")
    return "\n".join(L)


# ------------------------------------------------------------------------- main
def run(scope_path: Path, out_dir: Path) -> int:
    import fastf1 as ff1

    scope = yaml.safe_load(scope_path.read_text())
    cache_dir = resolve_cache_dir(scope_path, scope)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ff1.Cache.enable_cache(str(cache_dir))

    samples = scope["recon"]["samples"]
    results: list[dict] = []

    for i, s in enumerate(samples, 1):
        label = f"{s['season']} {s['event']} {s['session']}"
        print(f"[{i}/{len(samples)}] surveying {label} ...", flush=True)
        try:
            results.append(survey_session(ff1, s["season"], s["event"], s["session"]))
        except Exception as exc:                          # noqa: BLE001
            print(f"    failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            results.append({"season": s["season"], "event": s["event"],
                            "session": s["session"], "ok": False,
                            "error": f"{type(exc).__name__}: {exc}"[:200]})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recon_report.json").write_text(
        json.dumps({"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "scope": scope["era"], "sessions": results}, indent=1, default=str))
    md = render_markdown(results, ff1.Cache.get_cache_info(), scope)
    (out_dir / "recon_report.md").write_text(md)

    ok = sum(1 for r in results if r.get("ok"))
    print()
    print(f"surveyed {ok}/{len(results)} sessions")
    print(f"report -> {out_dir / 'recon_report.md'}")
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser(description="Survey FastF1 data before designing features.")
    p.add_argument("--scope", type=Path, default=Path("configs/scope.yaml"))
    p.add_argument("--out", type=Path, default=Path("../docs/recon"))
    args = p.parse_args()
    try:
        sys.exit(run(args.scope, args.out))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:                                     # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
