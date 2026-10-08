"""P11-1 — race timing sidecar: every lap of every race, pit laps included.

Bronze deliberately drops pit in/out laps (they are not laps the HUD should
learn pace from). A tyre-strategy mode needs exactly those laps: the cost of a
pit stop is the in-lap plus the out-lap compared with the laps around them,
and the thing a strategy is judged on is the race total, which includes them.

So this step reads the race sessions once more from the FastF1 cache - timing
only, no telemetry, offline - and writes beside each race's bronze folder:

    race_timing.json
      laps        per driver: lap_number, lap_time_s, pit_in, pit_out, compound,
                  tyre_life, stint, position, track_status
      totals      per driver: laps completed, race time (sum of timed laps),
                  whether every lap was timed, classified finisher or not
      pit_loss    per stop: (in-lap + out-lap) - 2 x the median of the driver's
                  clean green-flag laps around it; event median / p10 / p90 / n
      green_share share of laps run under green flag (SC / VSC / red flag
                  make a race unusable for a strategy back-test)

Usage
-----
    python -m pipeline.ingest.race_timing --scope configs/scope.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.ingest import loader, paths

GREEN = "1"
NEIGHBOURS = 3           # clean laps on each side of a stop used as the reference pace


def _secs(x) -> float | None:
    if x is None or pd.isna(x):
        return None
    return float(pd.to_timedelta(x).total_seconds())


def _green(ts) -> bool:
    # TrackStatus is a string of codes; '1' alone = green. '12' = green then SC etc.
    return str(ts) == GREEN


NEUTRAL_CODES = set("4567")      # 4 safety car, 5 red flag, 6 VSC, 7 VSC ending


def _neutralised(ts) -> bool:
    """A lap run (even partly) under SC / VSC / red flag: its time says nothing about pace."""
    return any(ch in NEUTRAL_CODES for ch in str(ts or ""))


def stints_of(laps: list[dict]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for l in sorted(laps, key=lambda x: x["lap_number"]):
        c = l.get("compound")
        if not c:
            return out
        if out and out[-1][0] == c and not l.get("pit_out"):
            out[-1] = (c, out[-1][1] + 1)
        else:
            out.append((c, 1))
    return out


def driver_laps(g: pd.DataFrame) -> list[dict]:
    g = g.sort_values("LapNumber")
    rows = []
    for _, r in g.iterrows():
        rows.append({
            "lap_number": int(r["LapNumber"]),
            "lap_time_s": _secs(r.get("LapTime")),
            "pit_in": bool(pd.notna(r.get("PitInTime"))),
            "pit_out": bool(pd.notna(r.get("PitOutTime"))),
            "compound": str(r.get("Compound")) if pd.notna(r.get("Compound")) else None,
            "tyre_life": int(r["TyreLife"]) if pd.notna(r.get("TyreLife")) else None,
            "stint": int(r["Stint"]) if pd.notna(r.get("Stint")) else None,
            "position": int(r["Position"]) if pd.notna(r.get("Position")) else None,
            "track_status": str(r.get("TrackStatus")) if pd.notna(r.get("TrackStatus")) else None,
            "neutralised": _neutralised(r.get("TrackStatus")),
        })
    return rows


def pit_losses(laps: list[dict]) -> list[dict]:
    """One entry per stop whose in-lap, out-lap and reference laps were all green and timed."""
    out = []
    by_n = {l["lap_number"]: l for l in laps}
    for l in laps:
        if not l["pit_in"]:
            continue
        n = l["lap_number"]
        out_lap = by_n.get(n + 1)
        if not out_lap or not out_lap["pit_out"]:
            continue
        if l["lap_time_s"] is None or out_lap["lap_time_s"] is None:
            continue
        if not (_green(l["track_status"]) and _green(out_lap["track_status"])):
            continue
        ref = []
        for k in list(range(n - NEIGHBOURS, n)) + list(range(n + 2, n + 2 + NEIGHBOURS)):
            r = by_n.get(k)
            if r and r["lap_time_s"] and not r["pit_in"] and not r["pit_out"] and _green(r["track_status"]):
                ref.append(r["lap_time_s"])
        if len(ref) < 3:
            continue
        loss = l["lap_time_s"] + out_lap["lap_time_s"] - 2.0 * float(np.median(ref))
        out.append({"in_lap": n, "loss_s": round(loss, 3), "reference_s": round(float(np.median(ref)), 3),
                    "n_reference": len(ref), "from": l["compound"], "to": out_lap["compound"]})
    return out


def build(sess: loader.LoadedSession, fill: dict | None = None) -> dict:
    laps = sess.laps.copy()
    # 2024 Baku: FastF1's minimal driver list leaves Driver blank for every lap, which would
    # fold twenty cars into one "driver". Borrow abbreviations from the weekend's sibling
    # sessions in bronze (same fix as ingest, defect #30); drop what cannot be named.
    if "Driver" in laps:
        abbr = laps["Driver"].astype(str).str.strip()
        bad = ~abbr.str.fullmatch(r"[A-Z]{3}")
        if bool(bad.any()) and fill and "DriverNumber" in laps:
            num = laps["DriverNumber"].astype(str).str.strip()
            laps.loc[bad, "Driver"] = num[bad].map(lambda n: fill.get(n, {}).get("driver"))
            abbr = laps["Driver"].astype(str).str.strip()
            bad = ~abbr.str.fullmatch(r"[A-Z]{3}")
        laps = laps[~bad]
    drivers = {}
    all_losses = []
    max_laps = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())
    n_green = 0
    n_total = 0
    for drv, g in laps.groupby("Driver"):
        rows = driver_laps(g)
        timed = [r["lap_time_s"] for r in rows if r["lap_time_s"] is not None]
        losses = pit_losses(rows)
        for x in losses:
            all_losses.append({"driver": str(drv), **x})
        n_green += sum(1 for r in rows if _green(r["track_status"]))
        n_total += len(rows)
        drivers[str(drv)] = {
            "laps": rows,
            "totals": {"laps": len(rows), "timed_laps": len(timed), "race_time_s": round(float(sum(timed)), 3),
                       # FastF1 never times lap 1 (no start-line crossing before it); "all timed" = every lap after it
                       "all_timed": len(timed) >= len(rows) - 1,
                       "classified_finisher": len(rows) >= max_laps - 1,     # lapped cars finish a lap short
                       "stops": sum(1 for r in rows if r["pit_in"])},
            "pit_stops": losses,
        }
    loss_vals = np.array([x["loss_s"] for x in all_losses], dtype=float)
    return {
        "season": sess.season, "event": sess.event, "event_slug": paths.slug(sess.event), "session": sess.session,
        "race_laps": max_laps,
        "green_share": round(n_green / max(n_total, 1), 4),
        "pit_loss": ({"median_s": round(float(np.median(loss_vals)), 3), "p10_s": round(float(np.quantile(loss_vals, 0.1)), 3),
                      "p90_s": round(float(np.quantile(loss_vals, 0.9)), 3), "n": int(len(loss_vals))}
                     if len(loss_vals) else {"median_s": None, "n": 0}),
        "drivers": drivers,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def store_doc(doc: dict) -> dict:
    """The part the API needs: race length, pit loss, flag share, per-driver stop counts."""
    stint_max: dict[str, int] = {}
    for v in doc["drivers"].values():
        for c, n in stints_of(v["laps"]):
            stint_max[c] = max(stint_max.get(c, 0), n)
    return {"race_laps": doc["race_laps"], "green_share": doc["green_share"], "pit_loss": doc["pit_loss"],
            "stops": {d: v["totals"]["stops"] for d, v in doc["drivers"].items()},
            # the longest stint anyone really ran per compound, pit laps included - the honest
            # upper bound for a planned stint (the bronze-based envelope can read 1-2 laps short)
            "stint_max": stint_max,
            "built_at": doc["built_at"]}


def run(scope_path: Path, only: set[str] | None, force: bool, verbose: bool,
        store: Path | None = Path("../data/artifacts/baselines")) -> int:
    import fastf1 as ff1
    scope = paths.load_scope(scope_path)
    cache = paths.cache_dir(scope_path, scope)
    bronze = paths.bronze_dir(scope_path, scope)
    loader.enable_cache(ff1, cache)
    ff1.Cache.offline_mode(True)

    todo = []
    for sj in sorted(bronze.rglob("session.json")):
        meta = json.loads(sj.read_text())
        if str(meta["session"]).upper() != "R":
            continue
        key = f"{meta['season']}/{meta['event_slug']}"
        if only and not ({meta["event_slug"], key} & only):
            continue
        if (sj.parent / "race_timing.json").exists() and not force:
            continue
        todo.append((meta, sj.parent))
    print(f"cache      : {cache}")
    print(f"races      : {len(todo)} to build")
    rows, failures = [], []
    for i, (meta, out) in enumerate(todo, 1):
        label = f"{meta['season']}/{meta['event_slug']}"
        try:
            sess = loader.load_session_timing(ff1, int(meta["season"]), meta["event"], "R")
            from pipeline.ingest.run import _driver_lookup_from_siblings
            fill = _driver_lookup_from_siblings(bronze, int(meta["season"]), meta["event_slug"], "R")
            doc = build(sess, fill=fill)
            (out / "race_timing.json").write_text(json.dumps(doc, separators=(",", ":")))
            if store is not None:
                sd = store / str(meta["season"]) / meta["event_slug"]
                if sd.exists():
                    (sd / "race.json").write_text(json.dumps(store_doc(doc), separators=(",", ":")))
            pl = doc["pit_loss"]
            fin = sum(1 for d in doc["drivers"].values() if d["totals"]["classified_finisher"] and d["totals"]["all_timed"])
            rows.append({"event": label, "laps": doc["race_laps"], "green": doc["green_share"],
                         "pit_median": pl.get("median_s"), "pit_n": pl["n"], "finishers_timed": fin})
            print(f"[{i:>3}/{len(todo)}] {label:<36s} {doc['race_laps']:>3} laps  green {doc['green_share']:.2f}  "
                  f"pit loss {pl.get('median_s') or float('nan'):5.1f} s (n={pl['n']:>2})  finishers fully timed {fin:>2}")
        except Exception as exc:                                   # noqa: BLE001
            failures.append({"event": label, "error": f"{type(exc).__name__}: {exc}"[:200]})
            print(f"[{i:>3}/{len(todo)}] {label:<36s} FAILED {exc}"[:150], file=sys.stderr)
    if rows:
        df = pd.DataFrame(rows)
        print()
        print(f"pit loss across {len(df)} races: median {df['pit_median'].median():.1f} s, "
              f"min {df['pit_median'].min():.1f} ({df.loc[df['pit_median'].idxmin(), 'event']}), "
              f"max {df['pit_median'].max():.1f} ({df.loc[df['pit_median'].idxmax(), 'event']})")
        print(f"green-flag share: {(df['green'] >= 0.9).sum()} of {len(df)} races >= 0.90")
    if failures:
        print(f"failures: {len(failures)}  e.g. {failures[0]}")
    return 0 if rows or not todo else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", type=Path, default=Path("configs/scope.yaml"))
    ap.add_argument("--only", default=None, help="comma-separated event slugs or season/slug")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    only = {x.strip() for x in a.only.split(",")} if a.only else None
    sys.exit(run(a.scope, only, a.force, a.verbose))


if __name__ == "__main__":
    main()
