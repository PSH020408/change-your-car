"""Whole-simulator back-test: qualifying lap -> race lap through the sliders.

Every part of the simulator has been validated on its own (segment model on a
held-out circuit, physics coefficients by regression). This is the first test
of the SUM. For each driver with a qualifying and a race baseline in the same
weekend, take the qualifying representative lap, set only what the race lap
changes — fuel for that lap number, the tyre it was on (compound, age), the
race session's track/air temperature — and compare the simulated lap time
with the race lap actually driven.

Population: race laps of the FIRST stint that the store already marks as
clean push laps in clean air (>= 2.5 s to the car ahead), telemetry clean or
normal, lap >= 3 (laps 1-2 are the start and the first cold-tyre lap).

Baselines the engine must beat:
  Q_only     : the qualifying lap time unchanged
  Q_plus_l2  : Q x (1 + race shift measured by level 2, 4.75 %)
  Q_plus_fuel: Q + physics fuel term only (no ML)
Oracle (not a competitor, an upper bound): Q + that event's own median Q->R gap.

    python -m pipeline.eval.backtest [--limit-events N] [--out ../data/artifacts/backtest]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import get_settings
from app.schemas import domain as S
from app.services.engine import Engine, NotFound

RACE_SHIFT_PCT = 4.75          # level-2 is_race coefficient, model card
MIN_LAP = 3
MIN_GAP_S = 2.5


def first_stint(avail: list[dict]) -> list[dict]:
    """Race laps of the first stint: same compound as the first recorded lap,
    tyre age non-decreasing; stop at the first reset."""
    laps = sorted((a for a in avail if a.get("lap_number")), key=lambda a: a["lap_number"])
    if not laps:
        return []
    comp, last_age, out = laps[0].get("compound"), -1, []
    for a in laps:
        age = a.get("tyre_life") or 0
        if a.get("compound") != comp or age < last_age:
            break
        last_age = age
        out.append(a)
    return out


def usable(a: dict) -> bool:
    return (a.get("lap_time_s") is not None and (a.get("lap_number") or 0) >= MIN_LAP
            and a.get("effort_class") == "push"
            and a.get("telemetry_quality") in ("clean", "normal")
            and (a.get("gap_ahead_s") is None or a["gap_ahead_s"] >= MIN_GAP_S))


def run(out_dir: Path, limit_events: int | None, verbose: bool) -> int:
    t0 = time.perf_counter()
    s = get_settings()
    eng = Engine(baselines_dir=s.model_registry_dir / "baselines", model_dir=s.model_registry_dir / "models")
    from pipeline.physics import modifiers as M
    cfg = eng.physics

    rows: list[dict] = []
    n_events = 0
    for season, events in sorted(eng.index["seasons"].items()):
        for slug, ev in sorted(events.items()):
            if "Q" not in ev["sessions"] or "R" not in ev["sessions"]:
                continue
            if limit_events is not None and n_events >= limit_events:
                break
            n_events += 1
            rdoc = eng._doc(int(season), slug, "R")
            qdoc = eng._doc(int(season), slug, "Q")
            total_laps = max((a.get("lap_number") or 0) for d in rdoc["drivers"].values() for a in d["available"]) or None
            r_temps = rdoc.get("session_temps") or {}
            for drv in sorted(set(qdoc["drivers"]) & set(rdoc["drivers"])):
                ref = S.BaselineRef(season=int(season), event=slug, session="Q", driver=drv, lap="representative")
                try:
                    q = eng.load_baseline(ref)
                except NotFound:
                    continue
                q_time = float(q.lap["lap_time_s"])
                stint = [a for a in first_stint(rdoc["drivers"][drv]["available"]) if usable(a)]
                for a in stint:
                    fuel = M.baseline_fuel_kg("R", a["lap_number"], total_laps, cfg)
                    env = S.Environment(track_temp_c=r_temps.get("track_temp_c"), air_temp_c=r_temps.get("air_temp_c"),
                                        weather=S.Weather.DRY,
                                        compound=S.Compound(a["compound"]) if a.get("compound") in S.Compound.__members__ else None,
                                        tyre_life=a.get("tyre_life"))
                    req = S.SimulationRequest(baseline=ref, setup=S.CarSetup(fuel_kg=fuel), environment=env)
                    try:
                        res = eng.simulate(req)
                    except Exception as exc:                        # noqa: BLE001
                        if verbose:
                            print(f"  skip {season}/{slug}/{drv} L{a['lap_number']}: {type(exc).__name__}: {exc}")
                        continue
                    rows.append({"season": int(season), "event": slug, "driver": drv, "lap_number": a["lap_number"],
                                 "compound": a.get("compound"), "tyre_life": a.get("tyre_life"), "fuel_kg": round(fuel, 1),
                                 "q_time_s": q_time, "actual_s": float(a["lap_time_s"]),
                                 "engine_s": res.lap.simulated_lap_time_s,
                                 "engine_lo_s": q_time + res.lap.delta_lo_s, "engine_hi_s": q_time + res.lap.delta_hi_s,
                                 "physics_s": res.lap.physics_s, "ml_s": res.lap.ml_s})
            if verbose:
                n_ev = sum(1 for r in rows if r["event"] == slug and r["season"] == int(season))
                print(f"[{n_events:3d}] {season}/{slug:<32s} {n_ev:4d} laps")

    if not rows:
        print("no usable Q->R pairs"); return 1
    df = pd.DataFrame(rows)
    df["q_plus_l2_s"] = df["q_time_s"] * (1 + RACE_SHIFT_PCT / 100)
    df["q_plus_fuel_s"] = df["q_time_s"] + df["physics_s"]
    # oracle: the event's own median Q->R gap (uses the answers; an upper bound, not a competitor)
    gap = (df["actual_s"] - df["q_time_s"]).groupby([df["season"], df["event"]]).transform("median")
    df["oracle_s"] = df["q_time_s"] + gap

    def score(col: str) -> dict:
        e = df[col] - df["actual_s"]
        return {"mae_s": round(float(e.abs().mean()), 3), "bias_s": round(float(e.mean()), 3),
                "median_abs_s": round(float(e.abs().median()), 3)}

    summary = {c: score(c) for c in ("q_time_s", "q_plus_l2_s", "q_plus_fuel_s", "engine_s", "oracle_s")}
    cover = float(((df["actual_s"] >= df["engine_lo_s"]) & (df["actual_s"] <= df["engine_hi_s"])).mean())
    by_event = (df.assign(err=df["engine_s"] - df["actual_s"])
                .groupby(["season", "event"])["err"].agg(n="size", mae=lambda x: x.abs().mean(), bias="mean")
                .reset_index().sort_values("mae"))
    by_compound = (df.assign(err=df["engine_s"] - df["actual_s"]).groupby("compound")["err"]
                   .agg(n="size", mae=lambda x: x.abs().mean(), bias="mean"))

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "laps.parquet", index=False)
    report = {"n_laps": int(len(df)), "n_events": int(by_event.shape[0]), "n_drivers": int(df["driver"].nunique()),
              "population": "race first stint, clean push, clean air, lap>=3, telemetry clean/normal",
              "engine_band80_coverage": round(cover, 3), "summary": summary,
              "by_compound": {str(k): {kk: round(float(vv), 3) for kk, vv in v.items()} for k, v in by_compound.iterrows()},
              "by_event": [{"season": int(r.season), "event": r.event, "n": int(r.n), "mae_s": round(float(r.mae), 3), "bias_s": round(float(r.bias), 3)}
                           for r in by_event.itertuples()],
              "model_version": eng.model_version, "seconds": round(time.perf_counter() - t0, 1)}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))

    print(f"\nBACK-TEST  qualifying lap -> race first-stint lap   {len(df):,} laps · {report['n_events']} events · {report['n_drivers']} drivers · model {eng.model_version}")
    print(f"  {'predictor':<14s} {'MAE':>7s} {'bias':>7s} {'median|e|':>10s}")
    for k, lab in (("q_time_s", "Q only"), ("q_plus_l2_s", "Q + l2 4.75%"), ("q_plus_fuel_s", "Q + fuel"), ("engine_s", "ENGINE"), ("oracle_s", "oracle")):
        v = summary[k]; print(f"  {lab:<14s} {v['mae_s']:7.3f} {v['bias_s']:+7.3f} {v['median_abs_s']:10.3f}")
    print(f"  engine 80 % band covers the race lap in {cover*100:.0f} % of cases")
    print("  by compound:")
    for k, v in by_compound.iterrows():
        print(f"    {k:<8s} n={int(v['n']):5d}  MAE {v['mae']:.3f}  bias {v['bias']:+.3f}")
    print(f"  best / worst events by MAE: {by_event.iloc[0]['event']} {by_event.iloc[0]['mae']:.3f} / {by_event.iloc[-1]['event']} {by_event.iloc[-1]['mae']:.3f}")
    print(f"report -> {out_dir / 'report.json'}   ({report['seconds']} s)")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Qualifying -> race back-test of the whole simulator.")
    p.add_argument("--out", type=Path, default=Path("../data/artifacts/backtest"))
    p.add_argument("--limit-events", type=int, default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args()
    sys.exit(run(a.out, a.limit_events, a.verbose))


if __name__ == "__main__":
    main()
