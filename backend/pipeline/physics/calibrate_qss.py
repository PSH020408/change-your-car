"""P9-3 — calibrate the QSS engine against real laps and grade it.

For every stored baseline lap the engine gets the circuit's line and the lap's
own conditions (fuel estimate, air density, where DRS was open) and THREE
free parameters: mu (effective grip), ClA (downforce area), CdA (drag area).
They are fitted to the measured SPEED trace only. The lap time is never a
target, so the lap-time error that comes out is a genuine measure of how
well a point-mass car reproduces a real one — the number the README quotes.

Gates (design P9):
  G1  median speed RMS <= 8 km/h
  G2  lap-time error: median |err| <= 0.5 s, p90 <= 1.5 s
  G3  parameters plausible: >= 90% inside the prior box, and the fitted ClA
      ranks the circuits the way the paddock does (Monaco/Hungary high,
      Monza/Baku low)
  G4  the fuel effect the engine was never told: +10 kg -> 0.0294 s/kg +-25%
  G5  a rear-wing click lowers top speed and speeds up the grip-limited corners

Usage
-----
    python -m pipeline.physics.calibrate_qss                   # top-3 drivers per session
    python -m pipeline.physics.calibrate_qss --drivers all
    python -m pipeline.physics.calibrate_qss --only 2024/bahrain_grand_prix/Q
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.physics import qss as Q, qss_fit as F
from pipeline.physics.modifiers import baseline_fuel_kg

from pipeline.physics.qss_fit import (BOUNDS, X0, MAX_PHASE_M, default_car, fit_lap, lap_on_line,  # noqa: F401
                                      phase_align)

MEASURED_FUEL_S_PER_KG = 0.0294          # P3 physics-check #2, race laps
HIGH_DOWNFORCE = ("monaco_grand_prix", "hungarian_grand_prix", "singapore_grand_prix")
LOW_DOWNFORCE = ("italian_grand_prix", "azerbaijan_grand_prix", "las_vegas_grand_prix")


def segment_time_errors(line: pd.DataFrame, v_real_kph: np.ndarray, prof: Q.Profile,
                        segments: list[dict]) -> list[dict]:
    """Simulated minus real time inside every segment of the baseline store.

    This is the quantity the simulator actually outputs (segment deltas), so
    it is the fairer accuracy figure: a braking point 10 m late shows up as a
    30 km/h speed error but only a few hundredths of a second.
    """
    d = line["distance_m"].to_numpy(float)
    ds = float(d[1] - d[0])
    t_real = ds / (np.maximum(v_real_kph, 5.0) / 3.6)
    t_sim = ds / (np.maximum(prof.speed_kph[:len(d)], 5.0) / 3.6)
    out = []
    for sg in segments:
        a, b = float(sg["start_m"]), float(sg["end_m"])
        m = (d >= a) & (d < b) if b >= a else (d >= a) | (d < b)
        if m.sum() < 2:
            continue
        out.append({"index": sg["index"], "kind": sg["kind"], "t_real_s": float(t_real[m].sum()),
                    "t_sim_s": float(t_sim[m].sum())})
    return out


def physics_probes(line: pd.DataFrame, drs: np.ndarray, car: Q.Car, prof: Q.Profile,
                   segments: list[dict]) -> dict:
    """What the calibrated car says about things it was never fitted to."""
    d, k, g = (line[c].to_numpy(float) for c in ("distance_m", "curvature_1pm", "grade"))
    fuel = Q.solve(d, k, g, drs, car.with_setup(fuel_delta_kg=10.0))
    wing = Q.solve(d, k, g, drs, car.with_setup(downforce_pct=10.0, drag_pct=8.0))
    # grip-limited corners: segments whose minimum speed sits on the corner limit
    faster, checked = 0, 0
    for sg in segments:
        if sg["kind"] not in ("low_speed_corner", "medium_speed_corner", "high_speed_corner"):
            continue
        a, b = float(sg["start_m"]), float(sg["end_m"])
        m = (prof.distance_m >= a) & (prof.distance_m <= b) if b >= a else (prof.distance_m >= a) | (prof.distance_m <= b)
        if not m.any():
            continue
        i = np.flatnonzero(m)[np.argmin(prof.speed_ms[m])]
        if prof.mode[i] != 0:                       # not on the grip limit -> power/brake shaped
            continue
        checked += 1
        faster += int(wing.speed_ms[i] > prof.speed_ms[i])
    return {"fuel_s_per_kg": (fuel.lap_time_s - prof.lap_time_s) / 10.0,
            "wing_vmax_delta_kph": float(wing.speed_kph.max() - prof.speed_kph.max()),
            "wing_lap_delta_s": wing.lap_time_s - prof.lap_time_s,
            "grip_corners_checked": checked, "grip_corners_faster": faster}


def race_total_laps(doc: dict) -> int | None:
    mx = 0
    for drv in doc["drivers"].values():
        for lap in drv.get("available", []):
            mx = max(mx, int(lap.get("lap_number") or 0))
    return mx or None


def run(baselines: Path, silver: Path, out_dir: Path, drivers: str, only: set[str] | None,
        limit: int | None, verbose: bool, power_kw: float = F.POWER_KW, driven_share: float = F.DRIVEN_SHARE,
        brake_share: float = F.BRAKE_SHARE, dump: bool = False, quiet: bool = False,
        load_sens: float = 0.0) -> int:
    docs = sorted(baselines.rglob("*.json"))
    rows, skipped, seg_rows = [], [], []
    t_start = time.perf_counter()
    n_done = 0
    for p in docs:
        season, event, session = p.parts[-3], p.parts[-2], p.stem
        key = f"{season}/{event}/{session}"
        if only and key not in only and f"{season}/{event}" not in only:
            continue
        line_p = silver / season / event / "line.parquet"
        if not line_p.exists():
            skipped.append({"session": key, "reason": "no line.parquet"})
            continue
        doc = json.loads(p.read_text())
        line = pd.read_parquet(line_p)
        meta = json.loads((silver / season / event / "line.json").read_text())
        temps = doc.get("session_temps") or {}
        total_laps = race_total_laps(doc) if session == "R" else None

        cands = []
        for code, drv in doc["drivers"].items():
            lap = drv["laps"].get("representative")
            if not lap or "trace" not in lap or not lap.get("lap_time_s"):
                continue
            cands.append((float(lap["lap_time_s"]), code, lap))
        cands.sort()
        if drivers != "all":
            cands = cands[: int(drivers)]

        for real_t, code, lap in cands:
            try:
                v_real, drs, shift = lap_on_line(lap["trace"], line)
                fuel = baseline_fuel_kg(session, lap.get("lap_number"), total_laps)
                base = Q.Car(mass_kg=798.0 + fuel, rho=Q.air_density(lap.get("air_temp_c") or temps.get("air_temp_c")),
                             power_kw=power_kw, driven_axle_share=driven_share, brake_share=brake_share,
                             load_sensitivity=load_sens)
                fit = fit_lap(line, v_real, drs, base)
                if dump:
                    prof = fit["profile"]
                    nn = len(line)
                    pd.DataFrame({"distance_m": line["distance_m"], "real_kph": np.round(v_real, 1),
                                  "sim_kph": np.round(prof.speed_kph[:nn], 1), "vlim_kph": np.round(prof.v_limit_ms[:nn] * 3.6, 1),
                                  "mode": prof.mode[:nn], "curvature_1pm": line["curvature_1pm"], "grade": line["grade"],
                                  "drs": drs.astype(int)}).to_csv(out_dir / f"debug_{lap['lap_uid']}.csv", index=False)
                probes = physics_probes(line, drs, fit["car"], fit["profile"], doc["segments"])
                seg_err = segment_time_errors(line, v_real, fit["profile"], doc["segments"])
                for e in seg_err:
                    seg_rows.append({"season": int(season), "event": event, "session": session, "driver": code,
                                     "lap_uid": lap["lap_uid"], **e, "err_s": e["t_sim_s"] - e["t_real_s"]})
                rows.append({
                    "season": int(season), "event": event, "session": session, "driver": code,
                    "lap_uid": lap["lap_uid"], "compound": lap.get("compound"), "condition": lap.get("condition"),
                    "real_lap_time_s": real_t, "sim_lap_time_s": round(fit["sim_lap_time_s"], 3),
                    "lap_time_err_s": round(fit["sim_lap_time_s"] - real_t, 3),
                    "rms_kph": round(fit["rms_kph"], 2), "bias_kph": round(fit["bias_kph"], 2),
                    "max_abs_kph": round(fit["max_abs_kph"], 1),
                    "mu": round(fit["mu"], 4), "cl_a": round(fit["cl_a"], 4), "cd_a": round(fit["cd_a"], 4),
                    "at_bound": fit["at_bound"], "fuel_kg": round(fuel, 1), "rho": round(base.rho, 4),
                    "phase_shift_m": shift * float(line["distance_m"].iloc[1] - line["distance_m"].iloc[0]),
                    "elevation_used": bool(meta["elevation"].get("used")),
                    "n_evals": fit["n_evals"], "fit_ms": fit["fit_ms"], **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in probes.items()},
                })
                n_done += 1
                if verbose:
                    r = rows[-1]
                    print(f"  {key:<40s} {code}  rms {r['rms_kph']:5.1f}  dt {r['lap_time_err_s']:+6.2f}  "
                          f"mu {r['mu']:.2f} ClA {r['cl_a']:.2f} CdA {r['cd_a']:.2f}  {r['fit_ms']:.0f} ms")
            except Exception as exc:                                  # noqa: BLE001
                skipped.append({"session": key, "driver": code, "reason": f"{type(exc).__name__}: {exc}"[:200]})
        if not verbose and not quiet:
            print(f"{key:<40s} {len(cands)} lap(s)   elapsed {time.perf_counter() - t_start:6.0f} s", flush=True)
        if limit and n_done >= limit:
            break

    if not rows:
        print("nothing calibrated", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "calibration.parquet", index=False)
    df.to_csv(out_dir / "calibration.csv", index=False)
    seg = pd.DataFrame(seg_rows)
    seg.to_parquet(out_dir / "calibration_segments.parquet", index=False)
    seg_abs = seg["err_s"].abs()
    seg_by_kind = seg.groupby("kind")["err_s"].agg(median_abs=lambda e: float(e.abs().median()),
                                                    bias=lambda e: float(e.median()), n="size")

    # ---------------------------------------------------------------- gates
    ok_box = (df["mu"].between(*BOUNDS["mu"], inclusive="neither") & df["cl_a"].between(*BOUNDS["cl_a"], inclusive="neither")
              & df["cd_a"].between(*BOUNDS["cd_a"], inclusive="neither"))
    by_circuit = df.groupby("event")["cl_a"].median().sort_values(ascending=False)
    ranks = {e: i for i, e in enumerate(by_circuit.index)}
    n_c = len(by_circuit)
    hi_ok = [ranks[e] < n_c / 2 for e in HIGH_DOWNFORCE if e in ranks]
    lo_ok = [ranks[e] >= n_c / 2 for e in LOW_DOWNFORCE if e in ranks]
    dry = df[df["condition"].fillna("dry") == "dry"]
    fuel_med = float(dry["fuel_s_per_kg"].median())
    gates = {
        "G1_speed_rms": {"median_kph": round(float(df["rms_kph"].median()), 2), "p90_kph": round(float(df["rms_kph"].quantile(0.9)), 2),
                         "limit_median_kph": 8.0, "passes": bool(df["rms_kph"].median() <= 8.0)},
        "G1b_segment_time": {"median_abs_s": round(float(seg_abs.median()), 3), "p90_abs_s": round(float(seg_abs.quantile(0.9)), 3),
                             "median_abs_rel": round(float((seg["err_s"] / seg["t_real_s"]).abs().median()), 4),
                             "bias_s": round(float(seg["err_s"].median()), 4), "segments": int(len(seg)),
                             "by_kind": {k: {"median_abs_s": round(float(r["median_abs"]), 3), "bias_s": round(float(r["bias"]), 3), "n": int(r["n"])}
                                         for k, r in seg_by_kind.iterrows()},
                             "limit_median_abs_s": 0.10, "passes": bool(seg_abs.median() <= 0.10)},
        "G2_lap_time": {"median_abs_s": round(float(df["lap_time_err_s"].abs().median()), 3),
                        "p90_abs_s": round(float(df["lap_time_err_s"].abs().quantile(0.9)), 3),
                        "bias_s": round(float(df["lap_time_err_s"].median()), 3),
                        "passes": bool(df["lap_time_err_s"].abs().median() <= 0.5 and df["lap_time_err_s"].abs().quantile(0.9) <= 1.5)},
        "G3_parameters": {"inside_prior_box": round(float(ok_box.mean()), 3), "at_bound_share": round(float(df["at_bound"].mean()), 3),
                          "mu_median": round(float(df["mu"].median()), 3), "cl_a_median": round(float(df["cl_a"].median()), 3),
                          "cd_a_median": round(float(df["cd_a"].median()), 3),
                          "cl_a_rank_top": list(by_circuit.index[:5]), "cl_a_rank_bottom": list(by_circuit.index[-5:]),
                          "high_downforce_in_top_half": hi_ok, "low_downforce_in_bottom_half": lo_ok,
                          "passes": bool(ok_box.mean() >= 0.9 and all(hi_ok) and all(lo_ok))},
        "G4_fuel_effect": {"median_s_per_kg": round(fuel_med, 5), "p10": round(float(dry["fuel_s_per_kg"].quantile(0.1)), 5),
                           "p90": round(float(dry["fuel_s_per_kg"].quantile(0.9)), 5),
                           "measured_s_per_kg": MEASURED_FUEL_S_PER_KG,
                           "passes": bool(abs(fuel_med - MEASURED_FUEL_S_PER_KG) <= 0.25 * MEASURED_FUEL_S_PER_KG)},
        "G5_wing_signs": {"vmax_drops_share": round(float((df["wing_vmax_delta_kph"] < 0).mean()), 3),
                          "grip_corners_faster_share": round(float(df["grip_corners_faster"].sum() / max(df["grip_corners_checked"].sum(), 1)), 3),
                          "passes": bool((df["wing_vmax_delta_kph"] < 0).mean() >= 0.95
                                         and df["grip_corners_faster"].sum() / max(df["grip_corners_checked"].sum(), 1) >= 0.95)},
    }
    by_session = df.groupby("session")[["rms_kph", "lap_time_err_s"]].agg(["median", "count"])
    report = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "laps": int(len(df)), "sessions": int(df.groupby(["season", "event", "session"]).ngroups),
              "circuits": int(df["event"].nunique()), "drivers_per_session": drivers,
              "elevation_used_share": round(float(df["elevation_used"].mean()), 3),
              "fit_ms_median": round(float(df["fit_ms"].median()), 1),
              "gates": gates, "gates_passed": sum(g["passes"] for g in gates.values()),
              "by_session": {s: {"rms_kph_median": round(float(by_session.loc[s, ("rms_kph", "median")]), 2),
                                 "lap_time_err_median_s": round(float(by_session.loc[s, ("lap_time_err_s", "median")]), 3),
                                 "laps": int(by_session.loc[s, ("rms_kph", "count")])} for s in by_session.index},
              "cl_a_by_circuit": {e: round(float(v), 3) for e, v in by_circuit.items()},
              "skipped": skipped}
    (out_dir / "calibration_report.json").write_text(json.dumps(report, indent=1))

    if quiet:
        print(f"P={power_kw:.0f} kW brake={brake_share:.2f} driven={driven_share:.2f} ls={load_sens:.2f} | rms {df['rms_kph'].median():5.1f}  "
              f"bias {df['bias_kph'].median():+5.1f} km/h  dt {df['lap_time_err_s'].median():+5.2f} s  "
              f"CdA {df['cd_a'].median():.2f} (at bound {df['at_bound'].mean():.0%})  mu {df['mu'].median():.2f}  ClA {df['cl_a'].median():.2f}  "
              f"fuel {fuel_med:.4f} s/kg")
        return 0
    print()
    print(f"QSS CALIBRATION  {len(df)} laps / {report['sessions']} sessions / {report['circuits']} circuits  "
          f"(median fit {report['fit_ms_median']:.0f} ms, elevation used on {report['elevation_used_share']:.0%})")
    for name, g in gates.items():
        flag = "PASS" if g["passes"] else "FAIL"
        detail = {k: v for k, v in g.items() if k not in ("passes",) and not isinstance(v, (list, dict))}
        print(f"  [{flag}] {name:<16s} " + "  ".join(f"{k}={v}" for k, v in detail.items()))
    for k, r in seg_by_kind.iterrows():
        print(f"      {k:<20s} median |dt| {r['median_abs']:.3f} s  bias {r['bias']:+.3f} s  (n={int(r['n'])})")
    print(f"  ClA top   : {', '.join(by_circuit.index[:5])}")
    print(f"  ClA bottom: {', '.join(by_circuit.index[-5:])}")
    for s, v in report["by_session"].items():
        print(f"  {s}: rms {v['rms_kph_median']:.1f} km/h, lap-time err {v['lap_time_err_median_s']:+.2f} s  ({v['laps']} laps)")
    if skipped:
        print(f"  skipped {len(skipped)} (see report)")
    print(f"  gates passed {report['gates_passed']} / {len(gates)}  ->  {out_dir / 'calibration_report.json'}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", type=Path, default=Path("../data/artifacts/baselines"))
    ap.add_argument("--silver", type=Path, default=Path("../data/silver"))
    ap.add_argument("--out", type=Path, default=Path("../data/artifacts/qss"))
    ap.add_argument("--drivers", default="3", help="laps per session: N fastest representative laps, or 'all'")
    ap.add_argument("--only", default=None, help="comma-separated season/event[/session]")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many laps")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--power", type=float, default=F.POWER_KW, help="wheel power, kW (fixed, not fitted)")
    ap.add_argument("--driven", type=float, default=F.DRIVEN_SHARE, help="driven-axle share of grip for traction (fixed)")
    ap.add_argument("--brake", type=float, default=F.BRAKE_SHARE, help="brake share of the grip limit (fixed)")
    ap.add_argument("--loadsens", type=float, default=0.0, help="tyre load sensitivity exponent (fixed)")
    ap.add_argument("--dump", action="store_true", help="write debug_<lap>.csv with real vs simulated speed")
    ap.add_argument("--sweep", action="store_true",
                    help="grid over power x driven share on the --only set; prints one line per combination")
    a = ap.parse_args()
    only = {x.strip() for x in a.only.split(",")} if a.only else None
    if a.sweep:
        for pw in (600.0, 540.0, 480.0):
            for br in (1.0, 0.8):
                run(a.baselines, a.silver, a.out / "sweep", a.drivers, only, a.limit, False,
                    power_kw=pw, driven_share=a.driven, brake_share=br, quiet=True, load_sens=a.loadsens)
        sys.exit(0)
    sys.exit(run(a.baselines, a.silver, a.out, a.drivers, only, a.limit, a.verbose,
                 power_kw=a.power, driven_share=a.driven, brake_share=a.brake, dump=a.dump,
                 load_sens=a.loadsens))


if __name__ == "__main__":
    main()
