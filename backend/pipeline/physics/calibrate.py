"""P3 — physics-check: do the grade-B coefficients agree with our own data?

Three checks, each a terminal table. None of them FITS the simulator's
coefficients to data — they measure the same quantity independently and
print it next to the configured value, so a wrong order of magnitude cannot
hide. (Fitting them would make the physics layer a second ML model trained
on the same rows the first one uses.)

  #1  aero efficiency line   across chassis, in one qualifying: more trap
      speed <-> less high-speed-corner speed. Sign and slope.
  #2  fuel slope             within a race, lap time vs lap number with tyre
      age held: seconds per kilogram of fuel.
  #3  corner grip fit        v_min^2 vs radius over every corner: effective
      friction mu and the aero cross-over speed v0.

    python -m pipeline.physics.calibrate --gold ../data/gold/features.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.physics.modifiers import default_config

G = 9.81


def _lap_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per lap with the lap-level columns (gold is one row per segment)."""
    lap_cols = [c for c in ("lap_uid", "season", "event", "event_slug", "session", "driver", "team",
                            "chassis", "compound", "tyre_life", "lap_number", "stint", "lap_time_s",
                            "track_temp_c", "condition", "lap_effort_class", "speed_st_kph",
                            "speed_i2_kph", "speed_fl_kph") if c in df.columns]
    return df.drop_duplicates("lap_uid")[lap_cols].set_index("lap_uid")


# ------------------------------------------------------------ check 1: aero
def aero_efficiency_line(df: pd.DataFrame) -> pd.DataFrame:
    cfg = default_config()
    q = df[(df["session"] == "Q") & (df["condition"] == "dry")]
    q = q[q["lap_effort_class"].isin(["push"])] if "lap_effort_class" in q.columns else q
    laps = _lap_table(q)

    straights = q[q["segment_kind"] == "straight"].groupby("lap_uid")["speed_max_kph"].max()
    hs_kind = "high_speed_corner"
    hs = q[q["segment_kind"] == hs_kind].groupby("lap_uid")["speed_min_kph"].mean()
    laps = laps.join(straights.rename("trap_kph")).join(hs.rename("hs_corner_kph"))
    if "speed_st_kph" in laps.columns:
        laps["trap_kph"] = laps["speed_st_kph"].fillna(laps["trap_kph"])
    laps = laps.dropna(subset=["trap_kph", "hs_corner_kph", "lap_time_s"])

    rows = []
    for (season, ev), g in laps.groupby(["season", "event_slug"]):
        per_chassis = []
        for ch, gg in g.groupby("chassis"):
            best = gg.nsmallest(3, "lap_time_s")
            per_chassis.append((ch, best["trap_kph"].median(), best["hs_corner_kph"].median()))
        if len(per_chassis) < 5:
            continue
        pc = pd.DataFrame(per_chassis, columns=["chassis", "trap", "hs"])
        x, y = pc["trap"].to_numpy(float), pc["hs"].to_numpy(float)
        slope = float(np.polyfit(x, y, 1)[0])
        corr = float(np.corrcoef(x, y)[0, 1])
        # implied drag-per-downforce for a wing change, from the two speed laws:
        #   dv_c/v_c = 0.5*alpha*d_df   ;   dv_t/v_t = -(1/3)*d_drag
        v0 = cfg.coeff("car", "aero_crossover_kph").value
        vc, vt = float(np.median(y)), float(np.median(x))
        alpha = vc * vc / (vc * vc + v0 * v0)
        ratio = -1.5 * alpha / (slope * vt / vc) if slope != 0 else np.nan
        rows.append({"season": season, "event": ev, "chassis": len(pc), "trap_kph": round(vt, 1),
                     "hs_corner_kph": round(vc, 1), "slope": round(slope, 3), "corr": round(corr, 2),
                     "drag_per_df": round(ratio, 2)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ check 2: fuel
def fuel_slope(df: pd.DataFrame) -> pd.DataFrame:
    cfg = default_config()
    full = float(cfg.raw["car"]["fuel_max_kg"])
    r = df[(df["session"] == "R") & (df["condition"] == "dry")]
    laps = _lap_table(r).reset_index()
    if "lap_effort_class" in laps.columns:
        laps = laps[laps["lap_effort_class"].isin(["push", "moderate"])]
    laps = laps.dropna(subset=["lap_time_s", "lap_number", "tyre_life", "compound"])
    laps = laps[laps["lap_number"] > 2]

    rows = []
    for (season, ev), g in laps.groupby(["season", "event_slug"]):
        total = int(g["lap_number"].max())
        kg_per_lap = full / total
        per_driver = []
        for drv, gd in g.groupby("driver"):
            if gd["stint"].nunique() < 2 or len(gd) < 15:
                continue
            comps = sorted(gd["compound"].unique())
            X = [np.ones(len(gd)), gd["lap_number"].to_numpy(float), gd["tyre_life"].to_numpy(float)]
            for c in comps[1:]:
                X.append((gd["compound"] == c).to_numpy(float))
            X = np.column_stack(X)
            y = gd["lap_time_s"].to_numpy(float)
            # trim the slowest 10% (traffic, damage) — robust enough for a slope
            keep = y <= np.quantile(y, 0.90)
            beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
            per_driver.append(-beta[1] / kg_per_lap)           # s per kg
        if len(per_driver) < 3:
            continue
        arr = np.asarray(per_driver)
        rows.append({"season": season, "event": ev, "laps": total, "drivers": len(arr),
                     "kg_per_lap": round(kg_per_lap, 2),
                     "s_per_kg_median": round(float(np.median(arr)), 4),
                     "s_per_kg_iqr": round(float(np.subtract(*np.percentile(arr, [75, 25]))), 4)})
    return pd.DataFrame(rows)


# ----------------------------------------------------- check 3: corner grip
def corner_grip_fit(df: pd.DataFrame) -> dict:
    q = df[(df["session"] == "Q") & (df["condition"] == "dry")]
    if "lap_effort_class" in q.columns:
        q = q[q["lap_effort_class"] == "push"]
    c = q[q["segment_kind"].isin(["low_speed_corner", "medium_speed_corner", "high_speed_corner"])]
    c = c.dropna(subset=["speed_min_kph", "segment_min_radius_m"])
    c = c[(c["segment_min_radius_m"] > 10) & (c["segment_min_radius_m"] < 600) & (c["speed_min_kph"] > 40)]
    # one point per (circuit, segment): the fastest 10% of laps through it
    pts = (c.groupby(["season", "event_slug", "segment_index"])
             .agg(v=("speed_min_kph", lambda s: float(np.quantile(s, 0.90))),
                  R=("segment_min_radius_m", "median"), kind=("segment_kind", "first"))
             .reset_index())
    v = pts["v"].to_numpy(float) / 3.6
    R = pts["R"].to_numpy(float)

    def predict(mu: float, v0_kph: float) -> np.ndarray:
        v0 = v0_kph / 3.6
        base = mu * G * R
        denom = 1.0 - base / (v0 * v0)
        out = np.where(denom > 0.05, base / np.maximum(denom, 0.05), np.nan)
        return np.sqrt(out)

    best = None
    for mu in np.arange(1.0, 2.6, 0.05):
        for v0 in np.arange(100.0, 260.0, 5.0):
            p = predict(mu, v0)
            ok = np.isfinite(p)
            if ok.sum() < len(p) * 0.8:
                continue
            err = float(np.median(np.abs(np.log(p[ok] / v[ok]))))
            if best is None or err < best[0]:
                best = (err, mu, v0)
    err, mu, v0 = best
    p = predict(mu, v0)
    resid = np.log(p / v)
    by_kind = {k: round(float(np.nanmedian(resid[pts["kind"] == k])) * 100, 1)
               for k in ("low_speed_corner", "medium_speed_corner", "high_speed_corner")}
    return {"points": int(len(pts)), "mu": round(mu, 2), "v0_kph": round(v0, 0),
            "median_abs_err_pct": round(err * 100, 1), "resid_pct_by_kind": by_kind}


# ------------------------------------------------------------------ report
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="../data/gold/features.parquet")
    a = ap.parse_args()
    df = pd.read_parquet(a.gold)
    cfg = default_config()
    print(f"gold : {Path(a.gold).resolve()}  ({len(df):,} rows)\n")

    print("#1 AERO EFFICIENCY LINE  (qualifying, push laps, best 3 per chassis)")
    print("   expect: negative slope — the chassis with the higher trap speed takes the")
    print("   high-speed corners slower. Cross-team differences also carry engine and")
    print("   car efficiency, so this is a SIGN and ORDER-OF-MAGNITUDE check only.")
    t1 = aero_efficiency_line(df)
    if len(t1):
        print(t1.to_string(index=False))
        rw = cfg.raw["aero"]["rear_wing"]
        print(f"   configured rear-wing drag/downforce : {rw['drag_pct']['value'] / rw['downforce_pct']['value']:.2f}"
              f"   measured median : {t1['drag_per_df'].median():.2f}"
              f"   ({int((t1['slope'] < 0).sum())}/{len(t1)} events with the expected sign)")
    else:
        print("   (no qualifying session had 5+ chassis with usable laps)")

    print("\n#2 FUEL SLOPE  (races, dry, push+moderate laps, lap_time ~ lap_number + tyre_life + compound)")
    t2 = fuel_slope(df)
    if len(t2):
        print(t2.to_string(index=False))
        cf = cfg.coeff("fuel", "s_per_kg_per_lap")
        print(f"   configured {cf.value:.3f} s/kg (u +-{cf.u:.0%})   measured median "
              f"{t2['s_per_kg_median'].median():.4f} s/kg over {len(t2)} races")
    else:
        print("   (no race with 3+ drivers on 2+ stints)")

    print("\n#3 CORNER GRIP FIT  v_min^2 = mu*g*R / (1 - mu*g*R/v0^2)   (qualifying push laps, one point per corner)")
    t3 = corner_grip_fit(df)
    mu, v0 = cfg.coeff("car", "mu_mechanical"), cfg.coeff("car", "aero_crossover_kph")
    print(f"   points {t3['points']}   fitted mu {t3['mu']}  v0 {t3['v0_kph']:.0f} km/h   "
          f"median |err| {t3['median_abs_err_pct']}%")
    print(f"   configured mu {mu.value} (u +-{mu.u:.0%})   v0 {v0.value:.0f} km/h (u +-{v0.u:.0%})")
    print(f"   residual by kind (model - data, %): {t3['resid_pct_by_kind']}")
    print("   a positive residual means the model is too fast there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
