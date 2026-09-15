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
    pooled = []
    for (season, ev), g in laps.groupby(["season", "event_slug"]):
        per_chassis = []
        for ch, gg in g.groupby("chassis"):
            best = gg.nsmallest(3, "lap_time_s")
            per_chassis.append((ch, best["trap_kph"].median(), best["hs_corner_kph"].median(),
                                best["lap_time_s"].median()))
        if len(per_chassis) < 5:
            continue
        pc = pd.DataFrame(per_chassis, columns=["chassis", "trap", "hs", "lap"])
        x, y = pc["trap"].to_numpy(float), pc["hs"].to_numpy(float)
        slope = float(np.polyfit(x, y, 1)[0])
        corr = float(np.corrcoef(x, y)[0, 1])
        # The raw slope is confounded by CAR QUALITY: a good car is faster
        # everywhere, so trap and corner speed rise together. Hold overall
        # pace (best lap time) fixed and ask again: among equally fast cars,
        # does the higher trap speed come with slower fast corners?
        z = lambda v: (v - v.mean()) / (v.std(ddof=0) + 1e-9)
        pooled.append(pd.DataFrame({"hs": z(pc["hs"]), "trap": z(pc["trap"]), "lap": z(pc["lap"])}))
        Xp = np.column_stack([np.ones(len(pc)), z(pc["trap"]), z(pc["lap"])])
        bp, *_ = np.linalg.lstsq(Xp, z(pc["hs"]).to_numpy(float), rcond=None)
        partial = float(bp[1])
        # implied drag-per-downforce for a wing change, from the two speed laws:
        #   dv_c/v_c = 0.5*alpha*d_df   ;   dv_t/v_t = -(1/3)*d_drag
        v0 = cfg.coeff("car", "aero_crossover_kph").value
        vc, vt = float(np.median(y)), float(np.median(x))
        alpha = vc * vc / (vc * vc + v0 * v0)
        ratio = -1.5 * alpha / (slope * vt / vc) if slope != 0 else np.nan
        rows.append({"season": season, "event": ev, "chassis": len(pc), "trap_kph": round(vt, 1),
                     "hs_corner_kph": round(vc, 1), "slope": round(slope, 3), "corr": round(corr, 2),
                     "partial_slope_z": round(partial, 2), "drag_per_df": round(ratio, 2)})
    out = pd.DataFrame(rows)
    if pooled:
        allp = pd.concat(pooled)
        X = np.column_stack([np.ones(len(allp)), allp["trap"], allp["lap"]])
        b, *_ = np.linalg.lstsq(X, allp["hs"].to_numpy(float), rcond=None)
        out.attrs["pooled"] = {"n": int(len(allp)), "trap_given_pace": round(float(b[1]), 3),
                               "pace": round(float(b[2]), 3)}
    return out


# ------------------------------------------------------------ check 2: fuel
def _fuel_design(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """lap_time ~ driver FE + compound FE + lap_number + tyre_life x compound.

    Within ONE driver, lap_number and tyre_life differ by a constant per
    stint, so with a stint (or compound) dummy the three are exactly
    collinear and least squares returns garbage — the first version of this
    check did exactly that and printed -0.8 s/kg. Pooling the field fixes
    it: drivers pit on different laps, so the same compound sits at different
    (lap_number - tyre_life) offsets and fuel separates from tyre age.
    """
    drivers = sorted(g["driver"].unique())
    comps = sorted(g["compound"].unique())
    cols = [np.ones(len(g))]
    for d in drivers[1:]:
        cols.append((g["driver"] == d).to_numpy(float))
    for c in comps[1:]:
        cols.append((g["compound"] == c).to_numpy(float))
    cols.append(g["lap_number"].to_numpy(float))                     # <- fuel
    idx_lap = len(cols) - 1
    for c in comps:
        cols.append(g["tyre_life"].to_numpy(float) * (g["compound"] == c).to_numpy(float))
    return np.column_stack(cols), g["lap_time_s"].to_numpy(float), idx_lap


def fuel_slope(df: pd.DataFrame, n_boot: int = 60, seed: int = 7) -> pd.DataFrame:
    cfg = default_config()
    full = float(cfg.raw["car"]["fuel_max_kg"])
    r = df[(df["session"] == "R") & (df["condition"] == "dry")]
    laps = _lap_table(r).reset_index()
    if "lap_effort_class" in laps.columns:
        laps = laps[laps["lap_effort_class"].isin(["push", "moderate"])]
    laps = laps.dropna(subset=["lap_time_s", "lap_number", "tyre_life", "compound", "driver"])
    laps = laps[laps["lap_number"] > 2]
    rng = np.random.default_rng(seed)

    rows = []
    for (season, ev), g in laps.groupby(["season", "event_slug"]):
        total = int(g["lap_number"].max())
        kg_per_lap = full / total
        # trim each driver's slowest 10% (traffic, damage, a botched lap)
        g = g[g["lap_time_s"] <= g.groupby("driver")["lap_time_s"].transform(lambda s: s.quantile(0.90))]
        drivers = sorted(g["driver"].unique())
        if len(drivers) < 5 or len(g) < 100:
            continue
        X, y, i = _fuel_design(g)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        point = -beta[i] / kg_per_lap
        boots = []
        for _ in range(n_boot):                       # resample DRIVERS, keep their laps whole
            pick = rng.choice(drivers, size=len(drivers), replace=True)
            gb = pd.concat([g[g["driver"] == d].assign(driver=f"{d}#{k}") for k, d in enumerate(pick)])
            Xb, yb, ib = _fuel_design(gb)
            bb, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
            boots.append(-bb[ib] / kg_per_lap)
        arr = np.asarray(boots)
        lo, hi = np.percentile(arr, [25, 75])
        # When resampling the field barely moves the number, the slope is a
        # property of the race. When it swings by an order of magnitude the
        # race had no steady rhythm (safety cars, red flags, formation
        # running) and the number means nothing.
        rows.append({"season": season, "event": ev, "laps": total, "drivers": len(drivers),
                     "n_laps": int(len(g)), "kg_per_lap": round(kg_per_lap, 2),
                     "s_per_kg": round(float(point), 4),
                     "boot_iqr": round(float(hi - lo), 4),
                     "trusted": bool((hi - lo) < 0.02 and 0 < point < 0.2)})
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
        print(f"   raw slope: {int((t1['slope'] < 0).sum())}/{len(t1)} events negative — confounded by car quality"
              f" (a good car is fast everywhere)")
        po = t1.attrs.get("pooled")
        if po:
            print(f"   PARTIAL (pace held fixed, pooled {po['n']} chassis-events, z-units):"
                  f"  trap -> hs corner {po['trap_given_pace']:+.3f}   pace -> hs corner {po['pace']:+.3f}")
            print(f"   {int((t1['partial_slope_z'] < 0).sum())}/{len(t1)} events negative once pace is held fixed."
                  f"  Expected: negative. If it is not, the wing ratio cannot be checked with public data")
            print(f"   and stays what it is in physics.yaml: rear-wing drag/downforce "
                  f"{rw['drag_pct']['value'] / rw['downforce_pct']['value']:.2f} (literature).")
    else:
        print("   (no qualifying session had 5+ chassis with usable laps)")

    print("\n#2 FUEL SLOPE  (races, dry, push+moderate laps; field-pooled: driver FE + compound FE + lap_number + tyre_life x compound)")
    t2 = fuel_slope(df)
    if len(t2):
        print(t2.to_string(index=False))
        cf = cfg.coeff("fuel", "s_per_kg_per_lap")
        tr = t2[t2["trusted"]]
        print(f"   configured {cf.value:.3f} s/kg (u +-{cf.u:.0%})   measured median over the "
              f"{len(tr)} stable race(s): "
              + (f"{tr['s_per_kg'].median():.4f} s/kg" if len(tr) else "n/a"))
        print(f"   untrusted: {len(t2) - len(tr)} race(s) with bootstrap IQR >= 0.02 s/kg or a negative slope — "
              f"no steady rhythm to measure (SC / red flag / formation running); the ML must not")
        print(f"   read a lap-number trend into those either -> P4 note: a 'laps since restart' feature or race-level gate.")
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
