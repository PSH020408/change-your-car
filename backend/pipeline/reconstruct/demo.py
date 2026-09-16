"""P5 smoke test on REAL telemetry — the toy lap proves the maths, this
proves the assumptions about the data.

    make reconstruct-demo
    python -m pipeline.reconstruct.demo --season 2024 --event bahrain_grand_prix --session Q --driver VER

Three checks, each a line:
  1. the baseline lap's integrated time vs its official lap time (our
     distance/speed axes are consistent?)
  2. a zero delta returns the baseline: no clamps, pedals agree
  3. a physics delta (rear wing 0.85) becomes a trace whose per-segment
     times match the request, with any physics refusals listed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.physics import modifiers as M, segment_delta as D
from pipeline.reconstruct import trace as T


def representative_lap(laps: pd.DataFrame, driver: str) -> pd.Series:
    """The driver's median clean push lap (the HUD's default baseline).

    Relaxes its filters one by one and says so, rather than failing: a
    qualifying session has only a handful of laps per driver.
    """
    d = laps[laps["driver"].astype(str) == driver].dropna(subset=["lap_time_s"])
    print(f"laps for {driver}: {len(d)} of {len(laps)} in the session "
          f"(drivers: {', '.join(sorted(laps['driver'].astype(str).unique())[:8])} ...)")
    if not len(d):
        alt = laps["driver"].astype(str).value_counts().index[0]
        print(f"  -> {driver} has no laps here; using {alt} instead")
        return representative_lap(laps, alt)
    steps = []
    if "telemetry_quality" in d:
        q = d[d["telemetry_quality"].isin(["clean", "normal"])]
        steps.append(("telemetry clean/normal", q))
        d2 = q if len(q) else d
    else:
        d2 = d
    if "gap_ahead_s" in d2 and str(d2["session"].iloc[0]) == "R":     # race laps only, as in P4
        g = pd.to_numeric(d2["gap_ahead_s"], errors="coerce")
        c = d2[g.isna() | (g >= 2.5)]
        steps.append(("clean air >= 2.5 s", c))
        d3 = c if len(c) else d2
    else:
        d3 = d2
        if "gap_ahead_s" in d2:
            g = pd.to_numeric(d2["gap_ahead_s"], errors="coerce")
            print(f"  gap ahead at the line   : median {g.median():.1f} s (qualifying: not filtered)")
    for name, kept in steps:
        print(f"  {name:<24}: {len(kept)} laps" + ("  (relaxed: none left)" if not len(kept) else ""))
    med = d3["lap_time_s"].median()
    return d3.iloc[(d3["lap_time_s"] - med).abs().argsort().iloc[0]]


def segment_table(tel: pd.DataFrame, segments: list[dict]) -> pd.DataFrame:
    """What lap_physics_delta needs, straight from the baseline telemetry."""
    d = tel["distance_m"].to_numpy(float); v = tel["speed_kph"].to_numpy(float)
    rows = []
    for (i0, i1), s in zip(T._segment_slices(d, segments), segments):
        vv = v[i0:i1]
        rows.append({"segment_index": s["index"], "segment_kind": s["kind"],
                     "segment_time_s": T.integrate_lap_time(vv, d[i0:i1]) if i1 - i0 >= 3 else 0.0,
                     "speed_min_kph": float(vv.min()) if len(vv) else np.nan,
                     "speed_mean_kph": float(vv.mean()) if len(vv) else np.nan,
                     "speed_max_kph": float(vv.max()) if len(vv) else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--event", default="bahrain_grand_prix")
    ap.add_argument("--session", default="Q")
    ap.add_argument("--driver", default="VER")
    ap.add_argument("--lake", type=Path, default=Path("../data"))
    a = ap.parse_args()

    bdir = a.lake / "bronze" / str(a.season) / a.event / a.session
    laps = pd.read_parquet(bdir / "laps.parquet")
    tel = pd.read_parquet(bdir / "telemetry.parquet")
    track = json.loads((a.lake / "silver" / str(a.season) / a.event / "track.json").read_text())
    segments = track["segments"]
    lap = representative_lap(laps, a.driver)
    base = tel[tel["lap_uid"] == lap["lap_uid"]].sort_values("distance_m").reset_index(drop=True)
    # the same per-lap distance rescale the feature store applies
    lap_len = float(track["geometry"]["lap_length_m"])
    base = base.copy(); base["distance_m"] = base["distance_m"] * lap_len / float(base["distance_m"].max())
    print(f"baseline  : {lap['lap_uid']}  {lap['driver']}  {lap['lap_time_s']:.3f} s  "
          f"{len(base)} samples  compound {lap.get('compound')}  tyre_life {lap.get('tyre_life')}")

    # 1 — integration vs official lap time
    t_int = T.integrate_lap_time(base["speed_kph"].to_numpy(float), base["distance_m"].to_numpy(float))
    print(f"1 integrate: {t_int:.3f} s vs official {lap['lap_time_s']:.3f} s  "
          f"(diff {t_int - lap['lap_time_s']:+.3f} s, {100 * (t_int / lap['lap_time_s'] - 1):+.2f}%  "
          f"= the two partial 240 ms intervals at the line; the HUD time axis is scaled by official/integrated)")

    # 2 — zero delta
    r0 = T.reconstruct(base, segments, [0.0] * len(segments), official_lap_time_s=float(lap["lap_time_s"]))
    agree = float(np.mean(r0.trace["brake_on"].to_numpy() == base["brake_on"].to_numpy().astype(bool)))
    thr_mae = float(np.mean(np.abs(r0.trace["throttle_pct"].to_numpy() - pd.to_numeric(base["throttle_pct"], errors="coerce").fillna(0).to_numpy())))
    tiny = int((r0.segments["baseline_s"] == 0).sum())
    print(f"2 zero delta: achieved {r0.achieved_delta_s:+.4f} s  clamps {r0.clamps}  "
          f"brake agreement {agree:.1%}  throttle MAE {thr_mae:.1f} pts  "
          f"DRS open {int(r0.trace['drs_open'].sum())} samples  "
          f"({tiny} segments shorter than 3 samples pass through unwarped; their deltas are ~0)")

    # 3 — a physics delta: rear wing 0.85, everything else baseline
    segs_df = segment_table(base, segments)
    phys = D.lap_physics_delta(segs_df, M.SetupInput(rear_wing=0.85), session=a.session)
    deltas = dict(zip(phys["segment_index"].astype(int), phys["physics_delta_s"].astype(float)))
    r = T.reconstruct(base, segments, deltas, official_lap_time_s=float(lap["lap_time_s"]))
    print(f"3 rear wing 0.85: requested {r.requested_delta_s:+.3f} s  achieved {r.achieved_delta_s:+.3f} s  "
          f"integration error {r.integration_error_s:.4f} s  clamps {r.clamps}")
    rep = r.segments.copy()
    rep["kind"] = rep["kind"].str.replace("_speed_corner", "").str.replace("straight", "str")
    print("  " + rep[["segment_index", "kind", "baseline_s", "requested_s", "achieved_s", "residual_s", "clamped"]]
          .round(3).to_string(index=False).replace("\n", "\n  "))
    worst = rep.loc[rep["residual_s"].abs().idxmax()]
    print(f"  largest residual: segment {int(worst['segment_index'])} ({worst['kind']}) {worst['residual_s']:+.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
