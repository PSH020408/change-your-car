"""P10-2 — what the physics engine knows about each segment, as ML features.

The P4 model's weakest segment kind is the low-speed corner (MAE 0.111 s vs
0.058 s on straights) and its only geometric features are length and minimum
radius. The calibrated point-mass solver (P9) knows more about a segment
before any lap is driven: how much of it is spent on the traction limit out
of the corner, on the brakes, on the grip limit, or flat-out; how steep it
is; where DRS opens. These are properties of the circuit and the weekend's
fitted car — not of the lap being predicted — so they are legitimate pre-lap
features, and they let the trees learn "tyre age hurts traction-limited
segments more" without a hand-written interaction.

Per (season, event): the fastest driver's representative qualifying lap is
fitted (mu, ClA, CdA), the profile is solved once, and every stored segment
gets its shares and statistics. Written to silver as segment_physics.parquet
and joined into gold by (season, event_slug, segment_index).

Usage
-----
    python -m pipeline.physics.segment_physics --scope configs/scope.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.ingest import paths
from pipeline.physics import qss as Q, qss_fit as F
from pipeline.physics.modifiers import baseline_fuel_kg
from pipeline.segment.run import silver_dir

MODES = {0: "grip", 1: "traction", 2: "power", 3: "brake"}


def segment_rows(line: pd.DataFrame, prof: Q.Profile, segments: list[dict]) -> list[dict]:
    d = line["distance_m"].to_numpy(float)
    n = len(d)
    grade = line["grade"].to_numpy(float)
    drs = line["drs_share"].to_numpy(float)
    v = prof.speed_kph[:n]
    mode = prof.mode[:n]
    step = float(d[1] - d[0])
    out = []
    for sg in segments:
        a, b = float(sg["start_m"]), float(sg["end_m"])
        m = (d >= a) & (d < b) if b >= a else (d >= a) | (d < b)
        if m.sum() < 2:
            continue
        g = grade[m]
        row = {"segment_index": int(sg["index"]),
               "segment_grade_mean": float(np.mean(g)),
               "segment_climb_m": float(np.sum(np.clip(g, 0, None)) * step),
               "segment_descent_m": float(-np.sum(np.clip(g, None, 0)) * step),
               "segment_drs_share": float(np.mean(drs[m])),
               "segment_sim_apex_kph": float(np.min(v[m])),
               "segment_sim_vmax_kph": float(np.max(v[m]))}
        for k, name in MODES.items():
            row[f"segment_{name}_share"] = float(np.mean(mode[m] == k))
        out.append(row)
    return out


def build_event(store_event_dir: Path) -> tuple[pd.DataFrame, dict] | None:
    q_path, line_path = store_event_dir / "Q.json", store_event_dir / "line.json"
    if not (q_path.exists() and line_path.exists()):
        return None
    doc = json.loads(q_path.read_text())
    line = F.load_store_line(line_path)
    cands = []
    for code, drv in doc["drivers"].items():
        lap = drv["laps"].get("representative")
        if lap and "trace" in lap and lap.get("lap_time_s"):
            cands.append((float(lap["lap_time_s"]), code, lap))
    if not cands:
        return None
    _, code, lap = min(cands)
    v_real, drs, _ = F.lap_on_line(lap["trace"], line)
    base = F.default_car(mass_kg=798.0 + baseline_fuel_kg("Q"), rho=Q.air_density(lap.get("air_temp_c")))
    fit = F.fit_lap(line, v_real, drs, base)
    rows = segment_rows(line, fit["profile"], doc["segments"])
    df = pd.DataFrame(rows)
    df.insert(0, "event_slug", doc["event"])
    df.insert(0, "season", int(doc["season"]))
    meta = {"driver": code, "lap_uid": lap["lap_uid"], "mu": round(fit["mu"], 3), "cl_a": round(fit["cl_a"], 3),
            "cd_a": round(fit["cd_a"], 3), "rms_kph": round(fit["rms_kph"], 1)}
    return df, meta


def run(scope_path: Path, store: Path, verbose: bool) -> int:
    scope = paths.load_scope(scope_path)
    silver = silver_dir(scope_path, scope)
    frames, failures = [], []
    events = sorted(p for p in store.glob("*/*") if p.is_dir())
    for i, ev in enumerate(events, 1):
        label = f"{ev.parent.name}/{ev.name}"
        try:
            res = build_event(ev)
            if res is None:
                failures.append({"event": label, "error": "no Q.json or line.json"})
                continue
            df, meta = res
            out = silver / ev.parent.name / ev.name
            out.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / "segment_physics.parquet", index=False, compression="zstd")
            # the API needs the same numbers at inference time: ship them with the store
            cols = [c for c in df.columns if c.startswith("segment_")]
            (ev / "segment_physics.json").write_text(json.dumps(
                {"fitted_lap": meta, "segments": [{k: (round(float(r[k]), 4) if k != "segment_index" else int(r[k]))
                                                   for k in cols} for _, r in df.iterrows()]}, separators=(",", ":")))
            frames.append(df)
            if verbose:
                tr = df["segment_traction_share"].mean()
                print(f"[{i:>3}/{len(events)}] {label:<36s} {len(df):>3} segments  {meta['driver']}  "
                      f"traction share {tr:.2f}  climb {df['segment_climb_m'].sum():5.1f} m  rms {meta['rms_kph']:.1f}")
        except Exception as exc:                                  # noqa: BLE001
            failures.append({"event": label, "error": f"{type(exc).__name__}: {exc}"[:200]})
            print(f"[{i:>3}/{len(events)}] {label:<36s} FAILED {exc}"[:150], file=sys.stderr)
    if frames:
        allp = pd.concat(frames, ignore_index=True)
        allp.to_parquet(silver / "segment_physics.parquet", index=False, compression="zstd")
        print(f"\nsegment physics: {len(allp)} segments over {allp.groupby(['season', 'event_slug']).ngroups} events")
        kinds = allp[["segment_traction_share", "segment_brake_share", "segment_grip_share", "segment_power_share"]].mean()
        print("  mean shares  " + "  ".join(f"{k.split('_')[1]} {v:.2f}" for k, v in kinds.items()))
    if failures:
        print(f"  failures: {len(failures)}  e.g. {failures[0]}")
    return 0 if frames else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", type=Path, default=Path("configs/scope.yaml"))
    ap.add_argument("--store", type=Path, default=Path("../data/artifacts/baselines"))
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    sys.exit(run(a.scope, a.store, a.verbose))


if __name__ == "__main__":
    main()
