"""P2 orchestrator — bronze -> silver track geometry.

Per session: pick one reference lap, derive the circuit's geometry from it,
split it into corners and straights, and write the track definition the ML
features (P2-5) and the HUD track map (P7-6) both read.

Choosing the reference lap is the decision that matters. It is the fastest
lap whose telemetry is intact — using `telemetry_quality` from P1, which
exists precisely because 21% of a qualifying session came back with feed
dropouts. Deriving a circuit's geometry from a lap with a 4-second hole in it
would put a phantom straight through the middle of a corner sequence.

Usage
-----
    python -m pipeline.segment.run --scope configs/scope.yaml
    python -m pipeline.segment.run --scope configs/scope.yaml --limit 1 --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.ingest import paths
from pipeline.segment import geometry as G, segmentation as S, svg as V

QUALITY_RANK = {"clean": 0, "normal": 1, "gappy": 2, "holed": 3, "missing": 4}
SWEEP = [0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0045, 0.0060, 0.0080]


def silver_dir(scope_path: Path, scope: dict) -> Path:
    return paths.lake_dir(scope_path, scope) / "silver"


def pick_reference_lap(laps: pd.DataFrame) -> pd.Series:
    """Fastest lap with intact telemetry, not simply the fastest lap."""
    df = laps.copy()
    df["_q"] = df.get("telemetry_quality", pd.Series("normal", index=df.index)) \
        .map(lambda v: QUALITY_RANK.get(str(v), 9))
    df["_t"] = pd.to_numeric(df.get("lap_time_s"), errors="coerce")
    df = df[df["_t"].notna()]
    if not len(df):
        raise ValueError("no lap with a valid lap time")

    # Prefer clean/normal; only fall back to gappy if nothing better exists.
    for ceiling in (1, 2, 3, 9):
        sub = df[df["_q"] <= ceiling]
        if len(sub):
            return sub.sort_values("_t").iloc[0]
    raise ValueError("no usable reference lap")


def process_session(session_dir: Path, out_dir: Path, cfg: dict,
                    verbose: bool = False) -> dict:
    laps = pd.read_parquet(session_dir / "laps.parquet")
    tel = pd.read_parquet(session_dir / "telemetry.parquet")
    meta = json.loads((session_dir / "session.json").read_text())

    ref = pick_reference_lap(laps)
    uid = str(ref["lap_uid"])
    frame = tel[tel["lap_uid"] == uid].sort_values("distance_m")
    if len(frame) < 20:
        raise ValueError(f"reference lap {uid} has only {len(frame)} samples")

    geo = G.build(frame, lap_distance_m=float(ref.get("lap_distance_m") or
                                              frame["distance_m"].iloc[-1]),
                  step_m=float(cfg.get("geometry_step_m", 10.0)),
                  smooth_window_m=float(cfg.get("smooth_window_m", 60.0)))

    seg_kw = {"min_gap_m": float(cfg.get("min_gap_m", 30.0)),
              "min_segment_len_m": float(cfg.get("min_segment_len_m", 40.0))}
    sweep = S.sweep_thresholds(geo.curvature_1pm, geo.grid_step_m, SWEEP, **seg_kw)

    raw_d = frame["distance_m"].to_numpy(dtype=float)
    raw_v = pd.to_numeric(frame.get("speed_kph"), errors="coerce").to_numpy(dtype=float)
    segments = S.build_segments(geo, cfg, raw_distance_m=raw_d, raw_speed_kph=raw_v)

    # Official sectors, converted from the reference lap's sector TIMES.
    bounds: list[float] = []
    if {"sector1_s", "sector2_s", "lap_start_s"} <= set(laps.columns) and \
            "session_time_s" in frame.columns:
        try:
            bounds = S.sector_boundaries_from_times(
                raw_d, frame["session_time_s"].to_numpy(dtype=float),
                float(ref["lap_start_s"]), float(ref["sector1_s"]), float(ref["sector2_s"]))
            S.assign_sectors(segments, bounds)
        except (TypeError, ValueError):
            bounds = []

    kinds: dict[str, int] = {}
    for s in segments:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    corners = sum(v for k, v in kinds.items() if k.endswith("_corner"))

    doc = {
        "season": meta["season"], "event": meta["event"],
        "event_slug": meta["event_slug"], "session": meta["session"],
        "circuit": meta.get("circuit", ""),
        "reference_lap": {
            "lap_uid": uid,
            "driver": str(ref.get("driver", "")),
            "lap_time_s": float(ref["_t"]),
            "telemetry_quality": str(ref.get("telemetry_quality", "")),
            "samples": int(len(frame)),
        },
        "geometry": {
            "lap_length_m": round(geo.lap_length_m, 1),
            "grid_step_m": geo.grid_step_m,
            "metres_per_xy_unit": round(geo.metres_per_unit, 6),
            "closure_error_m": round(geo.closure_error_m, 2),
            "smooth_window_m": float(cfg.get("smooth_window_m", 60.0)),
        },
        "threshold_sweep_corner_count": sweep,
        "threshold_used_1pm": float(cfg.get("curvature_threshold_1pm", 0.0035)),
        "counts": {"segments": len(segments), "corners": corners, **kinds},
        "sector_boundaries_m": [round(b, 1) for b in bounds],
        "segments": [s.to_dict() for s in segments],
        "microsectors": S.microsectors(geo.lap_length_m,
                                       int(cfg.get("microsectors", 28))),
        "track_map": V.build(geo),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "track.json").write_text(json.dumps(doc, indent=1))
    pd.DataFrame([s.to_dict() for s in segments]).to_parquet(
        out_dir / "segments.parquet", index=False, compression="zstd")

    if verbose:
        print(f"  reference : {uid}  {ref.get('driver')}  "
              f"{doc['reference_lap']['lap_time_s']:.3f}s  "
              f"({doc['reference_lap']['telemetry_quality']})")
        print(f"  geometry  : {geo.lap_length_m:.0f} m, scale "
              f"{geo.metres_per_unit:.5f} m/unit, closure {geo.closure_error_m:.2f} m")
        print("  curvature threshold sweep -> corner count "
              "(compare against the circuit's published count)")
        for t, c in sweep.items():
            mark = "  <-- current" if abs(float(t) - doc["threshold_used_1pm"]) < 1e-9 else ""
            print(f"    {t} 1/m  ({1/float(t):>5.0f} m radius)  -> {c:>3} corners{mark}")
        print(f"  segments  : {len(segments)}  ({corners} corners)  " +
              ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    return doc


def run(scope_path: Path, limit: int | None, force: bool, verbose: bool) -> int:
    scope = paths.load_scope(scope_path)
    bronze = paths.bronze_dir(scope_path, scope)
    silver = silver_dir(scope_path, scope)
    cfg = scope.get("segmentation", {})

    sessions = sorted(p.parent for p in bronze.rglob("session.json"))
    todo = []
    for d in sessions:
        rel = d.relative_to(bronze)
        out = silver / rel
        if (out / "track.json").exists() and not force:
            continue
        todo.append((d, out))
    if limit:
        todo = todo[:limit]

    print(f"bronze     : {bronze}")
    print(f"silver     : {silver}")
    print(f"ingested   : {len(sessions)} session(s)")
    print(f"to segment : {len(todo)}")
    print()

    done, failures = [], []
    for i, (src, out) in enumerate(todo, 1):
        label = "/".join(src.parts[-3:])
        try:
            doc = process_session(src, out, cfg, verbose=verbose)
            done.append(doc)
            print(f"[{i:>3}/{len(todo)}] {label:<40s} "
                  f"{doc['counts']['corners']:>3} corners, "
                  f"{doc['counts']['segments']:>3} segments, "
                  f"{doc['geometry']['lap_length_m']:>7.0f} m, "
                  f"closure {doc['geometry']['closure_error_m']:.2f} m")
        except Exception as exc:                            # noqa: BLE001
            failures.append({"session": label, "error": f"{type(exc).__name__}: {exc}"[:240]})
            print(f"[{i:>3}/{len(todo)}] {label:<40s} FAILED  "
                  f"{type(exc).__name__}: {exc}"[:160], file=sys.stderr)
            if verbose:
                traceback.print_exc()

    if done:
        print()
        print("GEOMETRY sanity")
        worst = max(d["geometry"]["closure_error_m"] for d in done)
        print(f"  worst closure error : {worst:.2f} m  "
              f"(a closed lap should return to its start; >20 m means the "
              f"reference lap was cut badly)")
        scales = [d["geometry"]["metres_per_xy_unit"] for d in done]
        print(f"  derived X/Y scale   : {min(scales):.5f} - {max(scales):.5f} m/unit "
              f"(should be consistent across circuits)")
        print()
        print("CORNER COUNT per circuit  (compare against published counts)")
        for d in done:
            print(f"  {d['season']} {d['event']:<32s} {d['counts']['corners']:>3} corners "
                  f"({d['counts'].get('low_speed_corner',0)} low / "
                  f"{d['counts'].get('medium_speed_corner',0)} med / "
                  f"{d['counts'].get('high_speed_corner',0)} high)")

    silver.mkdir(parents=True, exist_ok=True)
    (silver / "segment_report.json").write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sessions": [{k: d[k] for k in ("season", "event", "session", "counts",
                                        "geometry", "threshold_sweep_corner_count",
                                        "reference_lap")} for d in done],
        "failures": failures,
    }, indent=1))
    print()
    print(f"segmented {len(done)} session(s), {len(failures)} failure(s)")
    return 1 if failures and not done else 0


def main() -> None:
    p = argparse.ArgumentParser(description="Segment circuits from bronze telemetry.")
    p.add_argument("--scope", type=Path, default=Path("configs/scope.yaml"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args()
    try:
        sys.exit(run(a.scope, a.limit, a.force, a.verbose))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:                                       # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
