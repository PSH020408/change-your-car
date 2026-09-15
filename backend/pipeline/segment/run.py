"""P2 orchestrator — bronze -> silver track geometry.

Per CIRCUIT (season x event), not per session. The first version segmented
each session independently and the same circuit came out with 29 segments
from qualifying and 35 from the race — segment #5 meant one thing on Saturday
and another on Sunday, driver-style z-scores compared across mismatched
indices, and the HUD would have needed two track maps for one track.

A circuit's geometry is a property of the circuit. It is derived once, from
the best-covered lap across all of that weekend's sessions, and applied to
every session.

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

import yaml

from pipeline.ingest import paths
from pipeline.segment import ensemble as E, geometry as G, segmentation as S, svg as V

ENSEMBLE_LAPS = 16        # sqrt(16) = 4x jitter reduction; more gains little
ENSEMBLE_COVERAGE = (0.97, 1.03)


def load_circuit_reference(scope_path: Path) -> dict:
    """Published lap lengths and corner counts, where they have been verified."""
    f = scope_path.parent / "circuits.yaml"
    if not f.exists():
        return {}
    return yaml.safe_load(f.read_text()) or {}

QUALITY_RANK = {"clean": 0, "normal": 1, "gappy": 2, "holed": 3, "missing": 4}
SWEEP = [0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0045, 0.0060, 0.0080]
WINDOW_SWEEP = [40.0, 50.0, 65.0, 80.0, 100.0, 120.0]


def silver_dir(scope_path: Path, scope: dict) -> Path:
    return paths.lake_dir(scope_path, scope) / "silver"


SESSION_PREFERENCE = {"Q": 0, "SQ": 1, "R": 2, "S": 3}


def ensemble_frames(laps_by: dict[str, pd.DataFrame], tel_by: dict[str, pd.DataFrame],
                    anchor_length_m: float, n: int = ENSEMBLE_LAPS
                    ) -> list[tuple[str, pd.Series, pd.DataFrame]]:
    """The best-covered clean laps of the weekend, for the ensemble centreline.

    Coverage is judged against the anchor lap's length. Quality clean/normal
    only: a lap with a 4 s hole in its position stream would drag the median
    toward a straight line through the hole.
    """
    rows = []
    for ses, laps in laps_by.items():
        df = laps.copy()
        df["_q"] = df.get("telemetry_quality", pd.Series("normal", index=df.index)) \
            .map(lambda v: QUALITY_RANK.get(str(v), 9))
        df["_t"] = pd.to_numeric(df.get("lap_time_s"), errors="coerce")
        df["_len"] = pd.to_numeric(df.get("lap_distance_m"), errors="coerce")
        cov = df["_len"] / anchor_length_m
        df = df[df["_t"].notna() & (df["_q"] <= 1) & cov.between(*ENSEMBLE_COVERAGE)]
        df["_s"] = SESSION_PREFERENCE.get(ses, 9)
        for _, r in df.iterrows():
            rows.append((r["_s"], r["_q"], r["_t"], ses, str(r["lap_uid"])))
    rows.sort()
    out = []
    for _, _, _, ses, uid in rows[:n]:
        f = tel_by[ses][tel_by[ses]["lap_uid"] == uid].sort_values("distance_m")
        if len(f) >= 20:
            ref = laps_by[ses][laps_by[ses]["lap_uid"].astype(str) == uid].iloc[0]
            out.append((ses, ref, f))
    return out


def _lap_sector_bounds(laps: pd.DataFrame, ref: pd.Series, frame: pd.DataFrame) -> list[float]:
    """Sector boundaries on ONE lap's own distance axis, or [] if it can't say."""
    if not {"sector1_s", "sector2_s", "lap_start_s"} <= set(laps.columns) or \
            "session_time_s" not in frame.columns:
        return []
    try:
        return S.sector_boundaries_from_times(
            frame["distance_m"].to_numpy(dtype=float),
            frame["session_time_s"].to_numpy(dtype=float),
            float(ref["lap_start_s"]), float(ref["sector1_s"]), float(ref["sector2_s"]))
    except (TypeError, ValueError):
        return []


def candidate_laps(laps_by_session: dict[str, pd.DataFrame], n: int = 4) -> list[tuple[str, pd.Series]]:
    """Best few reference candidates across a weekend's sessions.

    Qualifying first — a Q lap is driven alone at the limit on a clean track —
    then by telemetry quality, then by lap time. Several candidates are
    returned because the decisive property (how much of the circuit the
    telemetry slice actually covers) is only known after the geometry is
    built, so the final choice is made on that.
    """
    rows = []
    for ses, laps in laps_by_session.items():
        df = laps.copy()
        df["_q"] = df.get("telemetry_quality", pd.Series("normal", index=df.index)) \
            .map(lambda v: QUALITY_RANK.get(str(v), 9))
        df["_t"] = pd.to_numeric(df.get("lap_time_s"), errors="coerce")
        df = df[df["_t"].notna() & (df["_q"] <= 2)]
        df["_s"] = SESSION_PREFERENCE.get(ses, 9)
        for _, r in df.sort_values(["_s", "_q", "_t"]).head(n).iterrows():
            rows.append((ses, r))
    rows.sort(key=lambda x: (SESSION_PREFERENCE.get(x[0], 9), x[1]["_q"], x[1]["_t"]))
    if not rows:
        raise ValueError("no usable reference lap in any session")
    return rows[:n]


def circuit_cfg(base: dict, ref: dict | None, lap_length_m: float) -> dict:
    """Segmentation settings for THIS circuit.

    Precedence: an explicit per-circuit override, then the global config, then
    a window auto-scaled to the circuit's size. Monaco needs a tighter window
    than Spa and no single number serves both.
    """
    cfg = dict(base)
    for k, v in ((ref or {}).get("segmentation") or {}).items():
        cfg[k] = v                                   # per-circuit override first
    if not cfg.get("smooth_window_m"):
        cfg["smooth_window_m"] = G.auto_window_m(lap_length_m)
    if not cfg.get("min_gap_m"):
        cfg["min_gap_m"] = round(float(cfg["smooth_window_m"]) / 3.0, 1)
    return cfg


def process_circuit(sessions: list[tuple[Path, dict]], out_dir: Path, cfg: dict,
                    reference: dict | None = None, verbose: bool = False,
                    calibrate: bool = False) -> dict:
    """One track definition for a (season, event), from its best-covered lap."""
    laps_by, tel_by, meta_by = {}, {}, {}
    for d, meta in sessions:
        ses = meta["session"]
        laps_by[ses] = pd.read_parquet(d / "laps.parquet")
        tel_by[ses] = pd.read_parquet(d / "telemetry.parquet")
        meta_by[ses] = meta
    first_meta = next(iter(meta_by.values()))

    circuit_ref = (reference or {}).get(first_meta["event_slug"])

    # Build geometry for each candidate and keep the one whose telemetry
    # slice covers the circuit best. Coverage, not lap time, is what makes a
    # lap a good template — the fastest lap of the weekend is useless as a
    # circuit definition if its position stream starts 100 m after the line.
    best = None
    for ses, ref in candidate_laps(laps_by):
        uid = str(ref["lap_uid"])
        frame = tel_by[ses][tel_by[ses]["lap_uid"] == uid].sort_values("distance_m")
        if len(frame) < 20:
            continue
        lap_len_hint = float(ref.get("lap_distance_m") or frame["distance_m"].iloc[-1])
        c = circuit_cfg(cfg, circuit_ref, lap_len_hint)
        try:
            geo = G.build(frame, lap_distance_m=lap_len_hint,
                          step_m=float(c.get("geometry_step_m", 10.0)),
                          smooth_window_m=float(c["smooth_window_m"]),
                          poly_order=int(c.get("poly_order", 2)))
        except ValueError:
            continue
        score = geo.slice_gap_m + geo.closure_error_m
        if best is None or score < best[0]:
            best = (score, ses, ref, frame, geo, c, lap_len_hint)
    if best is None:
        raise ValueError("no candidate lap produced a geometry")
    _, ses, ref, frame, geo, cfg, lap_len_hint = best
    uid = str(ref["lap_uid"])

    # The anchor lap settles length, phase and window. The SHAPE comes from
    # the ensemble: the median of the weekend's best laps, phase-locked on
    # speed. One lap's GPS lottery gave the same circuit 10, 14 and 14 turns
    # in three years; sixteen laps averaged do not.
    # Sector boundaries come from lap TIMES, so they need a frame that still
    # has a time axis. The ensemble frame has none (it is a median of
    # positions on a distance grid) — the first ensemble run lost every
    # sector boundary this way and 100% of segment_sector went null. Each
    # lap's boundaries are therefore measured on its own axis, moved onto
    # the ensemble axis (rescale + phase shift), and the median is kept.
    anchor_bounds = _lap_sector_bounds(laps_by[ses], ref, frame)
    ens = None
    ens_bounds: list[float] = []
    try:
        picks = ensemble_frames(laps_by, tel_by, lap_len_hint)
        if len(picks) >= 3:
            ens = E.build_ensemble([f for _, _, f in picks], lap_len_hint, step_m=5.0)
            geo = G.build(ens.frame, lap_distance_m=lap_len_hint,
                          step_m=float(cfg.get("geometry_step_m", 10.0)),
                          smooth_window_m=float(cfg["smooth_window_m"]),
                          poly_order=int(cfg.get("poly_order", 2)))
            frame = ens.frame
            per_lap = []
            for j, idx in enumerate(ens.used or []):
                p_ses, p_ref, p_frame = picks[idx]
                b = _lap_sector_bounds(laps_by[p_ses], p_ref, p_frame)
                if len(b) == 2:
                    per_lap.append([ens.to_ensemble_axis(j, x) for x in b])
            ens_bounds = S.median_sector_boundaries(per_lap, lap_len_hint)
    except ValueError:
        ens = None

    seg_kw = {"min_gap_m": float(cfg.get("min_gap_m", 30.0)),
              "min_segment_len_m": float(cfg.get("min_segment_len_m", 40.0))}
    sweep = S.sweep_thresholds(geo.curvature_1pm, geo.grid_step_m, SWEEP, **seg_kw)

    raw_d = frame["distance_m"].to_numpy(dtype=float)
    raw_v = pd.to_numeric(frame.get("speed_kph"), errors="coerce").to_numpy(dtype=float)
    sweep_rows = S.sweep_full(geo, cfg, SWEEP, raw_distance_m=raw_d,
                              raw_speed_kph=raw_v,
                              max_lateral_g=float(cfg.get("max_lateral_g", 6.5)))
    segments = S.build_segments(geo, cfg, raw_distance_m=raw_d, raw_speed_kph=raw_v)

    bounds: list[float] = ens_bounds if ens is not None and ens_bounds else anchor_bounds
    if bounds:
        S.assign_sectors(segments, bounds)

    physics = S.check_physics(segments,
                              float(cfg.get("curvature_threshold_1pm", 0.0035)),
                              float(cfg.get("max_lateral_g", 6.5)),
                              float(cfg.get("hidden_corner_factor", 0.7)))

    kinds: dict[str, int] = {}
    for sg in segments:
        kinds[sg.kind] = kinds.get(sg.kind, 0) + 1
    turns = S.count_turns(segments)
    corners = turns["turns"]

    doc = {
        "season": first_meta["season"], "event": first_meta["event"],
        "event_slug": first_meta["event_slug"],
        "sessions_covered": sorted(meta_by),
        "circuit": first_meta.get("circuit", ""),
        "reference_lap": {
            "lap_uid": uid, "session": ses,
            "driver": str(ref.get("driver", "")),
            "lap_time_s": float(ref["_t"]),
            "telemetry_quality": str(ref.get("telemetry_quality", "")),
            "samples": int(len(frame)),
        },
        "ensemble": ({
            "laps": ens.n_laps,
            "rejected": ens.rejected,
            "phase_shift_m_max": round(max(abs(x) for x in ens.phase_shifts_m), 1),
            "grid_step_m": 5.0,
            "sector_bounds_from": "ensemble median" if ens_bounds else "anchor lap",
        } if ens else {"laps": 1, "note": "fell back to the anchor lap alone"}),
        "geometry": {
            "lap_length_m": round(geo.lap_length_m, 1),
            "grid_step_m": geo.grid_step_m,
            "metres_per_xy_unit": round(geo.metres_per_unit, 6),
            "closure_error_m": round(geo.closure_error_m, 2),
            "slice_gap_m": geo.slice_gap_m,
            "implausible_curvature_fraction": geo.implausible_fraction,
            "smooth_window_m": float(cfg["smooth_window_m"]),
        },
        "threshold_sweep_corner_count": sweep,
        "threshold_sweep": sweep_rows,
        "threshold_used_1pm": float(cfg.get("curvature_threshold_1pm", 0.0035)),
        "counts": {"segments": len(segments), "corners": turns["corners"],
                   "kinks": turns["kinks"], "turns": turns["turns"], **kinds},
        "physics_check": physics,
        "sector_boundaries_m": [round(b, 1) for b in bounds],
        "segments": [sg.to_dict() for sg in segments],
        "microsectors": S.microsectors(geo.lap_length_m,
                                       int(cfg.get("microsectors", 28))),
        "track_map": V.build(geo),
        "smooth_window_m": float(cfg["smooth_window_m"]),
        "published_reference": _compare_reference(
            circuit_ref, first_meta["season"], geo.lap_length_m, corners),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "track.json").write_text(json.dumps(doc, indent=1))
    pd.DataFrame([sg.to_dict() for sg in segments]).to_parquet(
        out_dir / "segments.parquet", index=False, compression="zstd")

    if calibrate:
        pub = (circuit_ref or {}).get("turns")
        print(f"  CALIBRATION grid   window x threshold -> turns"
              + (f"   (published: {pub})" if pub else ""))
        print("    {:>7}".format("win\\1/m") + "".join("{:>8.4f}".format(t) for t in SWEEP))
        for w in WINDOW_SWEEP:
            g2 = G.build(frame, lap_distance_m=lap_len_hint,
                         step_m=float(cfg.get("geometry_step_m", 10.0)),
                         smooth_window_m=w, poly_order=int(cfg.get("poly_order", 2)))
            c2 = dict(cfg); c2["smooth_window_m"] = w
            rows2 = S.sweep_full(g2, c2, SWEEP, raw_distance_m=raw_d,
                                 raw_speed_kph=raw_v,
                                 max_lateral_g=float(cfg.get("max_lateral_g", 6.5)))
            cells = []
            for r in rows2:
                mark = "*" if pub and r["turns"] == pub else ("+" if r["physics_passes"] else " ")
                cells.append("{:>7}{}".format(r["turns"], mark))
            print("    {:>7.0f}".format(w) + "".join(cells))
        print("      * = matches published turn count, + = physics invariants pass")

    if verbose:
        print(f"  anchor    : {uid}  {ref.get('driver')}  {ses}  "
              f"{doc['reference_lap']['lap_time_s']:.3f}s  "
              f"({doc['reference_lap']['telemetry_quality']})")
        e = doc["ensemble"]
        print(f"  ensemble  : {e['laps']} laps"
              + (f", phase-locked within {e['phase_shift_m_max']} m, "
                 f"{e['rejected']} rejected" if e.get('laps', 1) > 1 else "  (anchor only)"))
        if bounds:
            print(f"  sectors   : S1 ends {bounds[0]:.0f} m, S2 ends {bounds[1]:.0f} m  "
                  f"({e.get('sector_bounds_from', 'anchor lap')}, "
                  f"{sum(1 for sg in segments if sg.sector is None)} segments unassigned)")
        else:
            print("  sectors   : NONE — segment_sector will be null downstream")
        print(f"  geometry  : {geo.lap_length_m:.0f} m, scale "
              f"{geo.metres_per_unit:.5f} m/unit, closure {geo.closure_error_m:.2f} m, "
              f"window {cfg['smooth_window_m']:.0f} m")
        ref_pub = doc.get("published_reference") or {}
        pub = ref_pub.get("published_turns") if ref_pub.get("applicable") else None
        print("  curvature threshold sweep" + (f"   (published: {pub} turns)" if pub else ""))
        print("    {:<8}{:>8}{:>7}{:>7}{:>13}{:>6}{:>7}".format(
            "1/m", "radius", "g@300", "turns", "L/M/H+kink", "segs", "phys"))
        for r in sweep_rows:
            mark = "  <-- current" if abs(r["threshold_1pm"] - doc["threshold_used_1pm"]) < 1e-9 else ""
            hit = "  == published" if pub and r["turns"] == pub else ""
            comp = "{}/{}/{}+{}k".format(r["low"], r["medium"], r["high"], r["kinks"])
            verdict = "PASS" if r["physics_passes"] else "fail"
            print("    {:<8.4f}{:>7} m{:>7.1f}{:>7}{:>13}{:>6}{:>7}{}{}".format(
                r["threshold_1pm"], r["radius_m"], r["lateral_g_at_300kph"],
                r["turns"], comp, r["segments"], verdict, hit, mark))
        print(f"  segments  : {len(segments)}  ({corners} turns)  " +
              ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
        ok = "PASS" if physics["passes"] else "FAIL"
        print(f"  physics   : {ok}  {len(physics['corners_over_g_limit'])} corner(s) over "
              f"{physics['max_lateral_g']} g, "
              f"{len(physics['straights_hiding_a_corner'])} straight(s) hiding a corner"
              + (f", {len(physics['straights_at_the_boundary'])} at the boundary"
                 if physics["straights_at_the_boundary"] else ""))
        for h in physics["straights_hiding_a_corner"][:3]:
            print(f"    seg #{h['index']}: {h['length_m']} m straight, interior R={h['radius_m']} m")
    return doc


def _compare_reference(ref: dict | None, season: int,
                       lap_length_m: float, corners: int) -> dict | None:
    """Measured versus published, when a verified figure exists."""
    if not ref:
        return None
    if ref.get("layout_from") and season < int(ref["layout_from"]):
        return {"applicable": False,
                "reason": f"reference describes the {ref['layout_from']}+ layout"}
    pub_m = float(ref["length_km"]) * 1000.0
    return {
        "applicable": True,
        "published_length_m": pub_m,
        "measured_length_m": round(lap_length_m, 1),
        "length_delta_pct": round(100.0 * (lap_length_m - pub_m) / pub_m, 2),
        "published_turns": int(ref["turns"]),
        "measured_corners": corners,
        "corner_delta": corners - int(ref["turns"]),
        "source": ref.get("source", ""),
    }


def run(scope_path: Path, limit: int | None, force: bool, verbose: bool,
        calibrate: bool = False) -> int:
    scope = paths.load_scope(scope_path)
    bronze = paths.bronze_dir(scope_path, scope)
    silver = silver_dir(scope_path, scope)
    cfg = scope.get("segmentation", {})
    reference = load_circuit_reference(scope_path)

    circuits: dict[tuple[int, str], list[tuple[Path, dict]]] = {}
    for sj in sorted(bronze.rglob("session.json")):
        meta = json.loads(sj.read_text())
        circuits.setdefault((int(meta["season"]), meta["event_slug"]), []).append((sj.parent, meta))

    todo = []
    for (season, slug), sessions in sorted(circuits.items()):
        out = silver / str(season) / slug
        if (out / "track.json").exists() and not force:
            continue
        todo.append(((season, slug), sessions, out))
    if limit:
        todo = todo[:limit]

    print(f"bronze     : {bronze}")
    print(f"silver     : {silver}")
    print(f"circuits   : {len(circuits)} (from {sum(len(v) for v in circuits.values())} sessions)")
    print(f"to segment : {len(todo)}")
    print()

    done, failures = [], []
    for i, ((season, slug), sessions, out) in enumerate(todo, 1):
        label = f"{season}/{slug}"
        try:
            doc = process_circuit(sessions, out, cfg, reference=reference,
                                  verbose=verbose, calibrate=calibrate)
            done.append(doc)
            print(f"[{i:>3}/{len(todo)}] {label:<36s} "
                  f"{doc['counts']['turns']:>3} turns, {doc['counts']['segments']:>3} segments, "
                  f"{doc['geometry']['lap_length_m']:>6.0f} m, "
                  f"ensemble {doc['ensemble']['laps']:>2} laps, "
                  f"sessions {','.join(doc['sessions_covered'])}")
        except Exception as exc:                            # noqa: BLE001
            failures.append({"circuit": label, "error": f"{type(exc).__name__}: {exc}"[:240]})
            print(f"[{i:>3}/{len(todo)}] {label:<36s} FAILED  {type(exc).__name__}: {exc}"[:160],
                  file=sys.stderr)
            if verbose:
                traceback.print_exc()

    if done:
        print()
        print("GEOMETRY sanity")
        worst = max(d["geometry"]["closure_error_m"] for d in done)
        gap = max(d["geometry"].get("slice_gap_m", 0.0) for d in done)
        print(f"  worst closure error : {worst:.2f} m beyond the slice gap")
        print(f"  worst slice gap     : {gap:.1f} m")
        scales = [d["geometry"]["metres_per_xy_unit"] for d in done]
        print(f"  derived X/Y scale   : {min(scales):.5f} - {max(scales):.5f} m/unit")
        failed = [d for d in done if not d["physics_check"]["passes"]]
        print()
        print("PHYSICS invariants")
        print(f"  circuits failing : {len(failed)} / {len(done)}")
        print()
        print("TURN COUNT per circuit  (measured vs published)")
        for d in done:
            r = d.get("published_reference")
            c = d["counts"]
            tail = "  (no verified reference)"
            if r and r.get("applicable"):
                tail = (f"  vs published {r['published_turns']} ({r['corner_delta']:+d}), "
                        f"length {r['length_delta_pct']:+.2f}%")
            phys = "PASS" if d["physics_check"]["passes"] else "fail"
            print(f"  {d['season']} {d['event']:<28s} {c.get('turns',0):>3} = "
                  f"{c.get('corners',0)} corners + {c.get('kinks',0)} kinks  "
                  f"[{phys}]" + tail)

    silver.mkdir(parents=True, exist_ok=True)
    (silver / "segment_report.json").write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "circuits": [{k: d[k] for k in ("season", "event", "event_slug", "sessions_covered",
                                        "counts", "geometry", "physics_check",
                                        "threshold_sweep_corner_count", "reference_lap")}
                     for d in done],
        "failures": failures,
    }, indent=1))
    print()
    print(f"segmented {len(done)} circuit(s), {len(failures)} failure(s)")
    return 1 if failures and not done else 0


def main() -> None:
    p = argparse.ArgumentParser(description="Segment circuits from bronze telemetry.")
    p.add_argument("--scope", type=Path, default=Path("configs/scope.yaml"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--calibrate", action="store_true",
                   help="sweep smoothing window x curvature threshold per circuit")
    a = p.parse_args()
    try:
        sys.exit(run(a.scope, a.limit, a.force, a.verbose, a.calibrate))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:                                       # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
