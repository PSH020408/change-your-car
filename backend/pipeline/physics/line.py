"""P9-1 — the racing line the physics engine drives on.

`track.json` carries curvature per SEGMENT (mean, peak, minimum radius). A
quasi-steady-state lap simulation needs it per METRE, plus two things the
segmenter never used: the road gradient and where DRS is actually opened.
This step rebuilds the very same ensemble line the segmenter used (same laps,
same phase alignment, same smoothing window) so its curvature agrees with
the segments, and writes it out on a 5 m grid:

    line.parquet   distance_m, x_m, y_m, z_m, curvature_1pm, grade, drs_share, speed_kph
    line.json      quality figures, the elevation gate, agreement with track.json

ELEVATION IS GATED, NOT ASSUMED. The position feed's Z channel is the least
trusted of the three: it is what the timing system reports, not a survey.
Before a gradient is allowed into the engine the lap must close in height
(start and finish are the same place) and the weekend's laps must agree with
each other. A circuit that fails either keeps a flat profile and says so.

Usage
-----
    python -m pipeline.physics.line --scope configs/scope.yaml
    python -m pipeline.physics.line --scope configs/scope.yaml --only monaco_grand_prix --force
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.ingest import paths
from pipeline.segment import ensemble as E, geometry as G
from pipeline.physics.qss_fit import line_to_store
from pipeline.segment.run import (candidate_laps, circuit_cfg, ensemble_frames,
                                  load_circuit_reference, silver_dir)

GRID_STEP_M = 5.0
DRS_OPEN_CODES = (10, 12, 14)
# Elevation gate (G0 in the physics-engine design): the lap must return to
# its starting height within this, and the laps must agree this closely.
Z_CLOSURE_MAX_M = 3.0
Z_SPREAD_MAX_M = 2.0


def elevation_profile(s: np.ndarray, z_m: np.ndarray, window_m: float, order: int,
                      period_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed height and its gradient dz/ds along the lap (closed loop)."""
    zs, dz, _ = G.local_poly_derivatives(s, z_m, window_m, order=order, closed=True,
                                          period=period_m)
    return zs, dz


def drs_share(picks, ens: E.EnsembleResult, grid: np.ndarray) -> np.ndarray | None:
    """Fraction of the ensemble laps with DRS open at each grid point.

    Every used lap is moved onto the ensemble axis exactly as its position
    was (rescale, then phase shift), so the share lines up with curvature.
    """
    rows = []
    step = float(grid[1] - grid[0])
    for j, idx in enumerate(ens.used or []):
        _, _, f = picks[idx]
        if "drs_raw" not in f.columns:
            continue
        d = pd.to_numeric(f["distance_m"], errors="coerce").to_numpy(dtype=float)
        d = d * (ens.scales[j] if ens.scales else 1.0)
        flag = pd.to_numeric(f["drs_raw"], errors="coerce").isin(DRS_OPEN_CODES).to_numpy(dtype=float)
        ok = np.isfinite(d)
        d, flag = d[ok], flag[ok]
        order = np.argsort(d)
        d, flag = d[order], flag[order]
        keep = np.concatenate([[True], np.diff(d) > 0])
        v = np.interp(grid, d[keep], flag[keep], period=float(grid[-1] + step))
        rows.append(np.roll(v, int(round(ens.phase_shifts_m[j] / step))))
    if not rows:
        return None
    return np.vstack(rows).mean(axis=0)


def curvature_agreement(track_doc: dict, grid: np.ndarray, k: np.ndarray) -> dict:
    """How well the 5 m line reproduces the peak curvature of each corner segment."""
    errs = []
    for sg in track_doc.get("segments", []):
        if sg.get("kind") == "straight" or not sg.get("peak_curvature_1pm"):
            continue
        a, b = float(sg["start_m"]), float(sg["end_m"])
        m = (grid >= a) & (grid <= b) if b >= a else (grid >= a) | (grid <= b)
        if not m.any():
            continue
        peak_line = float(np.max(np.abs(k[m])))
        peak_seg = abs(float(sg["peak_curvature_1pm"]))
        errs.append(abs(peak_line - peak_seg) / max(peak_seg, 1e-6))
    if not errs:
        return {"corners_checked": 0}
    e = np.asarray(errs)
    return {"corners_checked": int(len(e)),
            "peak_curvature_rel_err_median": round(float(np.median(e)), 4),
            "peak_curvature_rel_err_p90": round(float(np.quantile(e, 0.9)), 4)}


def build_line(sessions: list[tuple[Path, dict]], cfg: dict, reference: dict | None,
               track_doc: dict, window_override_m: float | None = None) -> tuple[pd.DataFrame, dict]:
    laps_by, tel_by, meta_by = {}, {}, {}
    for d, meta in sessions:
        ses = meta["session"]
        laps_by[ses] = pd.read_parquet(d / "laps.parquet")
        tel_by[ses] = pd.read_parquet(d / "telemetry.parquet")
        meta_by[ses] = meta
    first_meta = next(iter(meta_by.values()))
    circuit_ref = (reference or {}).get(first_meta["event_slug"])

    # Same anchor choice as the segmenter: the candidate whose position slice
    # covers the lap best settles length and window.
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
            best = (score, ses, ref, frame, c, lap_len_hint)
    if best is None:
        raise ValueError("no candidate lap produced a geometry")
    _, ses, ref, frame, c, lap_len_hint = best

    picks = ensemble_frames(laps_by, tel_by, lap_len_hint)
    if len(picks) < 3:
        raise ValueError(f"only {len(picks)} ensemble laps; the line needs the ensemble")
    ens = E.build_ensemble([f for _, _, f in picks], lap_len_hint, step_m=GRID_STEP_M)
    window = float(window_override_m or c["smooth_window_m"])
    order = int(c.get("poly_order", 2))
    geo = G.build(ens.frame, lap_distance_m=lap_len_hint, step_m=GRID_STEP_M,
                  smooth_window_m=window, poly_order=order)
    grid = geo.distance_m
    n = len(grid)

    # --- elevation, gated
    elev = {"available": False, "used": False, "reason": "no pos_z in the ensemble laps"}
    z_m = np.zeros(n)
    grade = np.zeros(n)
    if "pos_z" in ens.frame.columns and ens.z_spread is not None:
        d_raw = ens.frame["distance_m"].to_numpy(dtype=float)
        z_raw = ens.frame["pos_z"].to_numpy(dtype=float) * geo.metres_per_unit
        finite = np.isfinite(z_raw)
        if finite.sum() >= 20:
            z_raw = np.where(finite, z_raw, np.interp(d_raw, d_raw[finite], z_raw[finite]))
            zs, dz = elevation_profile(d_raw, z_raw, window, order, period_m=float(lap_len_hint))
            z_m = np.interp(grid, d_raw, zs)
            grade = np.interp(grid, d_raw, dz)
            z_m = z_m - float(np.min(z_m))                 # height above the lowest point
            closure = abs(float(z_raw[-1] - z_raw[0]))
            spread_m = ens.z_spread * geo.metres_per_unit
            elev = {
                "available": True,
                "range_m": round(float(np.max(z_m) - np.min(z_m)), 1),
                "closure_m": round(closure, 2),
                "spread_median_m": round(float(np.nanmedian(spread_m)), 2),
                "spread_p90_m": round(float(np.nanquantile(spread_m, 0.9)), 2),
                "grade_max_pct": round(float(np.max(np.abs(grade)) * 100.0), 2),
                "gate": {"closure_max_m": Z_CLOSURE_MAX_M, "spread_max_m": Z_SPREAD_MAX_M},
            }
            ok = closure <= Z_CLOSURE_MAX_M and elev["spread_median_m"] <= Z_SPREAD_MAX_M
            elev["used"] = bool(ok)
            elev["reason"] = ("passes" if ok else
                              f"closure {closure:.1f} m / spread {elev['spread_median_m']:.1f} m "
                              f"outside the gate; profile kept flat")
            if not ok:
                grade = np.zeros(n)

    share = drs_share(picks, ens, grid)
    if share is None:
        share = np.zeros(n)

    line = pd.DataFrame({
        "distance_m": np.round(grid, 2),
        "x_m": np.round(geo.x_m, 3), "y_m": np.round(geo.y_m, 3), "z_m": np.round(z_m, 3),
        "curvature_1pm": geo.curvature_1pm.astype(float),
        "grade": np.round(grade, 6),
        "drs_share": np.round(share, 3),
        "speed_kph": np.round(np.nan_to_num(geo.speed_kph, nan=0.0), 2) if geo.speed_kph is not None else np.zeros(n),
    })
    # DRS zones as contiguous runs where most laps had it open
    open_ = share >= 0.5
    zones = []
    if open_.any():
        edges = np.flatnonzero(np.diff(np.r_[0, open_.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            zones.append({"start_m": round(float(grid[a]), 1),
                          "end_m": round(float(grid[min(b, n - 1)]), 1)})

    meta = {
        "season": first_meta["season"], "event": first_meta["event"],
        "event_slug": first_meta["event_slug"],
        "grid_step_m": GRID_STEP_M, "points": int(n),
        "lap_length_m": round(float(geo.lap_length_m), 1),
        "ensemble_laps": ens.n_laps, "anchor_session": ses,
        "smooth_window_m": window, "poly_order": order,
        "elevation": elev,
        "drs_zones": zones,
        "curvature_vs_track_json": curvature_agreement(track_doc, grid, geo.curvature_1pm),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return line, meta


def export_store(silver: Path, store_dir: Path) -> int:
    """Copy every silver line into the baseline store in its compact form (no rebuild)."""
    n = 0
    for lp in sorted(silver.glob("*/*/line.parquet")):
        meta = json.loads((lp.parent / "line.json").read_text())
        sd = store_dir / lp.parent.parent.name / lp.parent.name
        if not sd.exists():
            continue
        (sd / "line.json").write_text(json.dumps(line_to_store(pd.read_parquet(lp), meta), separators=(",", ":")))
        n += 1
    print(f"exported {n} line(s) to {store_dir}")
    return 0


def run(scope_path: Path, limit: int | None, force: bool, only: set[str] | None,
        verbose: bool, window_m: float | None = None, quiet: bool = False,
        store_dir: Path | None = None) -> int:
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
        if not (out / "track.json").exists():
            continue                                   # not segmented -> nothing to agree with
        if only and not ({slug, f"{season}/{slug}"} & only):
            continue
        if (out / "line.parquet").exists() and not force:
            continue
        todo.append(((season, slug), sessions, out))
    if limit:
        todo = todo[:limit]

    if not quiet:
        print(f"bronze     : {bronze}")
        print(f"silver     : {silver}")
        print(f"to build   : {len(todo)} line(s)")
        print()

    done, failures = [], []
    for i, ((season, slug), sessions, out) in enumerate(todo, 1):
        label = f"{season}/{slug}"
        try:
            track_doc = json.loads((out / "track.json").read_text())
            line, meta = build_line(sessions, cfg, reference, track_doc, window_override_m=window_m)
            line.to_parquet(out / "line.parquet", index=False, compression="zstd")
            (out / "line.json").write_text(json.dumps(meta, indent=1))
            if store_dir is not None:
                # the engine ships with the baseline store, not with silver
                sd = store_dir / str(season) / slug
                if sd.exists():
                    (sd / "line.json").write_text(json.dumps(line_to_store(line, meta), separators=(",", ":")))
            done.append(meta)
            e = meta["elevation"]
            ztxt = (f"z {e['range_m']:>5.1f} m  spread {e['spread_median_m']:.1f}  "
                    f"closure {e['closure_m']:.1f}  {'USED' if e['used'] else 'flat'}"
                    if e.get("available") else "z n/a")
            agree = meta["curvature_vs_track_json"].get("peak_curvature_rel_err_median", float("nan"))
            if not quiet:
                print(f"[{i:>3}/{len(todo)}] {label:<36s} {meta['points']:>5} pts  "
                      f"DRS zones {len(meta['drs_zones'])}  {ztxt}  k-agree {agree:.3f}")
        except Exception as exc:                              # noqa: BLE001
            failures.append({"circuit": label, "error": f"{type(exc).__name__}: {exc}"[:240]})
            print(f"[{i:>3}/{len(todo)}] {label:<36s} FAILED  {type(exc).__name__}: {exc}"[:160],
                  file=sys.stderr)
            if verbose:
                traceback.print_exc()

    if done and not quiet:
        used = [d for d in done if d["elevation"].get("used")]
        avail = [d for d in done if d["elevation"].get("available")]
        print()
        print("ELEVATION gate")
        print(f"  circuits with a Z channel : {len(avail)} / {len(done)}")
        print(f"  passing the gate          : {len(used)} / {len(done)}")
        for d in done:
            e = d["elevation"]
            if e.get("available") and not e.get("used"):
                print(f"    flat: {d['season']} {d['event']:<28s} {e['reason']}")
        ag = [d["curvature_vs_track_json"].get("peak_curvature_rel_err_median") for d in done]
        ag = [a for a in ag if a is not None]
        if ag:
            print(f"  curvature vs track.json   : median rel. error {np.median(ag):.3f}, "
                  f"worst {max(ag):.3f}")

    silver.mkdir(parents=True, exist_ok=True)
    (silver / "line_report.json").write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lines": done, "failures": failures}, indent=1))
    if not quiet:
        print()
        print(f"built {len(done)} line(s), {len(failures)} failure(s)")
    return 1 if failures and not done else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", type=Path, default=Path("configs/scope.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="rebuild lines that already exist")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated event slugs (or season/slug) to build")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--window", type=float, default=None,
                    help="override the curvature smoothing window (metres) for every circuit")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--store", type=Path, default=Path("../data/artifacts/baselines"),
                    help="baseline store to receive a compact line.json per event ('' to skip)")
    ap.add_argument("--export-only", action="store_true",
                    help="only copy existing silver lines into the baseline store")
    a = ap.parse_args()
    only = {x.strip() for x in a.only.split(",")} if a.only else None
    store = a.store if str(a.store) not in ("", ".") else None
    if a.export_only:
        scope = paths.load_scope(a.scope)
        sys.exit(export_store(silver_dir(a.scope, scope), store or Path("../data/artifacts/baselines")))
    sys.exit(run(a.scope, a.limit, a.force, only, a.verbose, window_m=a.window, quiet=a.quiet,
                 store_dir=store))


if __name__ == "__main__":
    main()
