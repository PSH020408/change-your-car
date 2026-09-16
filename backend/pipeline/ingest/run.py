"""P1 — ingestion orchestrator: cache -> data/bronze.

Only touches sessions that the warm job has ALREADY cached, read from its
ledger. Two consequences worth stating:

  * this stage makes zero network calls, which is the P1 gate; and
  * it can run right now, against whatever has been downloaded so far, while
    `warm_cache.py` keeps filling the rest in the background. Ingestion never
    waits on the download.

Re-running is safe: a session with a `session.json` is skipped unless
`--force` is given.

Usage
-----
    python -m pipeline.ingest.run --scope configs/scope.yaml
    python -m pipeline.ingest.run --scope configs/scope.yaml --pilot
    python -m pipeline.ingest.run --scope configs/scope.yaml --limit 3 --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.ingest import filters, loader, metadata, paths, weather, writer


# --------------------------------------------------------------------- lazy
class LazyTelemetry:
    """Extract a lap's telemetry only when a filter actually asks for it.

    The gap filter is the first step that needs telemetry, and by then the
    cheap filters have already removed roughly a fifth of the laps. Extracting
    eagerly would parse those for nothing.
    """

    def __init__(self, laps: pd.DataFrame):
        self._rows = {str(u): i for i, u in zip(laps.index, laps["lap_uid"])}
        self._laps = laps
        self._cache: dict[str, loader.LapTelemetry] = {}

    def get(self, uid: str):
        uid = str(uid)
        if uid in self._cache:
            return self._cache[uid]
        idx = self._rows.get(uid)
        if idx is None:
            return None
        t = loader.extract_lap_telemetry(self._laps.loc[idx], uid)
        self._cache[uid] = t
        return t

    def frames_for(self, uids) -> list[pd.DataFrame]:
        out = []
        for u in uids:
            t = self.get(u)
            if t is not None and t.ok:
                out.append(t.frame)
        return out

    def quality_frame(self, uids) -> pd.DataFrame:
        rows = []
        for u in uids:
            t = self.get(u)
            rows.append({
                "lap_uid": str(u),
                "n_samples": t.n_samples if t else 0,
                "lap_distance_m": t.lap_distance_m if t else 0.0,
                "max_gap_m": t.max_gap_m if t else 0.0,
                "median_gap_m": t.median_gap_m if t else 0.0,
                "p95_gap_m": t.p95_gap_m if t else 0.0,
                "max_gap_s": t.max_gap_s if t else 0.0,
                "median_gap_s": t.median_gap_s if t else 0.0,
                "missed_samples_worst": t.missed_samples_worst if t else 0,
                "worst_gap_at_frac": t.worst_gap_at_frac if t else 0.0,
                "implied_speed_kph": getattr(t, "implied_speed_kph", 0.0) if t else 0.0,
                "telemetry_quality": (filters.telemetry_quality_band(t.max_gap_s)
                                      if t and t.ok else "missing"),
            })
        return pd.DataFrame(rows)


# ------------------------------------------------------------------ prepare
def _secs(col) -> pd.Series:
    return pd.to_timedelta(col, errors="coerce").dt.total_seconds()


def prepare_laps(laps: pd.DataFrame, sess: loader.LoadedSession,
                 event_slug: str, cmap: metadata.ChassisMap,
                 cond_cfg: dict, excluded_status: list[str] | None = None) -> pd.DataFrame:
    """Derive the columns the filters and the bronze schema need."""
    df = laps.reset_index(drop=True).copy()

    df["season"] = sess.season
    df["event"] = sess.event
    df["event_slug"] = event_slug
    df["session"] = sess.session
    df["circuit"] = sess.circuit

    df["lap_uid"] = [
        loader.lap_uid(sess.season, event_slug, sess.session,
                       str(d), n)
        for d, n in zip(df.get("Driver", pd.Series(dtype=str)),
                        df.get("LapNumber", pd.Series(dtype=float)))
    ]

    for src, dst in (("LapTime", "lap_time_s"),
                     ("Sector1Time", "sector1_s"),
                     ("Sector2Time", "sector2_s"),
                     ("Sector3Time", "sector3_s"),
                     ("LapStartTime", "lap_start_s")):
        df[dst] = _secs(df[src]) if src in df else pd.NA

    # Speed traps: preserved verbatim, with the null flagged rather than
    # filled. Imputation needs segments, which arrive in P2 (DECISIONS.md D6).
    for trap in ("SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"):
        flag = f"speed_{trap.replace('Speed', '').lower()}_missing"
        df[flag] = df[trap].isna() if trap in df else True

    df["condition"] = filters.tag_conditions(df, cond_cfg)
    # traffic at the line, from EVERY crossing in the session, before any
    # lap is filtered away
    df[["gap_ahead_s", "gap_behind_s"]] = filters.gap_to_neighbours(df)
    df["track_status_flag"] = filters.tag_track_status(
        df, excluded_status or ["4", "5", "6", "7"])
    return metadata.annotate(df, sess.season, cmap)


# --------------------------------------------------------------------- work
def cached_sessions(cache: Path) -> list[tuple[int, str, str]]:
    """Sessions the warm job has already downloaded, from its ledger."""
    ledger = cache / "_warm_ledger.json"
    if not ledger.exists():
        return []
    try:
        data = json.loads(ledger.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return [(e["season"], e["event"], e["session"])
            for e in data.values() if e.get("status") == "done"]


def build_worklist(scope: dict, cache: Path, pilot: bool, all_sessions: bool
                   ) -> tuple[list[tuple[int, str, str]], dict]:
    have = cached_sessions(cache)
    stats = {"cached": len(have)}

    wanted_sessions = (None if all_sessions
                       else set(scope.get("train", {}).get("sessions",
                                                           ["Q", "SQ", "R", "S"])))
    era = set(scope["era"]["seasons"])

    work = []
    for season, event, ses in have:
        if season not in era:
            continue
        if wanted_sessions and ses not in wanted_sessions:
            continue
        if pilot:
            p = scope.get("pilot", {})
            if season not in set(p.get("seasons", [])):
                continue
            if p.get("events") and not any(
                    paths.slug(e) in paths.slug(event) or paths.slug(event) in paths.slug(e)
                    for e in p["events"]):
                continue
        work.append((season, event, ses))

    stats["selected"] = len(work)
    stats["train_sessions"] = sorted(wanted_sessions) if wanted_sessions else "all"
    return sorted(work), stats


# ---------------------------------------------------------------------- one
def _driver_lookup_from_siblings(bronze: Path, season: int, event_slug: str, skip_session: str) -> dict[str, dict]:
    """car number -> {driver, team} from any already-ingested session of the same weekend."""
    out: dict[str, dict] = {}
    for lp in sorted((bronze / str(season) / event_slug).glob("*/laps.parquet")):
        if lp.parent.name == skip_session:
            continue
        try:
            t = pd.read_parquet(lp, columns=["driver", "driver_number", "team_raw"]).dropna()
        except Exception:                                   # noqa: BLE001
            continue
        for d, n, tm in t.drop_duplicates("driver_number").itertuples(index=False):
            out.setdefault(str(n).strip(), {"driver": str(d), "team": str(tm)})
    return out


def ingest_one(ff1, scope_path: Path, scope: dict, cmap: metadata.ChassisMap,
               season: int, event: str, ses: str, verbose: bool) -> dict:
    sess = loader.load_session(ff1, season, event, ses)
    event_slug = paths.slug(sess.event)

    laps = sess.laps
    if "Team" in laps:
        # 2024 Azerbaijan R: FastF1 could not fetch the driver list ("Generating
        # minimal driver list from timing data") so EVERY lap had Team == "" and
        # Driver == car number. The sibling sessions of the same weekend in
        # bronze know car number -> abbreviation / team; borrow from them.
        blank = laps["Team"].isna() | (laps["Team"].astype(str).str.strip() == "")
        if bool(blank.any()):
            fill = _driver_lookup_from_siblings(paths.bronze_dir(scope_path, scope), season, event_slug, ses)
            if fill and "DriverNumber" in laps:
                laps = laps.copy()
                num = laps["DriverNumber"].astype(str).str.strip()
                laps.loc[blank, "Team"] = num[blank].map(lambda n: fill.get(n, {}).get("team"))
                if "Driver" in laps:
                    # the minimal list leaves Driver EMPTY (not the car number as first assumed):
                    # 973 laps with the same blank abbreviation collapsed into one lap_uid per lap
                    # number and the telemetry merge multiplied the session 18x. Fill anything
                    # that is not a proper 3-letter abbreviation.
                    abbr = laps["Driver"].astype(str).str.strip()
                    bad = abbr.isna() | ~abbr.str.fullmatch(r"[A-Z]{3}")
                    laps.loc[bad, "Driver"] = num[bad].map(lambda n: fill.get(n, {}).get("driver"))
                still = laps["Team"].isna() | (laps["Team"].astype(str).str.strip() == "")
                print(f"    note: {int(blank.sum())} lap row(s) had no team; {int(blank.sum() - still.sum())} "
                      f"recovered from sibling sessions, {int(still.sum())} dropped")
                blank = still
            else:
                print(f"    note: {int(blank.sum())} lap row(s) with no team name dropped (no sibling session to borrow from)")
            laps = laps[~blank]
    raw_teams = sorted(laps["Team"].astype(str).unique()) if "Team" in laps else []
    unmapped = cmap.validate_coverage(season, raw_teams)
    if unmapped:
        raise ValueError(
            f"chassis map has no entry for {season}: {unmapped}. "
            f"Add them to configs/chassis.yaml — an unmapped team writes a null "
            f"chassis into bronze and becomes a missing one-hot at training time.")

    df = prepare_laps(laps, sess, event_slug, cmap, scope.get("conditions", {}),
                      scope["lap_filters"].get("exclude_track_status"))
    dup = int(df["lap_uid"].duplicated().sum())
    if dup:
        raise ValueError(f"{dup} duplicate lap_uid(s) — driver abbreviations are missing or not unique; "
                         f"refusing to write a session whose merges would multiply rows")
    tel = LazyTelemetry(df)

    kept, report = filters.apply_chain(  # noqa: E501
        laps=df,
        telemetry=tel,
        uids_for=lambda d: d["lap_uid"],
        cfg=scope["lap_filters"],
        key=f"{season}|{event}|{ses}",
        fastf1_laps=sess.laps,
    )

    if verbose:
        print(report.render())

    if len(kept):
        quality = tel.quality_frame(kept["lap_uid"])
        kept = kept.merge(quality, on="lap_uid", how="left")
        kept, wdiag = weather.merge_onto_laps(kept, sess.weather)
    else:
        wdiag = {"joined": False, "reason": "no laps survived the filter chain"}

    frames = tel.frames_for(kept["lap_uid"]) if len(kept) else []
    for f in frames:
        if "SessionTime" in f.columns:
            f["session_time_s"] = _secs(f["SessionTime"])

    out_dir = paths.session_dir(scope_path, scope, season, sess.event, ses)
    meta = writer.write_session(
        out_dir, kept, frames,
        meta={
            "key": f"{season}|{event}|{ses}",
            "season": season, "event": sess.event, "event_slug": event_slug,
            "session": ses, "official_name": sess.official_name,
            "circuit": sess.circuit, "event_date": sess.event_date,
            "drivers": int(kept["Driver"].nunique()) if len(kept) else 0,
            "conditions": {str(k): int(v) for k, v in kept["condition"].value_counts().items()}
                          if len(kept) else {},
            "filter_report": report.to_dict(),
            "weather": wdiag,
        })
    return meta


# ---------------------------------------------------------------------- main
def run(scope_path: Path, pilot: bool, all_sessions: bool,
        limit: int | None, force: bool, verbose: bool, only: set[str] | None = None) -> int:
    import fastf1 as ff1

    scope = paths.load_scope(scope_path)
    cache = paths.cache_dir(scope_path, scope)
    bronze = paths.bronze_dir(scope_path, scope)
    cmap = metadata.load_chassis_map(scope_path.parent / "chassis.yaml")

    loader.enable_cache(ff1, cache)
    ff1.Cache.offline_mode(True)        # P1 gate: no network calls, ever

    work, stats = build_worklist(scope, cache, pilot, all_sessions)

    todo = []
    for season, event, ses in work:
        d = paths.session_dir(scope_path, scope, season, event, ses)
        if only:
            # "2024/azerbaijan_grand_prix/R" — a named redo, always forced
            if f"{season}/{paths.slug(event)}/{ses}" not in only:
                continue
        elif (d / "session.json").exists() and not force:
            continue
        todo.append((season, event, ses))
    if limit:
        todo = todo[:limit]

    print(f"cache      : {cache}")
    print(f"bronze     : {bronze}")
    print(f"cached     : {stats['cached']} sessions in ledger")
    print(f"selected   : {stats['selected']} (sessions: {stats['train_sessions']})")
    print(f"to ingest  : {len(todo)}  (already done are skipped; --force to redo)")
    print()

    entries, failures = [], []
    for i, (season, event, ses) in enumerate(todo, 1):
        label = f"{season} {event} {ses}"
        try:
            meta = ingest_one(ff1, scope_path, scope, cmap, season, event, ses, verbose)
            entries.append(meta)
            fr = meta["filter_report"]
            print(f"[{i:>3}/{len(todo)}] {label:<44s} "
                  f"{fr['final']:>4}/{fr['raw']:<4} laps ({fr['yield_pct']:>5.1f}%)  "
                  f"{meta['telemetry_rows']:>7} samples  "
                  f"{(meta['laps_bytes'] + meta['telemetry_bytes'])/1e6:>6.1f} MB")
        except Exception as exc:                            # noqa: BLE001
            failures.append({"key": f"{season}|{event}|{ses}",
                             "error": f"{type(exc).__name__}: {exc}"[:300]})
            print(f"[{i:>3}/{len(todo)}] {label:<44s} FAILED  "
                  f"{type(exc).__name__}: {exc}"[:160], file=sys.stderr)
            if verbose:
                traceback.print_exc()

    if entries:
        writer.update_manifest(bronze, entries)

    # ---- run report: where the laps went, aggregated across sessions -----
    agg: dict[str, dict] = {}
    for m in entries:
        for s in m["filter_report"]["steps"]:
            a = agg.setdefault(s["name"], {"kept": 0, "removed": 0})
            a["kept"] += s["kept"]
            a["removed"] += s["removed"]

    raw_total = sum(m["filter_report"]["raw"] for m in entries)
    print()
    if entries:
        print(f"{'FILTER STEP':<18}{'kept':>8}{'removed':>9}{'% of raw':>10}")
        print(f"{'raw':<18}{raw_total:>8}{'':>9}{'100.0%':>10}")
        for name, a in agg.items():
            pct = 100.0 * a["kept"] / raw_total if raw_total else 0
            note = "   <-- removed nothing" if a["removed"] == 0 else ""
            print(f"{name:<18}{a['kept']:>8}{a['removed']:>9}{pct:>9.1f}%{note}")

    # ---- telemetry gap: what would each threshold have cost? -------------
    sweeps = [m["filter_report"]["diagnostics"].get("telemetry_gap", {})
              for m in entries]
    sweeps = [d for d in sweeps if d.get("threshold_sweep_laps_kept")]
    if sweeps:
        total_eval = sum(d["laps_evaluated"] for d in sweeps)
        print()
        print("TELEMETRY GAP — threshold sweep (set the config from this table)")
        print(f"  laps evaluated: {total_eval}")
        keys = list(sweeps[0]["threshold_sweep_laps_kept"])
        print(f"  {'threshold':<12}{'kept':>8}{'% kept':>9}")
        for k in keys:
            kept = sum(d["threshold_sweep_laps_kept"].get(k, 0) for d in sweeps)
            pct = 100.0 * kept / total_eval if total_eval else 0
            mark = "  <-- current" if k == f"{scope['lap_filters'].get('max_telemetry_gap_s', 1.0)}s" else ""
            print(f"  {k:<12}{kept:>8}{pct:>8.1f}%{mark}")
        p50 = [d.get("max_gap_s_p50") for d in sweeps if d.get("max_gap_s_p50") is not None]
        p95 = [d.get("max_gap_s_p95") for d in sweeps if d.get("max_gap_s_p95") is not None]
        if p50:
            print(f"  worst-gap per lap: p50 {min(p50):.2f}-{max(p50):.2f} s, "
                  f"p95 {min(p95):.2f}-{max(p95):.2f} s  (nominal period 0.24 s)")
        bands: dict[str, int] = {}
        for d in sweeps:
            for k, v in (d.get("quality_bands") or {}).items():
                bands[k] = bands.get(k, 0) + v
        if bands:
            tot = sum(bands.values())
            print("  quality tag: " + "  ".join(
                f"{k}={v} ({100.0*v/tot:.0f}%)"
                for k, v in sorted(bands.items(), key=lambda kv: -kv[1])))

        # Distance axis has to be physically possible — everything downstream
        # of P2 integrates along it.
        bad = sum(d.get("laps_with_impossible_implied_speed", 0) for d in sweeps)
        neg = sum(d.get("negative_distance_steps_total", 0) for d in sweeps)
        imax = [d.get("implied_speed_kph_max") for d in sweeps
                if d.get("implied_speed_kph_max") is not None]
        print()
        print("DISTANCE AXIS integrity")
        print(f"  implied speed across worst gap, max : "
              f"{max(imax) if imax else 'n/a'} km/h   (F1 record is ~372 km/h)")
        print(f"  laps implying >400 km/h             : {bad}")
        print(f"  negative distance steps             : {neg}")
        if bad or neg:
            print("  VERDICT: the distance axis has discontinuities — P2 segmentation "
                  "integrates along it, so fix before proceeding.")
        else:
            print("  VERDICT: distance axis is consistent with the time axis.")

    # ---- D4: is the track-status filter working? --------------------------
    ts_removed = agg.get("track status", {}).get("removed", 0)
    ts = [m["filter_report"]["diagnostics"].get("track_status", {}) for m in entries]
    broken = [v for v in ts if v.get("broken")]
    disagree = [v for v in ts if v.get("agrees_with_fastf1") is False]
    print()
    print("D4 — track status filter  (runs FIRST, on the full session)")
    print(f"  laps removed across all sessions : {ts_removed}")
    print(f"  sessions carrying an excluded code but removing nothing : {len(broken)}")
    print(f"  sessions where ours != FastF1's  : {len(disagree)}")
    if broken:
        print("  VERDICT: BROKEN — " + "; ".join(v["verdict"] for v in broken[:3]))
    elif ts_removed == 0:
        others: dict[str, int] = {}
        for v in ts:
            for k, n in (v.get("codes_present_but_not_excluded") or {}).items():
                others[k] = others.get(k, 0) + n
        print("  VERDICT: correct — no lap ran under an excluded code."
              + (f" Present but intentionally kept: {others}" if others else ""))
    else:
        print(f"  VERDICT: working — removed {ts_removed} lap(s).")

    report_path = bronze / "ingest_report.json"
    bronze.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sessions_ingested": len(entries),
        "failures": failures,
        "aggregate_filter_steps": agg,
        "raw_laps": raw_total,
        "sessions": [m["filter_report"] for m in entries],
    }, indent=1, default=str))

    print()
    print(f"ingested {len(entries)} session(s), {len(failures)} failure(s)")
    print(f"report -> {report_path}")
    return 1 if failures and not entries else 0


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest cached FastF1 sessions into bronze.")
    p.add_argument("--scope", type=Path, default=Path("configs/scope.yaml"))
    p.add_argument("--pilot", action="store_true", help="restrict to the pilot scope")
    p.add_argument("--all-sessions", action="store_true",
                   help="include practice (default: train sessions only)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true", help="re-ingest sessions already written")
    p.add_argument("--verbose", "-v", action="store_true", help="print the filter chain per session")
    p.add_argument("--only", type=str, default=None,
                   help="comma-separated season/event_slug/session keys to (re)ingest, forcing a redo")
    a = p.parse_args()
    try:
        only = {x.strip() for x in a.only.split(",") if x.strip()} if a.only else None
        sys.exit(run(a.scope, a.pilot, a.all_sessions, a.limit, a.force, a.verbose, only=only))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:                                       # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
