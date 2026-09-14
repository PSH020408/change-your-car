"""Background cache warming — pull every session's raw data into the FastF1 cache.

Why this runs on its own track
------------------------------
FastF1 rate-limits live requests but explicitly does NOT count cache hits
against those limits. So the expensive, slow, throttled part of this project
(fetching raw data) is completely separable from the design work that
consumes it. This job fills the cache once, in the background, and every
later pipeline run reads from disk at full speed.

Properties this job needs, and has:
  * resumable   — a JSON ledger records every completed session; re-running
                  skips them, so a crash or a laptop closing costs nothing
  * polite      — exponential backoff on RateLimitExceededError, a floor
                  delay between sessions, never a tight retry loop
  * measured    — Cache.get_cache_info() is sampled after every session, so
                  the ledger doubles as the real answer to "how big is this
                  going to get" instead of a guess
  * tolerant    — a session that fails (not yet run, no telemetry, API hole)
                  is recorded with its error and skipped, never fatal

Usage
-----
    python -m pipeline.ingest.warm_cache --scope configs/scope.yaml
    python -m pipeline.ingest.warm_cache --scope configs/scope.yaml --dry-run
    python -m pipeline.ingest.warm_cache --scope configs/scope.yaml --seasons 2024

Long run, detached:
    nohup python -m pipeline.ingest.warm_cache --scope configs/scope.yaml \
        > ../data/cache/warm.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

LEDGER_NAME = "_warm_ledger.json"

# Politeness knobs. FastF1 throttles on its own; these keep us well clear of
# the hard limit rather than discovering it.
BASE_DELAY_S = 1.5
BACKOFF_START_S = 60.0
BACKOFF_MAX_S = 1800.0
MAX_RETRIES = 6


@dataclass
class Entry:
    key: str
    season: int
    event: str
    session: str
    status: str                 # done | failed | skipped
    cache_bytes_total: int = 0  # cache size AFTER this session
    cache_bytes_delta: int = 0  # what this session cost
    laps: int = 0
    drivers: int = 0
    seconds: float = 0.0
    error: str = ""
    ts: str = ""


# --------------------------------------------------------------------------- io
def load_scope(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def ledger_path(cache_dir: Path) -> Path:
    return cache_dir / LEDGER_NAME


def read_ledger(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        with path.open() as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        # a half-written ledger must never block a resume
        backup = path.with_suffix(".corrupt")
        path.replace(backup)
        print(f"[warn] ledger unreadable, moved to {backup}", file=sys.stderr)
        return {}


def write_ledger(path: Path, ledger: dict[str, dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump(ledger, fh, indent=1, sort_keys=True)
    tmp.replace(path)          # atomic: a kill mid-write cannot corrupt it


def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} TB"



def resolve_cache_dir(scope_path: Path, scope: dict) -> Path:
    """Cache location, read from config rather than walked from __file__.

    The previous version did `scope_path.parent.parent.parent`, which
    saturates at "." for a relative path and silently put the cache under
    backend/ instead of the repo root. The path is now explicit in
    scope.yaml (docs/recon/DECISIONS.md D9).
    """
    cfg = (scope.get("paths") or {}).get("cache_dir", "data/cache")
    base = scope_path.resolve().parent.parent          # -> backend/
    return (base / cfg).resolve() if not Path(cfg).is_absolute() else Path(cfg)

# ---------------------------------------------------------------------- worklist
def build_worklist(ff1, seasons: list[int], sessions: list[str]) -> list[tuple[int, str, str]]:
    """Expand seasons into concrete (season, event, session) triples.

    Uses the real event schedule rather than a hardcoded calendar, so sprint
    weekends, cancelled rounds and mid-season additions are handled by data.
    Events dated in the future are dropped — they have nothing to download.
    """
    now = datetime.now(timezone.utc)
    work: list[tuple[int, str, str]] = []

    for season in seasons:
        try:
            sched = ff1.get_event_schedule(season, include_testing=False)
        except Exception as exc:                      # noqa: BLE001
            print(f"[warn] {season}: schedule unavailable ({exc})", file=sys.stderr)
            continue

        for _, ev in sched.iterrows():
            event_date = ev.get("EventDate")
            if event_date is not None and hasattr(event_date, "to_pydatetime"):
                dt = event_date.to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt > now:
                    continue                          # hasn't happened yet

            name = str(ev["EventName"])
            held = {str(ev.get(f"Session{i}", "")) for i in range(1, 6)}
            for ident in sessions:
                # Session identifiers FastF1 accepts vs. names in the schedule.
                alias = {
                    "FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3",
                    "Q": "Qualifying", "SQ": "Sprint Qualifying",
                    "S": "Sprint", "R": "Race",
                }.get(ident, ident)
                # "Sprint Shootout" was the 2023 name for sprint qualifying.
                if alias in held or (ident == "SQ" and "Sprint Shootout" in held):
                    work.append((season, name, ident))

    return work


# ------------------------------------------------------------------------- fetch
def warm_one(ff1, season: int, event: str, ident: str, load_cfg: dict) -> tuple[int, int]:
    """Load one session so its raw responses land in the cache.

    Returns (lap_count, driver_count) — cheap signal that the session really
    carried data rather than returning an empty shell.
    """
    session = ff1.get_session(season, event, ident)
    session.load(
        laps=load_cfg.get("laps", True),
        telemetry=load_cfg.get("telemetry", True),
        weather=load_cfg.get("weather", True),
        messages=load_cfg.get("messages", True),
    )
    laps = session.laps
    n_laps = 0 if laps is None else int(len(laps))
    n_drv = 0 if laps is None or not n_laps else int(laps["Driver"].nunique())
    return n_laps, n_drv


def run(scope_path: Path, only_seasons: list[int] | None, dry_run: bool, limit: int | None) -> int:
    import fastf1 as ff1

    scope = load_scope(scope_path)
    warm_cfg = scope["warm"]
    seasons = only_seasons or warm_cfg["seasons"]
    sessions = warm_cfg["sessions"]
    load_cfg = warm_cfg.get("load", {})

    cache_dir = resolve_cache_dir(scope_path, scope)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ff1.Cache.enable_cache(str(cache_dir))

    lpath = ledger_path(cache_dir)
    ledger = read_ledger(lpath)

    work = build_worklist(ff1, seasons, sessions)
    pending = [w for w in work if f"{w[0]}|{w[1]}|{w[2]}" not in ledger]
    if limit:
        pending = pending[:limit]

    print(f"cache      : {cache_dir}")
    print(f"seasons    : {seasons}")
    print(f"sessions   : {sessions}")
    print(f"discovered : {len(work)} sessions")
    print(f"already in : {len(work) - len([w for w in work if f'{w[0]}|{w[1]}|{w[2]}' not in ledger])}")
    print(f"to fetch   : {len(pending)}")

    if dry_run:
        for season, event, ident in pending:
            print(f"  would fetch  {season}  {event:<34s}  {ident}")
        return 0

    prev_total = ff1.Cache.get_cache_info()[1] or 0
    failures = 0

    for i, (season, event, ident) in enumerate(pending, 1):
        key = f"{season}|{event}|{ident}"
        started = time.time()
        delay = BACKOFF_START_S
        entry: Entry | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                n_laps, n_drv = warm_one(ff1, season, event, ident, load_cfg)
                total = ff1.Cache.get_cache_info()[1] or prev_total
                entry = Entry(
                    key=key, season=season, event=event, session=ident,
                    status="done",
                    cache_bytes_total=int(total),
                    cache_bytes_delta=int(total - prev_total),
                    laps=n_laps, drivers=n_drv,
                    seconds=round(time.time() - started, 1),
                    ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                prev_total = total
                break

            except ff1.RateLimitExceededError:
                if attempt == MAX_RETRIES:
                    entry = Entry(key=key, season=season, event=event, session=ident,
                                  status="failed", error="rate limit, retries exhausted",
                                  seconds=round(time.time() - started, 1),
                                  ts=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                    break
                wait = min(delay, BACKOFF_MAX_S) * (0.8 + 0.4 * random.random())
                print(f"  [rate-limit] sleeping {wait:.0f}s  (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                delay *= 2

            except Exception as exc:                   # noqa: BLE001
                # Missing session, no telemetry for that year, API hole — record
                # and move on. One bad session must not stop a 700-session job.
                entry = Entry(
                    key=key, season=season, event=event, session=ident,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}"[:300],
                    seconds=round(time.time() - started, 1),
                    ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                break

        assert entry is not None
        ledger[key] = asdict(entry)
        write_ledger(lpath, ledger)

        if entry.status == "done":
            print(f"[{i:>4}/{len(pending)}] {season} {event:<34s} {ident:<3s} "
                  f"{entry.laps:>4d} laps  {entry.drivers:>2d} drv  "
                  f"+{human(entry.cache_bytes_delta):>9s}  "
                  f"total {human(entry.cache_bytes_total)}  {entry.seconds:.0f}s")
        else:
            failures += 1
            print(f"[{i:>4}/{len(pending)}] {season} {event:<34s} {ident:<3s} "
                  f"FAILED  {entry.error}", file=sys.stderr)

        time.sleep(BASE_DELAY_S)

    done = [e for e in ledger.values() if e["status"] == "done"]
    total = ff1.Cache.get_cache_info()[1] or 0
    print()
    print(f"sessions cached : {len(done)}")
    print(f"failures        : {failures}")
    print(f"cache size      : {human(total)}")
    if done:
        avg = sum(e["cache_bytes_delta"] for e in done) / len(done)
        print(f"avg per session : {human(int(avg))}")
        remaining = len(work) - len(done)
        if remaining > 0:
            print(f"projected total : {human(int(total + avg * remaining))} "
                  f"({remaining} sessions left)")
    return 1 if failures and not done else 0


def main() -> None:
    p = argparse.ArgumentParser(description="Warm the FastF1 cache in the background.")
    p.add_argument("--scope", type=Path, default=Path("configs/scope.yaml"))
    p.add_argument("--seasons", type=int, nargs="*", default=None)
    p.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    p.add_argument("--limit", type=int, default=None, help="fetch at most N sessions")
    args = p.parse_args()
    try:
        sys.exit(run(args.scope, args.seasons, args.dry_run, args.limit))
    except KeyboardInterrupt:
        print("\ninterrupted — ledger saved, re-run to resume", file=sys.stderr)
        sys.exit(130)
    except Exception:                                   # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
