"""P1-1 — session loading, on top of the warm cache.

Reads only. Every session this touches should already be in the cache from
`warm_cache.py`; with the cache populated this makes zero network calls,
which is the property the P1 gate checks.

Telemetry is loaded per lap and kept at its RAW sampling — no uniform
resampling. Car data arrives at a fixed 240 ms period, so spacing is a
function of speed (4 m in a hairpin, 20 m at 300 km/h); a uniform grid would
interpolate detail that was never measured on the straights, where the
spacing is widest. Features integrate raw samples inside each segment
instead (docs/recon/DECISIONS.md D1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Channels we persist. `Source` is dropped (constant), `Z` kept for elevation.
CAR_CHANNELS = ["Speed", "Throttle", "Brake", "nGear", "RPM", "DRS"]
POS_CHANNELS = ["X", "Y", "Z"]


@dataclass
class LapTelemetry:
    """One lap's raw telemetry plus the quality metrics the filters need."""

    lap_uid: str
    frame: pd.DataFrame
    n_samples: int
    lap_distance_m: float
    max_gap_m: float
    median_gap_m: float
    p95_gap_m: float
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.n_samples >= 10


@dataclass
class LoadedSession:
    season: int
    event: str
    session: str
    official_name: str
    circuit: str
    event_date: str
    laps: pd.DataFrame
    weather: pd.DataFrame
    handle: object = field(repr=False, default=None)   # the fastf1 Session


def enable_cache(ff1, cache_path: Path) -> None:
    cache_path.mkdir(parents=True, exist_ok=True)
    ff1.Cache.enable_cache(str(cache_path))


def load_session(ff1, season: int, event: str, ses: str) -> LoadedSession:
    s = ff1.get_session(season, event, ses)
    s.load(laps=True, telemetry=True, weather=True, messages=True)

    laps = s.laps
    if laps is None or not len(laps):
        raise ValueError(f"no laps: {season} {event} {ses}")

    weather = s.weather_data
    if weather is None:
        weather = pd.DataFrame()

    ev = getattr(s, "event", None)
    return LoadedSession(
        season=season,
        event=event,
        session=ses,
        official_name=str(getattr(s, "name", ses)),
        circuit=str(getattr(ev, "Location", "") or "") if ev is not None else "",
        event_date=str(getattr(ev, "EventDate", "") or "") if ev is not None else "",
        laps=laps.copy(),
        weather=weather.copy(),
        handle=s,
    )


def lap_uid(season: int, event_slug: str, ses: str, driver: str, lap_number: float) -> str:
    n = int(lap_number) if pd.notna(lap_number) else -1
    return f"{season}_{event_slug}_{ses}_{driver}_{n:03d}"


def extract_lap_telemetry(lap, uid: str) -> LapTelemetry:
    """Raw car + position channels on a shared distance axis for one lap.

    `add_distance()` integrates differential distance, so FastF1's own docs
    warn to apply it per lap rather than across a session — which is exactly
    what we do here.
    """
    empty = pd.DataFrame()
    try:
        car = lap.get_car_data()
        if car is None or len(car) < 10:
            return LapTelemetry(uid, empty, 0, 0.0, 0.0, 0.0, 0.0, "empty car data")

        car = car.add_distance()

        # Position data rides a separate stream; merge it onto the car samples
        # by time rather than assuming the two share an index.
        try:
            pos = lap.get_pos_data()
        except Exception:                                   # noqa: BLE001
            pos = None

        frame = car[["Date", "SessionTime", "Distance"] +
                    [c for c in CAR_CHANNELS if c in car.columns]].copy()

        if pos is not None and len(pos) and "Date" in pos.columns:
            keep = ["Date"] + [c for c in POS_CHANNELS if c in pos.columns]
            frame = pd.merge_asof(
                frame.sort_values("Date"),
                pos[keep].sort_values("Date"),
                on="Date", direction="nearest",
                tolerance=pd.Timedelta("200ms"),
            )
        else:
            for c in POS_CHANNELS:
                frame[c] = np.nan

        dist = pd.to_numeric(frame["Distance"], errors="coerce")
        step = dist.diff().dropna()
        step = step[step >= 0]                              # guard against wrap

        if not len(step):
            return LapTelemetry(uid, empty, len(frame), 0.0, 0.0, 0.0, 0.0,
                                "no usable distance axis")

        frame.insert(0, "lap_uid", uid)
        return LapTelemetry(
            lap_uid=uid,
            frame=frame,
            n_samples=int(len(frame)),
            lap_distance_m=float(dist.iloc[-1]),
            max_gap_m=float(step.max()),
            median_gap_m=float(step.median()),
            p95_gap_m=float(step.quantile(0.95)),
        )
    except Exception as exc:                                # noqa: BLE001
        return LapTelemetry(uid, empty, 0, 0.0, 0.0, 0.0, 0.0,
                            f"{type(exc).__name__}: {exc}"[:200])


def extract_all(session: LoadedSession, laps: pd.DataFrame, event_slug: str) -> dict[str, LapTelemetry]:
    """Telemetry for every lap in `laps`, keyed by lap_uid.

    Called once; the result feeds both the gap filter and the bronze writer,
    so no lap's telemetry is ever fetched twice.
    """
    out: dict[str, LapTelemetry] = {}
    for idx in range(len(laps)):
        lap = laps.iloc[idx]
        uid = lap_uid(session.season, event_slug, session.session,
                      str(lap["Driver"]), lap.get("LapNumber", -1))
        out[uid] = extract_lap_telemetry(laps.iloc[[idx]].iloc[0], uid)
    return out
