"""Was the driver actually trying on this lap?

The conditions-matched pace gate (DECISIONS.md D3) answers "was this lap
normal for its moment", which is the right question for an anomaly filter and
the wrong one for a SETUP simulator. 2024 Monaco proved it: after the lap-one
red flag the race became a procession, so the rolling reference tracked the
cruise and 89% of the race passed the gate — a training set of laps nobody was
pushing on, with nothing in the data to say so.

Effort is that missing signal. It is deliberately built from what the driver
DID (throttle, brake, speed held) rather than from lap time, because lap time
already carries the setup effect we are trying to predict. Using pace to
decide which laps show the setup's effect would be circular.

Tagged, never filtered — the fourth time this project has landed there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# A push lap spends most of its time at full throttle or on the brakes. A
# cruise lap coasts: partial throttle, no braking, no load. Calibrate on real
# data before trusting the absolute numbers; the RANKING is what P4 uses.
PUSH_THRESHOLD = 0.60
CRUISE_THRESHOLD = 0.35


def lap_effort(tel: pd.DataFrame) -> dict:
    """Effort indicators for one lap's raw telemetry."""
    out = {"throttle_full_frac": np.nan, "brake_frac": np.nan,
           "coast_frac": np.nan, "effort_index": np.nan}
    if tel is None or not len(tel):
        return out

    thr = pd.to_numeric(tel.get("throttle_pct"), errors="coerce")
    brk = tel.get("brake_on")
    n = len(tel)
    if thr is None or not n:
        return out

    full = float((thr >= 95).sum()) / n
    braking = float(pd.Series(brk).fillna(False).astype(bool).sum()) / n if brk is not None else 0.0
    # Coasting: neither asking for power nor slowing the car. On a push lap
    # this is close to zero — an F1 driver is on one pedal or the other.
    coast = float(((thr < 20) & (~pd.Series(brk).fillna(False).astype(bool))).sum()) / n \
        if brk is not None else float((thr < 20).sum()) / n

    out["throttle_full_frac"] = round(full, 4)
    out["brake_frac"] = round(braking, 4)
    out["coast_frac"] = round(coast, 4)
    # One scalar P4 can filter or weight on. Full throttle and braking both
    # count as working the car; coasting counts against.
    out["effort_index"] = round(float(np.clip(full + braking - coast, 0.0, 1.0)), 4)
    return out


def classify_effort(effort_index: float | None) -> str:
    if effort_index is None or not np.isfinite(effort_index):
        return "unknown"
    if effort_index >= PUSH_THRESHOLD:
        return "push"
    if effort_index <= CRUISE_THRESHOLD:
        return "cruise"
    return "moderate"
