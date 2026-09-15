"""P2-7 — setup proxies, with the imputation deferred from P1.

F1 teams never publish setups, so the simulator's sliders have nothing real to
regress against. What IS published is where the car was fast and where it was
slow, and that carries the setup's fingerprint:

    trap speed on the longest straight   -> low drag
    speed through high-speed corners     -> high downforce
    the ratio between them               -> aero balance

The trap readings live in the lap table already, but with real gaps —
SpeedI1 is 20.8% null and SpeedST 13.4% (measured in reconnaissance). P1
stored them verbatim with a `*_missing` flag rather than filling them,
because filling needs to know WHERE each trap sits on the circuit, and that
only exists once segmentation has run.

Now it has. Each trap is imputed from the telemetry at its own location, and
every imputed value keeps its flag so the model can tell a measurement from a
reconstruction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRAPS = ("i1", "i2", "fl", "st")


def _speed_at(tel: pd.DataFrame, distance_m: float) -> float:
    d = tel["distance_m"].to_numpy(dtype=float)
    v = pd.to_numeric(tel["speed_kph"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(d) & np.isfinite(v)
    if ok.sum() < 2:
        return np.nan
    return float(np.interp(distance_m, d[ok], v[ok]))


def longest_straight(segments: list[dict]) -> dict | None:
    st = [s for s in segments if s["kind"] == "straight"]
    return max(st, key=lambda s: s["length_m"]) if st else None


def impute_traps(lap: pd.Series, tel: pd.DataFrame, segments: list[dict],
                 sector_boundaries_m: list[float], lap_length_m: float) -> dict:
    """Fill missing trap readings from telemetry at the trap's own location."""
    out: dict = {}
    straight = longest_straight(segments)

    # Where each trap physically sits. The intermediates are the sector
    # split points; the finish-line trap is the line; the speed trap is on
    # the longest straight, so its location is that segment's fastest point.
    where = {
        "i1": sector_boundaries_m[0] if len(sector_boundaries_m) > 0 else None,
        "i2": sector_boundaries_m[1] if len(sector_boundaries_m) > 1 else None,
        "fl": lap_length_m,
        "st": None,
    }

    for trap in TRAPS:
        col = f"speed_{trap}_kph"
        val = lap.get(col)
        missing = pd.isna(val)
        imputed = False

        if missing:
            if trap == "st" and straight is not None:
                seg = tel[(tel["distance_m"] >= straight["start_m"]) &
                          (tel["distance_m"] < min(straight["end_m"], lap_length_m))]
                v = pd.to_numeric(seg.get("speed_kph"), errors="coerce")
                val = float(v.max()) if len(v) and v.notna().any() else np.nan
            elif where.get(trap) is not None:
                val = _speed_at(tel, float(where[trap]))
            imputed = bool(pd.notna(val))

        out[col] = float(val) if pd.notna(val) else np.nan
        out[f"speed_{trap}_imputed"] = imputed

    # --- derived proxies --------------------------------------------------
    st_v, i2_v = out.get("speed_st_kph"), out.get("speed_i2_kph")
    # Aero balance: a low-drag car is fast in a straight line and gives that
    # back in the corners, so the ratio separates wing level even though
    # neither number is a wing angle.
    out["aero_balance_proxy"] = (round(st_v / i2_v, 4)
                                 if st_v and i2_v and np.isfinite(st_v)
                                 and np.isfinite(i2_v) and i2_v > 0 else np.nan)
    out["drag_proxy_kph"] = st_v
    out["downforce_proxy_kph"] = i2_v
    out["traps_imputed_count"] = int(sum(out[f"speed_{t}_imputed"] for t in TRAPS))
    return out
