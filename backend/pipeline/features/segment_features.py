"""P2-5 — per (lap x segment) aggregation. This is the model's row.

The target is per-segment time, so the row has to be per segment. Everything
here is integrated over the RAW samples inside the segment's distance window —
never a resampled grid (DECISIONS.md D1). Segment time in particular is the
label: computing it from interpolated samples would mean training on times
nobody drove.

Segment time is obtained by integrating dt over the samples in the window
rather than differencing the window's endpoints, so a sample gap inside the
segment does not silently become extra time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

G = 9.81


def _finite(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def segment_slice(tel: pd.DataFrame, start_m: float, end_m: float,
                  lap_length_m: float) -> pd.DataFrame:
    """Samples inside a segment, handling one that wraps start-finish."""
    d = tel["distance_m"].to_numpy(dtype=float)
    if end_m > lap_length_m:                       # wrapped segment
        m = (d >= start_m) | (d < end_m - lap_length_m)
    else:
        m = (d >= start_m) & (d < end_m)
    return tel[m]


def features_for_segment(seg_tel: pd.DataFrame, seg: dict,
                         lap_length_m: float) -> dict:
    out: dict = {
        "segment_index": seg["index"],
        "segment_kind": seg["kind"],
        "segment_sector": seg.get("sector"),
        "segment_length_m": seg["length_m"],
        "segment_min_radius_m": seg.get("min_radius_m"),
        "segment_direction": seg.get("direction"),
        "segment_is_kink": seg["kind"] == "kink",
        "n_samples": int(len(seg_tel)),
    }
    if not len(seg_tel):
        return out

    v = pd.to_numeric(seg_tel.get("speed_kph"), errors="coerce").to_numpy(dtype=float)
    thr = pd.to_numeric(seg_tel.get("throttle_pct"), errors="coerce").to_numpy(dtype=float)
    brk = pd.Series(seg_tel.get("brake_on")).fillna(False).astype(bool).to_numpy()
    t = pd.to_numeric(seg_tel.get("session_time_s"), errors="coerce").to_numpy(dtype=float)
    d = seg_tel["distance_m"].to_numpy(dtype=float)

    # --- the label -------------------------------------------------------
    # Integrate the sample intervals rather than differencing the endpoints:
    # a dropout inside the segment would otherwise be charged to the driver
    # as time spent.
    if np.isfinite(t).sum() >= 2:
        dt = np.diff(t[np.isfinite(t)])
        dt = dt[(dt > 0) & (dt < 2.0)]
        out["segment_time_s"] = round(float(dt.sum()), 4) if len(dt) else np.nan
        out["segment_time_gap_s"] = round(float(np.diff(t[np.isfinite(t)]).sum()
                                                - dt.sum()), 4) if len(dt) else np.nan
    else:
        out["segment_time_s"] = np.nan
        out["segment_time_gap_s"] = np.nan

    # --- speed -----------------------------------------------------------
    vf = _finite(v)
    if len(vf):
        out.update({
            "speed_mean_kph": round(float(vf.mean()), 2),
            "speed_min_kph": round(float(vf.min()), 2),
            "speed_max_kph": round(float(vf.max()), 2),
            "speed_entry_kph": round(float(vf[0]), 2),
            "speed_exit_kph": round(float(vf[-1]), 2),
        })

    # --- pedals ----------------------------------------------------------
    if np.isfinite(thr).any():
        out["throttle_mean_pct"] = round(float(np.nanmean(thr)), 2)
        out["throttle_full_frac"] = round(float((thr >= 95).mean()), 4)
        # Where in the segment the driver got back to full throttle: the
        # single most setup-sensitive thing a driver does in a corner.
        full = np.flatnonzero(thr >= 95)
        out["throttle_on_frac"] = (round(float((d[full[0]] - d[0])
                                               / max(seg["length_m"], 1e-6)), 4)
                                   if len(full) else np.nan)
    if brk.any():
        out["brake_frac"] = round(float(brk.mean()), 4)
        first = int(np.argmax(brk))
        out["brake_point_frac"] = round(float((d[first] - d[0])
                                              / max(seg["length_m"], 1e-6)), 4)
    else:
        out["brake_frac"] = 0.0
        out["brake_point_frac"] = np.nan

    if "drs_raw" in seg_tel:
        drs = pd.to_numeric(seg_tel["drs_raw"], errors="coerce").to_numpy(dtype=float)
        # FastF1 DRS codes: 10, 12, 14 mean the flap is open.
        out["drs_open_frac"] = round(float(np.isin(drs, (10, 12, 14)).mean()), 4)

    # --- loads -----------------------------------------------------------
    r = seg.get("min_radius_m") or np.inf
    if len(vf) and np.isfinite(r) and r > 0:
        out["lateral_g_peak"] = round(float((vf.max() / 3.6) ** 2 / r / G), 3)
        out["lateral_g_apex"] = round(float((vf.min() / 3.6) ** 2 / r / G), 3)
    if np.isfinite(t).sum() >= 3 and len(vf) >= 3:
        ms = v / 3.6
        dv = np.diff(ms)
        dtt = np.diff(t)
        ok = np.isfinite(dv) & np.isfinite(dtt) & (dtt > 0) & (dtt < 2.0)
        if ok.any():
            acc = dv[ok] / dtt[ok]
            out["long_g_accel_max"] = round(float(acc.max() / G), 3)
            out["long_g_brake_max"] = round(float(-acc.min() / G), 3)
    return out


def features_for_lap(tel: pd.DataFrame, segments: list[dict],
                     lap_length_m: float) -> pd.DataFrame:
    rows = [features_for_segment(segment_slice(tel, s["start_m"], s["end_m"],
                                               lap_length_m), s, lap_length_m)
            for s in segments]
    return pd.DataFrame(rows)
